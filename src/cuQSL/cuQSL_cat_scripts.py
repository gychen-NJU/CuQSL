import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import time
import pickle

from .cuQSL_cat import TorchRegularInterpolator, Cartesian_curl, qsl_solver_cat

# =============================================================== #
# 1. Initialization
# =============================================================== #
t0 = time.time()
parser = argparse.ArgumentParser(description="Compute the QSL for a given point data based on cuda")
parser.add_argument("--load_file", type=str, help="The input file containing the point data")
parser.add_argument("--save_file", type=str, help="The output file containing the QSL results")
parser.add_argument("--device", type=str, help="The device to use for the computation")

args = parser.parse_args()
load_file = args.load_file
save_file = args.save_file
device    = args.device
device    = torch.device(device)

solver = qsl_solver_cat.load(load_file)
Bxyz = solver.Bxyz
grid_xyz = solver.grid_xyz
usr_kwargs = solver.usr_kwargs
max_batch = usr_kwargs.get('max_batch',1000000)
points = solver.points
bhat = Bxyz/np.linalg.norm(Bxyz,axis=-1)[:,:,:,None]
jacb = np.gradient(bhat.transpose(3,0,1,2),axis=(1,2,3))
jacb = np.stack(jacb,axis=0).transpose(2,3,4,0,1)
rotB = Cartesian_curl(Bxyz)

bhat = torch.from_numpy(bhat).float().to(device)
jacb = torch.from_numpy(jacb).float().to(device)
rotB = torch.from_numpy(rotB).float().to(device)
Bxyz = torch.from_numpy(Bxyz).float().to(device)

gpu_interp_bhat = TorchRegularInterpolator(bhat)
gpu_interp_jacb = TorchRegularInterpolator(jacb)
gpu_interp_Jxyz = TorchRegularInterpolator(rotB)
gpu_interp_Bxyz = TorchRegularInterpolator(Bxyz)

xrange = [grid_xyz[0].min(), grid_xyz[0].max()]
yrange = [grid_xyz[1].min(), grid_xyz[1].max()]
zrange = [grid_xyz[2].min(), grid_xyz[2].max()]
xyz_min = torch.from_numpy(np.array([xrange[0],yrange[0],zrange[0]])).to(device)
xyz_max = torch.from_numpy(np.array([xrange[1],yrange[1],zrange[1]])).to(device)
nx,ny,nz = Bxyz.shape[:-1]
nxyz = np.array([nx,ny,nz])
dxyz = (xyz_max-xyz_min).cpu().numpy()/(nxyz-1)

# =============================================================== #
# 2. Define the integrator
# =============================================================== #

def scaling_xyz(xyz):
    if not isinstance(xyz,torch.Tensor):
        xyz = torch.from_numpy(xyz).float().to(device)
    else:
        xyz = xyz.float().to(device)
    ret = 2*(xyz-xyz_min[None,:])/(xyz_max[None,:]-xyz_min[None,:])-1
    return ret

def descaling_xyz(xyz0):
    if not isinstance(xyz0, torch.Tensor):
        xyz0 = torch.from_numpy(xyz0).float().to(device)
    else:
        xyz0 = xyz0.float().to(device)
    ret = (xyz0+1)/2*(xyz_max[None,:]-xyz_min[None,:])+xyz_min[None,:]

def qsl_rfunc(s,xuv):
    if not isinstance(xuv, torch.Tensor):
        xuv = torch.from_numpy(xuv).float().to(device)
    else:
        xuv = xuv.float().to(device)
    xyz  = xuv[:,:3]
    Uvec = xuv[:,3:6]
    Vvec = xuv[:,6:]
    xyz0 = scaling_xyz(xyz)
    # print(xyz0.shape)
    bhat = gpu_interp_bhat(xyz0,device=device)
    jacb = gpu_interp_jacb(xyz0,device=device)
    ret  = torch.zeros_like(xuv)
    ret[:,:3 ] = bhat
    # ret[:,3:6] = (Uvec[:,:,None]*jacb).sum(axis=1)
    # ret[:,6: ] = (Vvec[:,:,None]*jacb).sum(axis=1)
    ret[:,3:6] = Uvec[:,0:1]*jacb[:,0]+Uvec[:,1:2]*jacb[:,1]+Uvec[:,2:3]*jacb[:,2]
    ret[:,6: ] = Vvec[:,0:1]*jacb[:,0]+Vvec[:,1:2]*jacb[:,1]+Vvec[:,2:3]*jacb[:,2]
    return ret

class rkf45_stepper():
    def __init__(self,rfunc,**kwargs):
        self.atol = kwargs.get('atol',1e-6)
        self.rtol = kwargs.get('rtol',1e-6)
        self.hmax = kwargs.get('max_step',1)
        self.hmin = kwargs.get('min_step',0.01)
        self.rfunc = rfunc

    def __call__(self, t, y, dt):
        rfunc = self.rfunc
        k1 = dt*rfunc(t,         y)
        k2 = dt*rfunc(t+1/4*dt,  y+1/4*k1)
        k3 = dt*rfunc(t+3/8*dt,  y+3/32*k1+9/32*k2)
        k4 = dt*rfunc(t+12/13*dt,y+1932/2197*k1-7200/2197*k2+7296/2197*k3)
        k5 = dt*rfunc(t+dt,      y+439/216*k1-8*k2+3680/513*k3-845/4104*k4)
        k6 = dt*rfunc(t+1/2*dt,  y-8/27*k1+2*k2-3544/2565*k3+1859/4104*k4-11/40*k5)
        y4 = y+25/216*k1+1408/2565*k3+2197/4104*k4-1/5*k5+1/4*k6
        y5 = y+16/135*k1+6656/12825*k3+128/2565*k4-2197/7200*k5+1/4*k6
        err = torch.max(torch.abs(y4-y5)/(self.atol+self.rtol*torch.abs(y4)),dim=1)[0]
        scaling = 0.84*(1/err[:,None])**0.25
        dt_new = torch.where(scaling>2,2,scaling)*torch.abs(dt)
        dt_new = torch.where(dt_new>self.hmax,self.hmax,dt_new)*torch.sign(dt)
        if torch.all(err<=1):
            t_new = t+dt
            return t_new,y5,dt_new
        elif torch.all(err>1) and torch.all(dt<=self.hmin):
            return t+dt,y5,dt
        else:
            cond = torch.where(err>1)
            dt[cond] = torch.where(scaling[cond]<0.5,0.5,scaling[cond])*dt[cond]
            t_new = (t+dt).clone()
            y_new = y5.clone()
            dt_new = dt.clone()
            t_new[cond],y_new[cond],dt_new[cond] = self(t[cond],y[cond],dt[cond])
            return t_new,y_new,dt_new
            
def is_next_boundary(xuv,ds):
    if not isinstance(xuv, torch.Tensor):
        xuv = torch.from_numpy(xuv).float().to(device)
    else:
        xuv = xuv.float().to(device)
    xyz   = xuv[:,:3]
    xyz0  = scaling_xyz(xyz)
    bhat  = gpu_interp_bhat(xyz0,device=device)
    xyz1  = xyz+ds*bhat
    flag = torch.zeros(len(xuv),dtype=torch.int8) # 0: not boundary, 1: nan, 2: boundary, 3: bhat is nan
    flag[torch.any(torch.isnan(xyz1),axis=1)] = 1
    flag[torch.any((xyz1<=xyz_min[None,:]) | (xyz1>=xyz_max[None,:]),axis=1)] = 2
    # print(f'bhat shape: {bhat.shape}')
    flag[torch.any(torch.isnan(bhat),axis=1)] = 3
    return flag.to(device)

def is_boundary(xuv):
    if not isinstance(xuv,torch.Tensor):
        xuv = torch.from_numpy(xuv).float().to(device)
    else:
        xuv = xuv.float().to(device)
    xyz = xuv[:,:3]
    check = torch.any(torch.isnan(xyz) | (xyz<=xyz_min[None,:]) | (xyz>=xyz_max[None,:]),dim=1)
    return check.to(device)

def process_boundary(xuv,ds):
    if not isinstance(xuv,torch.Tensor):
        xuv = torch.from_numpy(xuv).float().to(device)
    else:
        xuv = xuv.float().to(device)
    xyz  = xuv[:,:3]
    xyz0 = scaling_xyz(xyz)
    bhat = gpu_interp_bhat(xyz0,device=device)
    ds_new = torch.abs(ds)
    distance = torch.cat(
        [
            (xyz_min[None,:]-xyz)/(bhat*torch.sign(ds)),
            (xyz_max[None,:]-xyz)/(bhat*torch.sign(ds)),
        ],
        dim=1
    )
    distance[distance<0] = distance.max()
    ds_new = distance.min(dim=1)[0][:,None]*torch.sign(ds)
    xuvf = xuv+ds_new*qsl_rfunc(torch.zeros_like(ds),xuv)
    check = torch.argmin(distance,dim=1)
    xuvf[check==0,0] = xyz_min[0]
    xuvf[check==1,1] = xyz_min[1]
    xuvf[check==2,2] = xyz_min[2]
    xuvf[check==3,0] = xyz_max[0]
    xuvf[check==4,1] = xyz_max[1]
    xuvf[check==5,2] = xyz_max[2]
    return xuvf.float(),ds_new.float()

def twist_number_integral(xuv,ds):
    if not isinstance(xuv,torch.Tensor):
        xuv = torch.from_numpy(xuv).float().to(device)
    else:
        xuv = xuv.float().to(device)
    xyz = xuv[:,:3]
    xyz0 = scaling_xyz(xyz)
    Bxyz = gpu_interp_Bxyz(xyz0,device=device)
    Jxyz = gpu_interp_Jxyz(xyz0,device=device)/torch.from_numpy(dxyz[None,:]).to(device)
    dTw   = torch.sum(Bxyz*Jxyz,dim=-1, keepdim=True)/(4*torch.pi*torch.norm(Bxyz,dim=-1, keepdim=True)**2)*ds
    return dTw
    

def initializeUV(xyz, **kwargs):
    if not isinstance(xyz,torch.Tensor):
        xyz = torch.from_numpy(xyz).float().to(device)
    else:
        xyz = xyz.float().to(device)
    xyz0  = scaling_xyz(xyz)
    bhat  = gpu_interp_bhat(xyz0,device=device)
    err   = kwargs.get('err', 1e-10)
    Vvec  = torch.zeros_like(xyz)
    Uvec  = torch.zeros_like(xyz)
    cond1 = torch.abs(bhat[:,2])<err
    cond2 = torch.abs(bhat[:,0])<err
    cond3 = torch.abs(bhat[:,1])<err
    if torch.any(cond1 & cond2 & ~cond3):
        Vvec[cond1 & cond2 & ~cond3,0] = 1.
        Vvec[cond1 & cond2 & ~cond3,1] = -bhat[cond1 & cond2 & ~cond3,0]/bhat[cond1 & cond2 & ~cond3,1]
        Vvec[cond1 & cond2 & ~cond3,2] = 0.
    if torch.any(cond1 & ~cond2):
        Vvec[cond1 & ~cond2,0] = -bhat[cond1 & ~cond2,1]/bhat[cond1 & ~cond2,0]
        Vvec[cond1 & ~cond2,1] = 1.
        Vvec[cond1 & ~cond2,2] = 0.
    if torch.any(~cond1):
        Vvec[~cond1,0] = 0.
        Vvec[~cond1,1] = 1.
        Vvec[~cond1,2] = -bhat[~cond1,1]/bhat[~cond1,2]
    Uvec = torch.cross(bhat, Vvec, dim=-1)
    Vn   = torch.norm(Vvec,dim=-1)
    Un   = torch.norm(Uvec,dim=-1)
    Vvec = torch.where(Vn[:,None]>=err,Vvec/Vn[:,None],Vvec)
    Uvec = torch.where(Un[:,None]>=err,Uvec/Un[:,None],Uvec)
    xuv = torch.cat([xyz,Uvec,Vvec],dim=-1)
    return xuv

def Line_Integration(xyz,**kwargs):
    if not isinstance(xyz,torch.Tensor):
        print('xyz is not a torch.Tensor')
        xyz = torch.from_numpy(xyz).float().to(device)
    else:
        xyz = xyz.float().to(device)
    ds = kwargs.get('step_size',dxyz.min())
    Ns = kwargs.get('max_steps',1000000)
    IP = kwargs.get('is_print',False)
    xuv0 = initializeUV(xyz,**kwargs)
    xuvF = xuv0.clone()
    xuvB = xuv0.clone()
    stepper = rkf45_stepper(qsl_rfunc,**kwargs)
    Tw = torch.zeros((len(xuv0),1),dtype=torch.float32).to(device)
    # forward integration
    sf = torch.zeros((len(xuvF),1),dtype=torch.float32).to(device)
    ds_f = ds*torch.ones((len(xuvF),1),dtype=torch.float32).to(device)
    # check_stop = is_boundary(xuvF)
    check_stop = torch.zeros((len(xuvF)),dtype=torch.bool).to(device)
    for i in range(Ns):
        if torch.all(check_stop):
            break
        xuvI = xuvF.clone()[~check_stop]
        if i>=Ns-1:
            if IP:
                print(f'Forward Integration over {Ns} iterations')
            break
        # check boundary and process boundary
        check_next_boundary = is_next_boundary(xuvI,ds_f[~check_stop])
        if torch.any(check_next_boundary==2):
            xuvI[check_next_boundary==2],ds_f[~check_stop][check_next_boundary==2] = process_boundary(xuvI[check_next_boundary==2],ds_f[~check_stop][check_next_boundary==2])
            sf[~check_stop][check_next_boundary==2] += ds_f[~check_stop][check_next_boundary==2]
            Tw[~check_stop][check_next_boundary==2] += twist_number_integral(xuvI[check_next_boundary==2],ds_f[~check_stop][check_next_boundary==2])    
        xuvF[~check_stop] = xuvI
        check_boundary = is_boundary(xuvI)
        check_stop[~check_stop] = (check_boundary & (check_next_boundary!=0))
        if torch.all(check_stop):
            break
        # forward stepping
        xuvI = xuvF.clone()[~check_stop]
        sf[~check_stop],xuvI,ds_f[~check_stop] = stepper(sf[~check_stop],xuvI,ds_f[~check_stop])
        xuvF[~check_stop] = xuvI
        Tw[~check_stop] += twist_number_integral(xuvI,ds_f[~check_stop])
    # print(f'i={i}')
    # backward integration
    sb = torch.zeros((len(xuvB),1),dtype=torch.float32).to(device)
    ds_b = -ds*torch.ones((len(xuvB),1),dtype=torch.float32).to(device)
    # check_stop = is_boundary(xuvB)
    check_stop = torch.zeros((len(xuvF)),dtype=torch.bool).to(device)
    for i in range(Ns):
        if torch.all(check_stop):
            break
        xuvI = xuvB.clone()[~check_stop]
        if i>=Ns-1:
            if IP:
                print(f'Backward Integration over {Ns} iterations')
            break
        # check boundary and process boundary
        check_next_boundary = is_next_boundary(xuvI,ds_b[~check_stop])
        if torch.any(check_next_boundary==2):
            xuvI[check_next_boundary==2],ds_b[~check_stop][check_next_boundary==2] = process_boundary(xuvI[check_next_boundary==2],ds_b[~check_stop][check_next_boundary==2])
            sb[~check_stop][check_next_boundary==2] += ds_b[~check_stop][check_next_boundary==2]
            Tw[~check_stop][check_next_boundary==2] -= twist_number_integral(xuvI[check_next_boundary==2],ds_b[~check_stop][check_next_boundary==2])    
        xuvB[~check_stop] = xuvI
        check_boundary = is_boundary(xuvI)
        check_stop[~check_stop] = (check_boundary & (check_next_boundary!=0))
        if torch.all(check_stop):
            break
        # backward stepping
        xuvI = xuvB.clone()[~check_stop]
        sb[~check_stop],xuvI,ds_b[~check_stop] = stepper(sb[~check_stop],xuvI,ds_b[~check_stop])
        xuvB[~check_stop] = xuvI
        Tw[~check_stop] += twist_number_integral(xuvI,ds_b[~check_stop])
    return xuvF,xuvB,sf,sb,Tw

# =============================================================== #
# 3. Define the QSL calculator
# =============================================================== #

def qsl_calculator(xyz,**kwargs):
    if not isinstance(xyz,torch.Tensor):
        xyz = torch.from_numpy(xyz).float().to(device)
    else:
        xyz = xyz.float().to(device)
    xuvF,xuvB,sf,sb,Tw = Line_Integration(xyz,**kwargs)
    B0 = torch.norm(gpu_interp_Bxyz(scaling_xyz(xyz),device=device),dim=-1)
    BF = torch.norm(gpu_interp_Bxyz(scaling_xyz(xuvF[:,:3]),device=device),dim=-1)
    BB = torch.norm(gpu_interp_Bxyz(scaling_xyz(xuvB[:,:3]),device=device),dim=-1)
    UF = xuvF[:,3:6]
    UB = xuvB[:,3:6]
    VF = xuvF[:,6:9]
    VB = xuvB[:,6:9]
    b0 = gpu_interp_bhat(xyz,device=device)
    bf = gpu_interp_bhat(scaling_xyz(xuvF[:,:3]),device=device)
    bb = gpu_interp_bhat(scaling_xyz(xuvB[:,:3]),device=device)
    UF = UF-torch.sum(bf*UF,dim=1,keepdim=True)*bf
    VF = VF-torch.sum(bf*VF,dim=1,keepdim=True)*bf
    UB = UB-torch.sum(bb*UB,dim=1,keepdim=True)*bb
    VB = VB-torch.sum(bb*VB,dim=1,keepdim=True)*bb
    Det = B0**2/(BF*BB)
    Norm = torch.sum(UF*UF,dim=-1)*torch.sum(VB*VB,dim=-1)+torch.sum(UB*UB,dim=-1)*torch.sum(VF*VF,dim=-1)-2*torch.sum(UF*VF,dim=-1)*torch.sum(UB*VB,dim=-1)
    logQ = torch.log10(Norm)-torch.log10(Det)
    # logQ = torch.where(logQ<np.log10(2),np.log10(2),logQ)
    # logQ = -logQ if torch.sum(xyz*b0)<0 else logQ
    length = torch.abs(sf)+torch.abs(sb)
    return logQ,length,Tw

# =============================================================== #
# 4. Compute the QSL
# =============================================================== #
print(f'## Computing the QSL for {len(points)} points',flush=True)

if len(points)<max_batch:
    logQ,length,Tw = qsl_calculator(points)
    logQ = logQ.cpu().numpy()
    length = length.cpu().numpy()
    Tw = Tw.cpu().numpy()
else:
    logQ = []
    length = []
    Tw = []
    for i in range(0,len(points),max_batch):
        i_logQ,i_length,i_Tw = qsl_calculator(points[i:i+max_batch])
        logQ.append(i_logQ.cpu().numpy())
        length.append(i_length.cpu().numpy())
        Tw.append(i_Tw.cpu().numpy())
    logQ = np.concatenate(logQ,axis=0)
    length = np.concatenate(length,axis=0)
    Tw = np.concatenate(Tw,axis=0)

save_dict = {
    'logQ': logQ,
    'length': length,
    'Tw': Tw
}

with open(save_file, 'wb') as f:
    pickle.dump(save_dict, f)

t1 = time.time()
print(f'## QSL results saved to {save_file}')
print(f'## Time taken: {(t1-t0)/60:.3f} minutes')
