import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import time
import pickle

from .cuQSL_cat import TorchRegularInterpolator
from .cuQSL_sph import qsl_solver_sph, Spherical_curl, Spherical_Grad_Vec

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

solver = qsl_solver_sph.load(load_file)
Brtp = solver.Brtp
nrtp = Brtp.shape[:-1]
Nr,Nt,Np = nrtp
rtp_range = solver.rtp_range
stretched = np.array(solver.streched)
rrange,trange,prange = rtp_range
rr = np.linspace(rrange[0],rrange[1],Nr) if not stretched[0] else np.geomspace(rrange[0],rrange[1],Nr)
tt = np.linspace(trange[0],trange[1],Nt) if not stretched[1] else np.geomspace(trange[0],trange[1],Nt)
pp = np.linspace(prange[0],prange[1],Np) if not stretched[2] else np.geomspace(prange[0],prange[1],Np)
rtp_list = [rr,tt,pp]
rtp = np.stack(np.meshgrid(rr,tt,pp,indexing='ij'),axis=-1)
usr_kwargs = solver.usr_kwargs
max_batch = usr_kwargs.get('max_batch',50000000)
points = solver.points
bhat = Brtp/np.linalg.norm(Brtp,axis=-1)[:,:,:,None]
jacb = Spherical_Grad_Vec(bhat,rtp)
rotB = Spherical_curl(Brtp,rtp)

bhat = torch.from_numpy(bhat).float().to(device)
jacb = torch.from_numpy(jacb).float().to(device)
rotB = torch.from_numpy(rotB).float().to(device)
Brtp = torch.from_numpy(Brtp).float().to(device)

gpu_interp_bhat = TorchRegularInterpolator(bhat)
gpu_interp_jacb = TorchRegularInterpolator(jacb)
gpu_interp_Jrtp = TorchRegularInterpolator(rotB)
gpu_interp_Brtp = TorchRegularInterpolator(Brtp)

rtp_min = torch.from_numpy(np.array([rrange[0],trange[0],prange[0]])).to(device)
rtp_max = torch.from_numpy(np.array([rrange[1],trange[1],prange[1]])).to(device)
nrtp = np.array([Nr,Nt,Np])
# drtp = np.stack([
#     np.gradient(rtp[:,:,:,0],axis=0),
#     np.gradient(rtp[:,:,:,1],axis=1),
#     np.gradient(rtp[:,:,:,2],axis=2)
# ],axis=-1)
# gpu_interp_drtp = TorchRegularInterpolator(torch.from_numpy(drtp).float().to(device))
qs = np.ones(3,dtype=np.float32)
for dim in range(3):
    if stretched[dim]:
        qs[dim] = (rtp_list[dim][2]-rtp_list[dim][1])/(rtp_list[dim][1]-rtp_list[dim][0])

# =============================================================== #
# 2. Define the integrator
# =============================================================== #

def scaling_rtp(rtp):
    if not isinstance(rtp,torch.Tensor):
        rtp = torch.from_numpy(rtp).float().to(device)
    else:
        rtp = rtp.float().to(device)
    if not np.any(stretched):
        ret = 2*(rtp-rtp_min[None,:])/(rtp_max[None,:]-rtp_min[None,:])-1
    else:
        ret = 2*(rtp-rtp_min[None,:])/(rtp_max[None,:]-rtp_min[None,:])-1
        for dim in range(3):
            if stretched[dim] and (qs[dim]!=1.):
                ret[:,dim] = torch.log(1+(rtp[:,dim]-rtp_min[dim])/(rtp_list[dim][1]-rtp_min[dim])*(qs[dim]-1))/np.log(qs[dim])
                ret[:,dim] = 2*ret[:,dim]/(nrtp[dim]-1)-1
    return ret

def descaling_rtp(rtp0):
    if not isinstance(rtp0, torch.Tensor):
        rtp0 = torch.from_numpy(rtp0).float().to(device)
    else:
        rtp0 = rtp0.float().to(device)
    if not np.any(stretched):
        ret = (rtp0+1)/2*(rtp_max[None,:]-rtp_min[None,:])+rtp_min[None,:]
    else:
        ret = (rtp0+1)/2*(rtp_max[None,:]-rtp_min[None,:])+rtp_min[None,:]
        for dim in range(3):
            if stretched[dim] and (qs[dim]!=1.):
                ret[:,dim] = (rtp0[:,dim]+1)/2*(nrtp[dim]-1)
                ret[:,dim] = (torch.exp(ret[:,dim]*np.log(qs[dim]))-1.)/(qs[dim]-1.)*(rtp_list[dim][1]-rtp_list[dim][0])+rtp_list[dim][0]
    return ret

def qsl_rfunc(s,xuv):
    eps = 1e-8
    if not isinstance(xuv, torch.Tensor):
        xuv = torch.from_numpy(xuv).float().to(device)
    else:
        xuv = xuv.float().to(device)
    rtp  = xuv[:,:3]
    Uvec = xuv[:,3:6]
    Vvec = xuv[:,6:]
    rtp0 = scaling_rtp(rtp)
    # print(xyz0.shape)
    bhat = gpu_interp_bhat(rtp0,device=device)
    jacb = gpu_interp_jacb(rtp0,device=device)
    ret  = torch.zeros_like(xuv)
    ret[:,:3 ] = bhat
    ret[:,1] *= 1/rtp[:,0]
    ret[:,2] *= 1/(rtp[:,0]*torch.sin(torch.where(rtp[:,1]<eps,eps,rtp[:,1])))
    ret[:,3:6] = Uvec[:,0:1]*jacb[:,0]+Uvec[:,1:2]*jacb[:,1]+Uvec[:,2:3]*jacb[:,2]
    ret[:,6: ] = Vvec[:,0:1]*jacb[:,0]+Vvec[:,1:2]*jacb[:,1]+Vvec[:,2:3]*jacb[:,2]
    return ret

class rkf45_stepper():
    MAX_ITERATIONS = 100
    def __init__(self,rfunc,**kwargs):
        self.atol = kwargs.get('atol',1e-4)
        self.rtol = kwargs.get('rtol',1e-4)
        self.hmax = kwargs.get('max_step',10*(rr[-1]-rr[0]))
        self.hmin = kwargs.get('min_step',0.2*(rr[1]-rr[0]))
        self.rfunc = rfunc

    def stepping(self, t, y, dt):
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
        dt_new = torch.where(scaling<0.5,0.5,scaling)*torch.abs(dt)
        dt_new = torch.where(dt_new>self.hmax,self.hmax,dt_new)
        dt_new = torch.where(dt_new<self.hmin,self.hmin,dt_new)
        dt_new*= torch.sign(dt)
        
        return y4,y5,err,dt_new,scaling

    def __call__(self, t, y, dt):
        t_new  = t.clone()
        y_new  = y.clone()
        dt_new = dt.clone()
        need_to_stop = False
        cond = torch.ones(len(y_new),dtype=torch.bool,device=y_new.device)
        iteration = 0
        while not need_to_stop:
            iteration += 1
            y4,y5,err,dti,scaling = self.stepping(t[cond],y[cond],dt[cond])
            if iteration>=self.MAX_ITERATIONS:
                need_to_stop = True
                y_new[cond] = y5
                dt_new[cond] = dti
            else:
                cond_new = cond.clone()
                if torch.any(err<=1):
                    y_new[cond] = torch.where(err[:,None]<=1,y5,y_new[cond])
                    dt_new[cond] = torch.where(err[:,None]<=1,dti,dt_new[cond])
                if torch.any(err>1):
                    dt_new[cond] = torch.where(err[:,None]>1,dti,dt_new[cond])
                cond_new[cond] = cond_new[cond] & (err>1)
                if torch.any(dt_new[cond]<=self.hmin):
                    cond_new[cond] = cond_new[cond] & (dt_new[cond,0]>self.hmin)
                    y_new[cond] = torch.where(dt_new[cond]<=self.hmin,y5,y_new[cond])
                    dt_new[cond] = torch.where(dt_new[cond]<self.hmin,self.hmin,dt_new[cond])
                y = y_new.clone()
                dt = dt_new.clone()
                cond = cond_new.clone()
                need_to_stop = (not torch.any(cond))
        t_new = t+dt_new
        return t_new,y_new,dt_new           
            
def is_next_boundary(xuv,ds,stepper=None):
    if not isinstance(xuv, torch.Tensor):
        xuv = torch.from_numpy(xuv).float().to(device)
    else:
        xuv = xuv.float().to(device)
    rtp   = xuv[:,:3]
    rtp0  = scaling_rtp(rtp)
    bhat  = gpu_interp_bhat(rtp0,device=device)
    if stepper is None:
        rtp1  = rtp+ds*bhat
    else:
        rtp1 = stepper(ds,xuv,ds)[1][:,:3]
    flag = torch.zeros(len(xuv),dtype=torch.int8) # 0: not boundary, 1: nan, 2: boundary, 3: bhat is nan
    flag[torch.any(torch.isnan(rtp1),axis=1)] = 1
    flag[torch.any((rtp1<=rtp_min[None,:]) | (rtp1>=rtp_max[None,:]),axis=1)] = 2
    # print(f'bhat shape: {bhat.shape}')
    flag[torch.any(torch.isnan(bhat),axis=1)] = 3
    return flag.to(device)

def is_boundary(xuv):
    if not isinstance(xuv,torch.Tensor):
        xuv = torch.from_numpy(xuv).float().to(device)
    else:
        xuv = xuv.float().to(device)
    rtp = xuv[:,:3]
    check = torch.any(torch.isnan(rtp) | (rtp<=rtp_min[None,:]) | (rtp>=rtp_max[None,:]),dim=1)
    return check.to(device)

def process_boundary(xuv,ds):
    eps=1.e-8
    if not isinstance(xuv,torch.Tensor):
        xuv = torch.from_numpy(xuv).float().to(device)
    else:
        xuv = xuv.float().to(device)
    rtp  = xuv[:,:3]
    rtp0 = scaling_rtp(rtp)
    bhat = gpu_interp_bhat(rtp0,device=device)
    ds_new = torch.abs(ds)
    metric = torch.stack([torch.ones_like(rtp[:,0]),rtp[:,0],(rtp[:,0]*torch.sin(rtp[:,1]))],dim=1)
    drtp = torch.cat(
        [
            (rtp_min[None,:]-rtp)*(-1),
            (rtp_max[None,:]-rtp),
        ],
        dim=1
    )
    distance = torch.cat(
        [
            (rtp_min[None,:]-rtp)*metric/(bhat*torch.sign(ds)),
            (rtp_max[None,:]-rtp)*metric/(bhat*torch.sign(ds)),
        ],
        dim=1
    )
    distance = torch.where((torch.abs(distance)<eps) | (drtp<0),0,distance)
    distance = torch.where(distance<0,1.0,distance)
    ds_new = distance.min(dim=1)[0][:,None]*torch.sign(ds)
    xuvf = xuv+ds_new*qsl_rfunc(torch.zeros_like(ds),xuv)
    check = torch.argmin(distance,dim=1)
    xuvf[check==0,0] = rtp_min[0]
    xuvf[check==1,1] = rtp_min[1]
    xuvf[check==2,2] = rtp_min[2]
    xuvf[check==3,0] = rtp_max[0]
    xuvf[check==4,1] = rtp_max[1]
    xuvf[check==5,2] = rtp_max[2]
    xuvf = torch.where(torch.abs(torch.fmod(rtp[:,1:2],torch.pi))<eps,xuv,xuvf)
    ds_new = torch.where(torch.abs(torch.fmod(rtp[:,1:2],torch.pi))<eps,0,ds_new)
    return xuvf.float(),ds_new.float()

def twist_number_integral(xuv,ds):
    if not isinstance(xuv,torch.Tensor):
        xuv = torch.from_numpy(xuv).float().to(device)
    else:
        xuv = xuv.float().to(device)
    rtp = xuv[:,:3]
    rtp0 = scaling_rtp(rtp)
    Brtp = gpu_interp_Brtp(rtp0,device=device)
    Jrtp = gpu_interp_Jrtp(rtp0,device=device)
    dTw   = torch.sum(Brtp*Jrtp,dim=-1, keepdim=True)/(4*torch.pi*torch.sum(Brtp**2,dim=-1, keepdim=True))*ds
    return dTw
    

def initializeUV(rtp, **kwargs):
    print_interval = kwargs.get('print_interval', 100)
    if not isinstance(rtp,torch.Tensor):
        rtp = torch.from_numpy(rtp).float().to(device)
    else:
        rtp = rtp.float().to(device)
    rtp0  = scaling_rtp(rtp)
    bhat  = gpu_interp_bhat(rtp0,device=device)
    err   = kwargs.get('err', 1e-10)
    Vvec  = torch.zeros_like(rtp)
    Uvec  = torch.zeros_like(rtp)
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
    # else:
    #     pass
    Uvec = torch.cross(bhat, Vvec, dim=-1)
    Vn   = torch.norm(Vvec,dim=-1)
    Un   = torch.norm(Uvec,dim=-1)
    Vvec = torch.where(Vn[:,None]>=err,Vvec/Vn[:,None],Vvec)
    Uvec = torch.where(Un[:,None]>=err,Uvec/Un[:,None],Uvec)
    xuv = torch.cat([rtp,Uvec,Vvec],dim=-1)
    return xuv

def Line_Integration(rtp,**kwargs):
    time0 = time.time()
    if not isinstance(rtp,torch.Tensor):
        print('rtp is not a torch.Tensor')
        rtp = torch.from_numpy(rtp).float().to(device)
    else:
        rtp = rtp.float().to(device)
    ds = kwargs.get('step_size',2*(rr[1]-rr[0]))
    Ns = kwargs.get('max_steps',10000)
    IP = kwargs.get('is_print',False)
    PI = kwargs.get('print_interval', 100)
    idx = kwargs.get('prob_idx', None)
    xuv0 = initializeUV(rtp,**kwargs)
    xuvF = xuv0.clone()
    xuvB = xuv0.clone()
    stepper = rkf45_stepper(qsl_rfunc,**kwargs)
    Tw = torch.zeros((len(xuv0),1),dtype=torch.float32).to(device)
    # forward integration
    sf = torch.zeros((len(xuvF),1),dtype=torch.float32).to(device)
    ds_f = ds*torch.ones((len(xuvF),1),dtype=torch.float32).to(device)
    if idx is not None:
        hist_f  = dict(xuv=[xuvF[idx:idx+1].detach().cpu().clone()],ds=[ds_f[idx:idx+1].detach().cpu().clone()])
    # check_stop = is_boundary(xuvF)
    check_stop = torch.zeros((len(xuvF)),dtype=torch.bool).to(device)
    if IP:
        print(f"### Forward Integration ###")
    for i in range(Ns):
        if torch.all(check_stop):
            break
        xuvI = xuvF.clone()[~check_stop]
        if i>=Ns-1:
            if IP:
                print(f'Forward Integration over {Ns} iterations')
            break
        # check boundary and process boundary
        check_next_boundary = is_next_boundary(xuvI,ds_f[~check_stop],stepper)
        if torch.any(check_next_boundary==2):
            xuvI[check_next_boundary==2],ds_f[~check_stop][check_next_boundary==2] = process_boundary(xuvI[check_next_boundary==2],ds_f[~check_stop][check_next_boundary==2])
            sf[~check_stop][check_next_boundary==2] += ds_f[~check_stop][check_next_boundary==2]
            Tw[~check_stop][check_next_boundary==2] += twist_number_integral(xuvI[check_next_boundary==2],ds_f[~check_stop][check_next_boundary==2])    
            xuvF[~check_stop] = xuvI
        check_boundary = is_boundary(xuvI)
        # check_stop[~check_stop] = (check_boundary & (check_next_boundary!=0))
        check_stop[~check_stop] = (check_next_boundary!=0)
        if IP:
            if (i==0) or (i==Ns-1) or ((i+1)%PI==0 or torch.all(check_stop)):
                ti = time.time()
                print(f" ## Iteration: {i+1:8d} | Stop nums: {len(check_stop[check_stop]):8d} | Task nums: {len(check_stop)} | Wall time: {(ti-t0)/60:8.3f} min")
        if torch.all(check_stop):
            if idx is not None:
                hist_f['xuv'].append(xuvF[idx:idx+1].detach().cpu().clone())
                hist_f['ds'].append(ds_f[idx:idx+1].detach().cpu().clone())
            break
        # forward stepping
        xuvI = xuvF.clone()[~check_stop]
        sf[~check_stop],xuvI,ds_f[~check_stop] = stepper(sf[~check_stop],xuvI,ds_f[~check_stop])
        xuvF[~check_stop] = xuvI
        Tw[~check_stop] += twist_number_integral(xuvI,ds_f[~check_stop])
        if idx is not None:
            hist_f['xuv'].append(xuvF[idx:idx+1].detach().cpu().clone())
            hist_f['ds'].append(ds_f[idx:idx+1].detach().cpu().clone())
    # print(f'i={i}')
    # backward integration
    sb = torch.zeros((len(xuvB),1),dtype=torch.float32).to(device)
    ds_b = -ds*torch.ones((len(xuvB),1),dtype=torch.float32).to(device)
    # check_stop = is_boundary(xuvB)
    check_stop = torch.zeros((len(xuvF)),dtype=torch.bool).to(device)
    if IP:
        print(f"### Backward Integration ###")
    if idx is not None:
        hist_b = dict(xuv=[],ds=[])
    for i in range(Ns):
        if torch.all(check_stop):
            break
        xuvI = xuvB.clone()[~check_stop]
        if i>=Ns-1:
            if IP:
                print(f'Backward Integration over {Ns} iterations')
            break
        # check boundary and process boundary
        check_next_boundary = is_next_boundary(xuvI,ds_b[~check_stop],stepper)
        if torch.any(check_next_boundary==2):
            xuvI[check_next_boundary==2],ds_b[~check_stop][check_next_boundary==2] = process_boundary(xuvI[check_next_boundary==2],ds_b[~check_stop][check_next_boundary==2])
            sb[~check_stop][check_next_boundary==2] += ds_b[~check_stop][check_next_boundary==2]
            Tw[~check_stop][check_next_boundary==2] -= twist_number_integral(xuvI[check_next_boundary==2],ds_b[~check_stop][check_next_boundary==2])    
        xuvB[~check_stop] = xuvI
        check_boundary = is_boundary(xuvI)
        # check_stop[~check_stop] = (check_boundary & (check_next_boundary!=0))
        check_stop[~check_stop] = (check_next_boundary!=0)
        if IP:
            if (i==0) or (i==Ns-1) or ((i+1)%PI==0 or torch.all(check_stop)):
                ti = time.time()
                print(f" ## Iteration: {i+1:8d} | Stop nums: {len(check_stop[check_stop]):8d} | Task nums: {len(check_stop)} | Wall time: {(ti-t0)/60:8.3f} min")
        if torch.all(check_stop):
            if idx is not None:
                hist_b['xuv'].append(xuvB[idx:idx+1].detach().cpu().clone())
                hist_b['ds'].append(ds_b[idx:idx+1].detach().cpu().clone())
            break
        # backward stepping
        xuvI = xuvB.clone()[~check_stop]
        sb[~check_stop],xuvI,ds_b[~check_stop] = stepper(sb[~check_stop],xuvI,ds_b[~check_stop])
        xuvB[~check_stop] = xuvI
        Tw[~check_stop] += twist_number_integral(xuvI,ds_b[~check_stop])
        if idx is not None:
            hist_b['xuv'].append(xuvB[idx:idx+1].detach().cpu().clone())
            hist_b['ds'].append(ds_b[idx:idx+1].detach().cpu().clone())
    hist = dict() if idx is None else dict(
        xuv = torch.cat(hist_f['xuv'][::-1]+hist_b['xuv'],dim=0),
        ds  = torch.cat(hist_f['ds'][::-1]+hist_b['ds'],dim=0)
    )
    return xuvF,xuvB,sf,sb,Tw,hist

# =============================================================== #
# 3. Define the QSL calculator
# =============================================================== #

def qsl_calculator(rtp,**kwargs):
    if not isinstance(rtp,torch.Tensor):
        rtp = torch.from_numpy(rtp).float().to(device)
    else:
        rtp = rtp.float().to(device)
    xuvF,xuvB,sf,sb,Tw,hist = Line_Integration(rtp,**kwargs)
    B0 = torch.norm(gpu_interp_Brtp(scaling_rtp(rtp),device=device),dim=-1)
    BF = torch.norm(gpu_interp_Brtp(scaling_rtp(xuvF[:,:3]),device=device),dim=-1)
    BB = torch.norm(gpu_interp_Brtp(scaling_rtp(xuvB[:,:3]),device=device),dim=-1)
    UF = xuvF[:,3:6]
    UB = xuvB[:,3:6]
    VF = xuvF[:,6:9]
    VB = xuvB[:,6:9]
    b0 = gpu_interp_bhat(scaling_rtp(rtp),device=device)
    bf = gpu_interp_bhat(scaling_rtp(xuvF[:,:3]),device=device)
    bb = gpu_interp_bhat(scaling_rtp(xuvB[:,:3]),device=device)
    UF = UF-torch.sum(bf*UF,dim=1,keepdim=True)*bf
    VF = VF-torch.sum(bf*VF,dim=1,keepdim=True)*bf
    UB = UB-torch.sum(bb*UB,dim=1,keepdim=True)*bb
    VB = VB-torch.sum(bb*VB,dim=1,keepdim=True)*bb
    Det = B0**2/(BF*BB)
    UF = UF.double()
    UB = UB.double()
    VF = VF.double()
    VB = VB.double()
    Norm = torch.sum(UF*UF,dim=-1)*torch.sum(VB*VB,dim=-1)+torch.sum(UB*UB,dim=-1)*torch.sum(VF*VF,dim=-1)-2*torch.sum(UF*VF,dim=-1)*torch.sum(UB*VB,dim=-1)
    logQ = torch.log10(Norm).float()-torch.log10(Det)
    # logQ = torch.where(logQ<np.log10(2),np.log10(2),logQ)
    # logQ = -logQ if torch.sum(xyz*b0)<0 else logQ
    length = torch.abs(sf)+torch.abs(sb)
    if torch.any(torch.isinf(logQ)):
        position = torch.where(torch.isinf(logQ))
        print(f"position: {position[0][0]}")
        print(f"Det: {Det[position][0]} | Norm: {Norm[position][0]}")
        print(f"B0: {B0[position][0]} | BF: {BF[position][0]} | BB: {BB[position][0]}")
        print(f"UF: {UF[position][0]} | VF: {VF[position][0]} | UB: {UB[position][0]} | VB: {VB[position][0]}")
        print(f"B0: {B0[position][0]} | BF: {BF[position][0]} | BB: {BB[position][0]}")
        print(f"b0: {b0[position][0]} | bf: {bf[position][0]} | bb: {bb[position][0]}")
        print(f"x0: {rtp[position][0]} | xf: {xuvF[position][0][:3]} | xb: {xuvB[position][0][:3]}")
        print(f"p0: {scaling_rtp(rtp)[position][0]} | pf: {scaling_rtp(xuvF[:,:3])[position][0]} | pb: {scaling_rtp(xuvB[:,:3])[position][0]}")
        print(f"InitializeUV: {initializeUV(rtp)[position[0][0]]}")
    return logQ,length,Tw,hist

# =============================================================== #
# 4. Compute the QSL
# =============================================================== #
print(f'## Computing the QSL for {len(points)} points',flush=True)

if len(points)<max_batch:
    logQ,length,Tw,hist = qsl_calculator(points, **usr_kwargs)
    logQ = logQ.cpu().numpy()
    length = length.cpu().numpy()
    Tw = Tw.cpu().numpy()
else:
    logQ = []
    length = []
    Tw = []
    for i in range(0,len(points),max_batch):
        i_logQ,i_length,i_Tw,i_hist = qsl_calculator(points[i:i+max_batch], **usr_kwargs)
        logQ.append(i_logQ.cpu().numpy())
        length.append(i_length.cpu().numpy())
        Tw.append(i_Tw.cpu().numpy())
        if not i_hist:
            hist = dict()
        else:
            if i==0:
                hist = dict(xuv=i_hist['xuv'],ds=i_hist['ds'])
            else:
                hist['xuv'].append(i_hist['xuv'])
                hist['ds'].append(i_hist['ds'])
    logQ = np.concatenate(logQ,axis=0)
    length = np.concatenate(length,axis=0)
    Tw = np.concatenate(Tw,axis=0)

save_dict = {
    'logQ': logQ,
    'length': length,
    'Tw': Tw,
    'hist':hist
}

with open(save_file, 'wb') as f:
    pickle.dump(save_dict, f)

t1 = time.time()
print(f'## QSL results saved to {save_file}')
print(f'## Time taken: {(t1-t0)/60:.3f} minutes')
