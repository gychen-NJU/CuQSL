"""
cuQSL_sph.py
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import pickle
import subprocess
import time
import shutil
from scipy.interpolate import RegularGridInterpolator

def Spherical_curl(Vrtp, rtp):
    """
    Calculate the curl of a vector field in spherical coordinates.

    Args:
        Vrtp: array_like
            The vector field in spherical coordinates.
        rtp: array_like
            The spherical coordinates.

    Returns:
        rotV_rtp: array_like
            The curl of the vector field in spherical coordinates.
    """
    eps = 1.e-10
    rr,tt,pp = rtp.transpose(3,0,1,2)
    Vr,Vt,Vp = Vrtp.transpose(3,0,1,2)
    dr = np.gradient(rr,axis=0)
    dt = np.gradient(tt,axis=1)
    dp = np.gradient(pp,axis=2)
    rotV_r = (np.gradient(np.sin(tt)*Vp,axis=1)/dt-np.gradient(Vt,axis=2)/dp)/(rr*np.sin(np.where(np.abs(tt)<eps,(np.sign(tt)+1)*eps,tt)))
    rotV_t = (np.gradient(Vr,axis=2)/dp/np.sin(np.where(np.abs(tt)<eps,(np.sign(tt)+1)*eps,tt))-np.gradient(rr*Vp,axis=0)/dr)/rr
    rotV_p = (np.gradient(rr*Vt,axis=0)/dr-np.gradient(Vr,axis=1)/dt)/rr
    rotV_rtp = np.stack([rotV_r,rotV_t,rotV_p],axis=-1)
    return rotV_rtp

def Spherical_Grad_Vec(Vrtp,rtp):
    """
    Calculate the gradient of a vector field in spherical coordinates.

    Args:
        Vrtp: array_like
            The vector field in spherical coordinates.
        rtp: array_like
            The spherical coordinates.

    Returns:
        gradV_rtp: array_like
            The gradient of the vector field in spherical coordinates.
    """
    eps = 1.e-10
    rr,tt,pp = rtp.transpose(3,0,1,2)
    Vr,Vt,Vp = Vrtp.transpose(3,0,1,2)
    dr = np.gradient(rr,axis=0)
    dt = np.gradient(tt,axis=1)
    dp = np.gradient(pp,axis=2)
    Jacobi = np.stack(list(np.gradient(Vrtp,axis=(0,1,2))),axis=-2)
    gradV = Jacobi.copy()
    gradV[:,:,:,0,:] = Jacobi[:,:,:,0,:]/dr[:,:,:,None]
    gradV[:,:,:,1,:] = Jacobi[:,:,:,1,:]/((dt*rr)[:,:,:,None])
    gradV[:,:,:,2,:] = Jacobi[:,:,:,2,:]/((dp*rr*np.sin(np.where(np.abs(tt)<eps,(np.sign(tt)+1)*eps,tt)))[:,:,:,None])
    gradV[:,:,:,1,0]-= Vt/rr
    gradV[:,:,:,1,1]+= Vr/rr
    gradV[:,:,:,2,0]-= Vp/rr
    gradV[:,:,:,2,1]-= Vp/np.tan(np.where(tt<eps,eps,tt))/rr
    gradV[:,:,:,2,2]+= Vr/rr+Vt/np.tan(np.where(np.abs(tt)<eps,(np.sign(tt)+1)*eps,tt))/rr
    return gradV

class qsl_solver_sph:
    """
    """
    def __init__(self, Brtp, rtp_range, stretched=[False,False,False]):
        self.Brtp = Brtp
        self.rtp_range = rtp_range
        self.save_name = None
        self.usr_kwargs = {}
        self.points = None
        self.streched = stretched
        self.message = None
        self.info = dict()
    
    def save(self, file_name='cuQSL_sph.pkl'):
        self.save_name = file_name
        directory = os.path.dirname(file_name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(file_name, 'wb') as f:
            pickle.dump(self, f)
        print(f'cuQSL_sph saved to {file_name}')
    
    @classmethod
    def load(cls, file_name):
        with open(file_name, 'rb') as f:
            return pickle.load(f)
    
    def __call__(self, points=None, **kwargs):
        """
        Compute the QSL for a given point data based on CUDA in Cartesian coordinates

        Parameters:
        -----------
        points: array_like
            The point data with shape (N,3)
        
        Optional parameters:
        --------------------
        devices: list
            The devices to use for the computation
        python: str
            The python path
        max_batch: int
            The maximum batch size
        temp_save_path: str
            The temporary save path
        atol: float
            The absolute tolerance
        rtol: float
            The relative tolerance
        max_step: float
            The maximum step size in the RKF45 integrator
        min_step: float
            The minimum step size in the RKF45 integrator
        max_steps: int
            The maximum number of steps in the line integration
        err: float
            The error tolerance in the UV initialization
        step_size: float
            The step size in the line integration
        is_print: bool
            Whether to print the information
        """
        t0 = time.time()
        if points is None:
            nrtp = self.Brtp.shape[:3]
            stretched = self.stretched
            rtp_range = self.rtp_range
            rgrid = np.linspace(*rtp_range[0],nrtp[0]) if not self.stretched[0] else np.geomspace(*rtp_range[0],nrtp[0])
            tgrid = np.linspace(*rtp_range[1],nrtp[1]) if not self.stretched[1] else np.geomspace(*rtp_range[1],nrtp[1])
            pgrid = np.linspace(*rtp_range[2],nrtp[2]) if not self.stretched[2] else np.geomspace(*rtp_range[2],nrtp[2])
            rtp = np.stack(list(np.meshgrid(
                rgrid,
                tgrid,
                pgrid,
                indexing='ij'
            )),axis=-1)
            points = rtp.reshape(-1,3)
        devices = kwargs.pop('devices',['cuda'] if torch.cuda.is_available() else ['cpu'])
        total_batch = len(points)
        ntasks = len(devices)
        batch_size = int(np.ceil(total_batch/ntasks))
        temp_save_path = './.qsl_temp_save_path/'
        python = kwargs.pop('python', 'python')
        self.usr_kwargs = kwargs
        if not os.path.exists(temp_save_path):
            os.makedirs(temp_save_path)
        task_list = []
        for i,device in enumerate(devices):
            batch_points = points[batch_size*i:batch_size*(i+1)]
            self.points = batch_points
            self.save(os.path.join(temp_save_path,f'task_{i:04d}.pkl'))
            task_list.append(os.path.join(temp_save_path,f'task_{i:04d}.pkl'))
        commands = []
        for i,task_name in enumerate(task_list):
            command = f'{python} -m cuQSL.cuQSL_sph_scripts --load_file {task_name} --save_file {task_name.replace(".pkl","_results.pkl")} --device {devices[i]}'
            command = command + f' > {task_name.replace(".pkl",".out")} 2>&1'
            commands.append(command)

        processes = []
        for command in commands:
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            processes.append(process)

        for process in processes:
            process.wait()
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                print(f'Error: {stderr.decode()}')
        self.message = {}
        for i,task_name in enumerate(task_list):
            try:
                with open(task_name.replace(".pkl",".out"), 'r', encoding='utf-8') as file:
                    content = file.read()
                self.message[f'task_{i:04d}']=content
            except FileNotFoundError:
                print(f'file: `{task_name.replace(".pkl",".out")}`  do not exist!')
        
        logQ = []
        Length = []
        Twist = []
        hist = dict()
        for i,task_name in enumerate(task_list):
            with open(task_name.replace(".pkl","_results.pkl"), 'rb') as f:
                results = pickle.load(f)
            logQ.append(results['logQ'])
            Length.append(results['length'])
            Twist.append(results['Tw'])
            hist[f'task_{i:04d}']=results['hist']
        self.info['prob_history'] = hist
        logQ = np.concatenate(logQ,axis=0)
        Length = np.concatenate(Length,axis=0)
        Twist = np.concatenate(Twist,axis=0)
        t1 = time.time()
        print(f'## Time taken: {(t1-t0)/60:.3f} minutes')
        shutil.rmtree(temp_save_path)
        return logQ,Length,Twist