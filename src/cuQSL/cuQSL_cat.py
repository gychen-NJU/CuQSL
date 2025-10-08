'''
cuQSL_cat.py
'''

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import os
import pickle
import subprocess
import time
import shutil

os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '0'

def Cartesian_curl(V):
    V = V.transpose(3,0,1,2)
    Vx = V[0]
    Vy = V[1]
    Vz = V[2]
    curl = np.zeros_like(V)
    curl[0] = np.gradient(Vz,axis=1)-np.gradient(Vy,axis=2)
    curl[1] = np.gradient(Vx,axis=2)-np.gradient(Vz,axis=0)
    curl[2] = np.gradient(Vy,axis=0)-np.gradient(Vx,axis=1)
    return curl.transpose(1,2,3,0)

class TorchRegularInterpolator:
    def __init__(self, grid_data):
        '''
        grid_data: input grid data with shape of (nx,ny,nz,ndir1,ndir2,...)
        '''
        self.data = grid_data if isinstance(grid_data, torch.Tensor) else torch.from_numpy(grid_data).float()
        self.ndim = grid_data.ndim
        self.size = grid_data.shape
        if self.ndim == 4:
            self.data = self.data.permute(3,0,1,2).unsqueeze(0)
        elif self.ndim == 5:
            self.data = self.data.permute(3,4,0,1,2).unsqueeze(0).contiguous()
            self.data = self.data.view(self.data.shape[0],-1,*self.data.shape[3:])
        else:
            raise ValueError(f'Invalid grid data shape: {self.data.shape}')

    def __call__(self, points, **kwargs):
        '''
        points: interpolation coordinates with shape (N, dim) of range [-1,1]
        parameters are the same is torch.nn.functional.grid_sample
        Return: interpolation results with shape (N,dir1,dir2,...)
        '''
        mode   = kwargs.pop('mode', 'bilinear')
        device = torch.device(kwargs.pop('device','cuda' if torch.cuda.is_available() else 'cpu'))
        points = torch.from_numpy(points).float().to(device) if isinstance(points, np.ndarray) else points.float().to(device)
        points = points[:,[2,1,0]].view(1,-1,1,1,3)
        data   = self.data.to(device)
        if self.ndim == 4:
            interp = F.grid_sample(data, points, mode=mode, align_corners=True, **kwargs).squeeze().t()
            # print(f'interp shape: {interp.shape}')
            # print(f'points shape: {points.shape}')
            interp = interp.view(points.shape[1],-1)
        elif self.ndim == 5:
            interp = F.grid_sample(data, points, mode=mode, align_corners=True, **kwargs).squeeze().t()
            # print(f'interp shape: {interp.shape}')
            # print(f'points shape: {points.shape}')
            interp = interp.view(points.shape[1],*self.size[-2:])
        else:
            raise ValueError(f'Invalid grid data shape: {self.data.shape}')
        return interp
        
class qsl_solver_cat:
    """
    Compute the QSL for a given point data based on CUDA in Cartesian coordinates

    Parameters:
    -----------
    Bxyz: array_like
        The magnetic field data with shape (nx,ny,nz,3)
    grid_xyz: list
        The grid list [xgrid,ygrid,zgrid]

    Example:
    --------
    >>> Bxyz = np.load('Bxyz.npy')
    >>> grid_xyz = [xgrid,ygrid,zgrid]
    >>> solver = qsl_solver_cat(Bxyz, grid_xyz)
    >>> points = np.load('points.npy')
    >>> device = ['cuda:0','cuda:1']
    >>> logQ,Length,Twist = solver(points, devices=device)
    """
    def __init__(self, Bxyz, grid_xyz):
        self.Bxyz = Bxyz
        self.grid_xyz = grid_xyz
        self.save_name = None
        self.usr_kwargs = {}
        self.points = None
        
    def save(self, file_name='cuQSL_cat.pkl'):
        self.save_name = file_name
        directory = os.path.dirname(file_name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(file_name, 'wb') as f:
            pickle.dump(self, f)
        print(f'cuQSL_cat saved to {file_name}')
    
    @classmethod
    def load(cls, file_name):
        with open(file_name, 'rb') as f:
            return pickle.load(f)

    def __call__(self, points, **kwargs):
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
        devices = kwargs.pop('devices',['cuda'] if torch.cuda.is_available() else ['cpu'])
        total_batch = len(points)
        ntasks = len(devices)
        batch_size = int(np.ceil(total_batch/ntasks))
        current_path   = kwargs.get('current_path','./')
        temp_save_path = os.path.join(current_path,'.qsl_temp_save_path/')
        print('Temporary file path: ', temp_save_path)
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
            command = f'{python} -m cuQSL.cuQSL_cat_scripts --load_file {task_name} --save_file {task_name.replace(".pkl","_results.pkl")} --device {devices[i]}'
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
        
        logQ = []
        Length = []
        Twist = []
        for i,task_name in enumerate(task_list):
            with open(task_name.replace(".pkl","_results.pkl"), 'rb') as f:
                results = pickle.load(f)
            logQ.append(results['logQ'])
            Length.append(results['length'])
            Twist.append(results['Tw'])
        logQ = np.concatenate(logQ,axis=0)
        Length = np.concatenate(Length,axis=0)
        Twist = np.concatenate(Twist,axis=0)
        t1 = time.time()
        print(f'## Time taken: {(t1-t0)/60:.3f} minutes')
        shutil.rmtree(temp_save_path)
        return logQ,Length,Twist
