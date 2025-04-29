import numpy as np
import time
from pyevtk.hl import gridToVTK

def xyz2rtp(xyz):
    """
    Convert Cartesian coordinates (x, y, z) to spherical coordinates (r, theta, phi).
    
    Parameters:
    x (float): x-coordinate
    y (float): y-coordinate
    z (float): z-coordinate
    
    Returns:
    tuple: (r, theta, phi)
        r (float): radial distance
        theta (float): polar angle (in radians)
        phi (float): azimuthal angle (in radians)
    """
    x,y,z=xyz.transpose(3,0,1,2)
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.where(r!=0,np.arccos(z/r),0)
    phi = np.arctan2(y, x) % (np.pi*2)
    return np.stack([r, theta, phi],axis=-1)

def rtp2xyz(rtp):
    """
    Convert spherical coordinates (r, theta, phi) to Cartesian coordinates (x, y, z).
    
    Parameters:
    r (float): radial distance
    theta (float): polar angle (in radians)
    phi (float): azimuthal angle (in radians)
    
    Returns:
    tuple: (x, y, z)
        x (float): x-coordinate
        y (float): y-coordinate
        z (float): z-coordinate
    """
    r,theta,phi=rtp.transpose(3,0,1,2)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.stack([x, y, z],axis=-1)

def data2vts(data_ls, name_ls, vts_name='data', **kwargs):
    """
    Convert a list of data arrays to a VTK structured grid file (.vts).

    Parameters:
    -----------
    data_ls : list of numpy.ndarray
        A list of data arrays to be saved. Each array should be either 3D or a tuple of 3D arrays.
    name_ls : list of str
        A list of names corresponding to each data array in `data_ls`.
    vts_name : str, optional
        The name of the VTK file to be saved (default is 'data').
    **kwargs : dict
        Additional keyword arguments:
        - xyz : tuple of numpy.ndarray
            Cartesian coordinates (x, y, z) for the structured grid.
        - rtp : tuple of numpy.ndarray
            Spherical coordinates (r, theta, phi) for the structured grid. If provided, they will be converted to Cartesian coordinates.

    Returns:
    --------
    None
        The function saves the data to a VTK file and does not return any value.

    Raises:
    -------
    TypeError
        If neither `xyz` nor `rtp` is provided.

    Notes:
    ------
    - The function uses `np.gradient` to measure the time taken for the conversion and saving process.
    - If the data array is not 3D, it is assumed to be a tuple of 3D arrays and is converted accordingly.
    - The VTK file is saved using the `gridToVTK` function from the `evtk.hl` module.

    Example:
    --------
    >>> import numpy as np
    >>> x = np.linspace(0, 1, 10)
    >>> y = np.linspace(0, 1, 10)
    >>> z = np.linspace(0, 1, 10)
    >>> X, Y, Z = np.meshgrid(x, y, z)
    >>> data1 = np.sin(X) * np.cos(Y) * np.cos(Z)
    >>> data2 = np.cos(X) * np.sin(Y) * np.sin(Z)
    >>> data2vts([data1, data2], ['data1', 'data2'], vts_name='example', xyz=(X, Y, Z))
    Save data to example.vts.
    Time Used: 0.001 min...
    """
    t0 = time.time()
    xyz = kwargs.get('xyz', None)
    rtp = kwargs.get('rtp', None)
    if xyz is None and rtp is None:
        return TypeError('`xyz` or `rtp` should be provided at least one...')
    X,Y,Z = xyz.transpose(3,0,1,2) if xyz is not None else rtp2xyz(rtp).transpose(3,0,1,2)
    pointData = dict()
    for name,data in zip(name_ls,data_ls):
        pointData[name]=np.ascontiguousarray(data) if data.ndim==3 else tuple(np.ascontiguousarray(idata) for idata in data)
    gridToVTK(vts_name, np.ascontiguousarray(X), np.ascontiguousarray(Y), np.ascontiguousarray(Z), 
              pointData=pointData)
    print(f'Save data to {vts_name}.vts.\nTime Used: {(time.time()-t0)/60:8.3} min...')