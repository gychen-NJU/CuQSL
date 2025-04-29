# -*- coding: utf-8 -*-
"""
cuQSL package initialization module
"""

__author__ = 'Chen Guoyin'
__email__ = 'gychen@smail.nju.edu.cn'
__version__ = '1.0.0'

from .cuQSL_cat import qsl_solver_cat as cat
from .cuQSL_sph import qsl_solver_sph as sph

__all__ = ['cat', 'sph', "__version__"]
