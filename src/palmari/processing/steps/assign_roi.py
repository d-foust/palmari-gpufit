from .base import LocProcessor
from ...data_structure.acquisition import Acquisition
import dask.array as da
import pandas as pd
import numpy as np
from skimage.morphology import binary_dilation, dilation, disk

class AssignRoi(LocProcessor):
    
    widget_types = {'buffer_size' : 'SpinBox'}
    widget_options = {
        'buffer_size': {
            'label': 'Buffer size (pixels)',
            'min': 0,
            'max': 10,
            'step': 1
        }
    }
    
    
    def __init__(self, buffer_size = 0, acq: Acquisition = None):
        self.buffer_size = buffer_size
        self.process_data = {}
        
    def process(self, mov: da.Array, locs: pd.DataFrame, pixel_size: float):
        return assign_roi(locs, pixel_size, self.buffer_size, process_data=self.process_data)
    
    @property
    def name(self):
        return "ROI Assigner"
    
    @property
    def action_name(self):
        return "Assign ROIs"
    

    
def assign_roi(locs: pd.DataFrame, pixel_size: float, buffer_size: int, process_data: dict):
    """
    """
    rows = np.rint(locs['x'] / pixel_size).astype('int')
    cols = np.rint(locs['y'] / pixel_size).astype('int')
    
    # if hasattr(acq, 'rois'):
    if 'rois' in process_data:
        # rois = acq.rois.copy()
        rois = process_data['rois'].copy()
        if buffer_size > 0:
            rois = expand_rois(rois, buffer_size)
        R, C = rois.shape
        rows_filt = (rows > 0) & (rows < R)
        cols_filt = (cols > 0) & (cols < C)
        filt = rows_filt & cols_filt
        
        locs['rois'] = 0
        locs.loc[filt, 'rois'] = rois[rows[filt], cols[filt]]
        
    else:
        locs['rois'] = 1
    
    return locs

def expand_rois(rois, buffer_size=1):
    """
    """
    rois_exp = rois.copy()
    for i in range(buffer_size):
        rois_bool = rois_exp > 0
        edges = binary_dilation(rois_bool, footprint=disk(1)) ^ rois_bool
        rois_exp[edges] = dilation(rois_exp, footprint=disk(1))[edges]
        
    return rois_exp
        
    