"""
Same as window perecentile, but force into numpy array
"""
import numpy as np
import scipy.ndimage as ndi

from .base import *
# from ...image_tools.localization import sliding_window_filter


class WindowPercentileFilterNP(MoviePreProcessor):

    widget_types = {"percentile": "FloatSpinBox", "window_size": "SpinBox"}
    widget_options = {
        "percentile": {
            "step": 1.0,
            "tooltip": "percentile of the pixel intensity values in the window which will be considered as the ground level.",
            "min": 0.0,
            "max": 100.,
            "value": 50,
            "label": "Percentile",
        },
        "window_size": {
            "step": 1,
            "tooltip": "Size of the windows along which quantiles are computed",
            "label": "Window",
            "min": 5,
            "max": 10001,
            "value": 101
        },
    }

    def __init__(self, percentile: float = 3.0, window_size: int = 100, clip: bool = True):
        self.percentile = percentile
        self.window_size = window_size
        self.clip = clip

    def preprocess(self, mov: np.ndarray) -> np.ndarray:

        if not hasattr(self, "_mov_dict"):
            self._mov_dict = {}

        if isinstance(mov, da.Array) and "mov_np" not in self._mov_dict:
            mov = mov.compute().astype(np.float64)
        elif isinstance(mov, da.Array) and "mov_np" in self._mov_dict:
            mov = self._mov_dict["mov_np"]

        return sliding_window_filter(
            data=mov, percentile=self.percentile, window_size=self.window_size, clip=self.clip, mov_dict=self._mov_dict,
        )

    @property
    def name(self):
        return "Local percentile filtering w/ NumPy"

    @property
    def action_name(self):
        return "Filter"
    
def sliding_window_filter(
    data: np.ndarray, percentile: float = 50, window_size: int = 100, clip: bool = True, mov_dict: dict = {}
):
    if not isinstance(data.ravel()[0], np.float64):
        data = data.astype(np.float64)

    percent = ndi.percentile_filter(
        data, percentile=percentile, size=(window_size, 1, 1), mode="reflect"
    )

    if clip == True:
        clipped = (data - percent).clip(0.0)
    else:
        clipped = data - percent

    return clipped