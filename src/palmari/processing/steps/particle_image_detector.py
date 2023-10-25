"""
"""
import dask
from dask import delayed
import dask.array as da
from dask.diagnostics import ProgressBar
import numpy as np
import pandas as pd

from .base import Detector
from ...quot.findSpots import log

class PIDetector(Detector):
    """
    Detector class that collects particle images in addition to guesses for initial positions.
    """
    def __init__(
        self, k: float = 1.5, w: int = 21, t: float = 1.0, pi_size: int = 8
    ):
        """
        Parameters
        ----------
        k : float
            Estimated spot size.
        w : float
            Window size.
        t : float
            Threshold value.
        pi_size : int
            Length of one side of target particle image size in pixels. 
        """
        self.k = k # sigma, i.e. spot size
        self.w = w # window size
        self.t = t # threshold
        self.pi_size = pi_size

    def detect_frame(self, img: np.array) -> pd.DataFrame:
        return detect(img, t=self.t, k=self.k, w=self.w, pi_size=self.pi_size)

    @property
    def name(self):
        return "Particle image detector"

    widget_types = {"t": "FloatSlider", "method": "ComboBox"}
    widget_options = {
        "t": {
            "step": 0.1,
            "min": 0.1,
            "max": 1000,
            "label": "Threshold",
            "readout": True,
        },
        "k": {
            "step": 0.1,
            "label": "Spot size (px)",
            "min": 0.0,
        },
        "w": {
            "step": 2,
            "label": "Window size (px)",
            "min": 5,
            "tooltip": "Size of the square window used for testing.",
        },
        "pi_size": {
            "step": 1,
            "label": "Particle Image Size (px)",
            "min": 2,
            "max": 100,
        },
    }

    def movie_detection(self, mov: da.Array):

        if hasattr(self, '_mov_dict'):
            mov_dict = self._mov_dict
        else:
            mov_dict = {}
            self._mov_dict = mov_dict
    
        # check if movie is already calculated (not lazy)
        if isinstance(mov, np.ndarray): # if numpy array, don't treat like dask
            loc_results, particle_images = self.detect(mov, frame_start=0)
            
        if 'mov_np' in mov_dict: # if numpy array already in memory, don't compute from dask
            loc_results, particle_images = self.detect(mov_dict['mov_np'], frame_start=0)
        
        elif isinstance(mov, da.Array):
            slice_size = mov.chunksize[0]
            n_slices = mov.shape[0] // slice_size

            positions_pis_delayed = [] # positions and particle images, computed lazily
            for i in range(n_slices + 1):
                start = i * slice_size
                end = min((i + 1) * slice_size, mov.shape[0])
                if start >= end:
                    continue
                positions_pis_delayed.append(delayed(self.detect)(img=mov[start:end], 
                                                                  frame_start=start)) # delayed list of tuples

            with ProgressBar():
                # with warnings.catch_warnings():
                # warnings.simplefilter("ignore", category="RuntimeWarning")
                positions_pis = dask.compute(*positions_pis_delayed) # list of tuples no longer delayed

            # split tuples into separate lists
            positions_dfs = [data[0] for data in positions_pis] 
            particle_images = [data[1] for data in positions_pis]

            # individual chunks concatenated
            loc_results = pd.concat(positions_dfs)
            particle_images = np.concatenate(particle_images, axis=0, dtype=np.float32) # float32 necessary for gpufit

            loc_results.set_index(np.arange(loc_results.shape[0]), inplace=True)
            loc_results["detection_index"] = np.arange(loc_results.shape[0])

        mov_dict['particle_data'] = particle_images # save individual particle images for future
        # print('end pid particle_data:', self._mov_dict['particle_data'])
    
        return loc_results

    def detect(self, img: np.array, frame_start: int = 0) -> pd.DataFrame:
        detections = []
        particles = []
        for frame_idx in range(img.shape[0]): # loop over every frame
            frame = img[frame_idx] # no longer transposing here
            loc_data, part_data  = self.detect_frame(frame)
            if loc_data.shape[0] > 0:
                frame_detections = pd.DataFrame(data=loc_data, columns=["y", "x"])
                frame_detections["frame"] = frame_idx
                # Localize spots to subpixel resolution
                detections.append(frame_detections)
                particles.append(part_data)

        if len(detections) > 0:
            detections = pd.concat(detections, ignore_index=True, sort=False)
            particles = np.concatenate(particles, axis=0)
        else:
            detections = pd.DataFrame.from_dict(
                {"x": [], "y": [], "frame": []}
            )
            particles = np.array([]).reshape(0, self.pi_size*self.pi_size)

        # Performs checks on the returned pd.DataFrame
        detections["frame"] += frame_start
        for c, v in self.cols_dtype.items():
            assert c in detections.columns, "%s not in columns" % c
        for c in self.cols_dtype:
            if c in detections.columns:
                detections[c] = detections[c].astype(self.cols_dtype[c])

        return detections, particles

def detect(img, t, k, w, pi_size):
    """
    For laplacian of gaussian detection.
    """
    positions = log(img, k=k, w=w, t=t, return_filt=False) # return row, col (y, x)

    if len(positions.shape) == 2: # at least one particle detected
        hw = pi_size // 2 # half-window
        hwr = pi_size % 2 # half-window remainder

        # check to see if window fits on frame
        positions = positions[(positions[:, 0] >= hw)
                            & (positions[:, 0] < img.shape[0] - hw - hwr + 1)
                            & (positions[:, 1] >= hw)
                            & (positions[:, 1] < img.shape[1] - hw - hwr + 1),
                            :,
                            ]
        
        particles = [img[row - hw : row + hw + hwr, col - hw : col + hw + hwr].ravel() for row, col in positions]
        particles = np.asarray(particles)
    else:
        particles = np.array([]).reshape(0, pi_size*pi_size)

    return positions, particles # localizations, particle images