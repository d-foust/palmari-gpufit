from .base import Detector
import numpy as np
from numpy.fft import rfft2, irfft2, fftshift
from dask.array.fft import rfft2 as rfft2_da
from dask.array.fft import irfft2 as irfft2_da
from dask.array.fft import fftshift as fftshift_da
from dask.array import from_array
import pandas as pd
from skimage.measure import regionprops, regionprops_table, label
from skimage.morphology import binary_dilation, disk
import dask.array as da
from ...data_structure.acquisition import Acquisition
import scipy.ndimage as ndi
import napari as na
import warnings
warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars")

class GodinezRohr(Detector):
    
    widget_types = {'sigma': 'FloatSpinBox',
                   'c': 'FloatSpinBox',
                   'min_spot_size': 'FloatSpinBox',
                   'max_spot_size': 'FloatSpinBox',
                   'eccentricity': 'FloatSpinBox',
                   'show_log': 'CheckBox',
                   'show_mask': 'CheckBox'}
    
    widget_options = {
        'sigma': {
            'label': 'sigma (pixels)',
            'min': 0.2,
            'max': 15,
            'step': 0.1
        },
        'c': {
            'label': 'c (stdevs)',
            'min': 0.1,
            'max': 100,
            'step': 0.1,
        },
        'min_spot_size': {
            'label': 'min. diameter (pixels)',
            'min': 1,
            'max': 30,
            'step': 0.1,
        },
        'max_spot_size': {
            'label': 'max. diameter (pixels)',
            'min': 1,
            'max': 30,
            'step': 0.1
        },
        'eccentricity': {
            'label': 'eccentricity',
            'min': 0,
            'max': 1,
            'step': 0.01,
        },
        'show_log': {
            'label': 'Show LoG',
        },
        'show_mask': {
            'label': 'Show thresholded',
        },
        "pi_size": {
            "step": 1,
            "label": "Particle Image Size (px)",
            "min": 2,
            "max": 100,
        },
    }
    
    def __init__(self,
                 sigma: float = 1, 
                 c: float = 3,
                 min_spot_size: float = 3,
                 max_spot_size: float = 20,
                 eccentricity: float = 1,
                 show_log = False,
                 show_mask = False,
                 pi_size: int = 8):
        
        self.sigma = sigma
        self.c = c # stdevs above mean to threshold
        self.min_spot_size = min_spot_size
        self.max_spot_size = max_spot_size
        self.eccentricity = eccentricity
        self.show_log = show_log
        self.show_mask = show_mask
        self.pi_size = pi_size
        
    def movie_detection(self, mov: da.Array):
        if not hasattr(self, "_mov_dict"):
            self._mov_dict = {}
        return log_detection(mov,
                            sigma=self.sigma,
                            c=self.c,
                            min_spot_size=self.min_spot_size,
                            max_spot_size=self.max_spot_size,
                            eccentricity=self.eccentricity,
                            show_log = self.show_log,
                            show_mask = self.show_mask,
                            pi_size = self.pi_size,
                            mov_dict = self._mov_dict)
        
    @property
    def name(self):
        return "Godinez-Rohr Detector"
    
    def detect_frame(self, image: np.array) -> pd.DataFrame:
        pass
    
    def show(self, viewer: na.Viewer):
            
        if self.show_log == True:
            viewer.add_image(data=self._mov_dict['image_log'], name='LoG',
                            scale=(1, self._mov_dict["pixel_size"], self._mov_dict["pixel_size"]),
                            visible=False)
        if self.show_mask == True:
            viewer.add_labels(data=self._mov_dict["foreground_mask"].astype('int'), name='Thresholded Image',
                        scale=(1, self._mov_dict["pixel_size"], self._mov_dict["pixel_size"]),
                        visible=False,
                        color={0: [0,0,0,0], 1: 'magenta'})
    
def log_detection(mov,
                 sigma,
                 c,
                 min_spot_size,
                 max_spot_size,
                 eccentricity,
                 show_log,
                 show_mask,
                 pi_size,
                 mov_dict):
    
    if isinstance(mov, da.Array):
        if "mov_np" in mov_dict:
            mov = mov_dict["mov_np"]
        else:
            mov = mov.compute() # need data in memory to do calculations
            mov_dict["mov_np"] = mov # keep in case needed for future use
        
    if "goodframes" in mov_dict:
        goodframes = mov_dict["goodframes"]
    else:
        goodframes = np.ones(mov.shape[0], dtype='bool')
    gfi = np.arange(mov.shape[0], dtype='int')[goodframes] # goodframes indexes
        
    if "rois" in mov_dict:
        rois_bool = mov_dict["rois"] > 0
        rois_neighborhood = rois_bool
    else:
        rois_bool = np.ones([mov.shape[1], mov.shape[2]], dtype='bool')
    rois_neighborhood = binary_dilation(rois_bool, footprint=disk(4))
    
    positions_dfs = []
    particle_images = []

    H, W = mov.shape[1:] # height and width
    window_size = np.min([H, W])
    G_ft = _log_setup(H, W, sigma, window_size)
    if show_mask == True:
        mov_foreground_mask = np.zeros(mov.shape, dtype='bool')

    for fr, image_frame in zip(gfi, mov[goodframes]):
        log_frame = log_filter(image_frame, G_ft) # laplacian of gaussian filtered frame
        rois_log_mean = log_frame[rois_neighborhood].mean()
        rois_log_std = log_frame[rois_neighborhood].std()
        foreground_mask = log_frame >= rois_log_mean + c*rois_log_std
        if show_mask == True:
            mov_foreground_mask[fr] = foreground_mask
        
        frame_detections, particles = guess_from_mask(foreground_mask, image_frame, min_spot_size, max_spot_size, eccentricity, pi_size)
        
        if frame_detections.shape[0] > 0:
            frame_detections['frame'] = fr
            positions_dfs.append(frame_detections)
            particle_images.append(particles)
            
    if len(positions_dfs) > 0: # at least one particle detected in entire movie
        detections = pd.concat(positions_dfs, ignore_index=True)
        particles = np.concatenate(particle_images, axis=0)

        # if hasattr(acq, 'rois'):
        if "rois" in mov_dict:
            rois_bool_expanded = binary_dilation(rois_bool, footprint=disk(2)) # expand a little so don't miss any that are close
            is_in_roi = rois_bool_expanded[detections['y'], detections['x']]
            detections_in_rois = detections[is_in_roi]
            detections_in_rois.reset_index(inplace=True)
            detections = detections_in_rois
            particles = particles[is_in_roi]
    else: # no particles were detected
        detections = pd.DataFrame(columns=('frame', 'x', 'y'))
        particles = np.array([]).reshape(0, pi_size*pi_size)
          
    # acq._guesses = detections
    mov_dict["guesses"] = detections
    mov_dict["particle_data"] = particles
    
    if show_log == True or show_mask == True:
        mov_dict["image_log"] = fftshift_da(
                                    irfft2_da(
                                        rfft2_da(from_array(mov, chunks=(1,-1,-1)), axes=(1,2)) * G_ft[None,:,:], axes=(1,2), s=mov.shape[1:]
                                        ),
                                    axes=(1,2)
                                    )
    if show_mask == True:
        mov_dict["foreground_mask"] = mov_foreground_mask #threshold_dask(mov_dict["image_log"], rois_bool, c)
        
    return detections

def guess_from_mask(foreground_mask, image_frame, min_spot_size, max_spot_size, eccentricity, pi_size):
    
    labels = label(foreground_mask)

    props = pd.DataFrame(
        regionprops_table(
            labels,
            intensity_image=image_frame,
            properties=(
                'equivalent_diameter_area',
                'centroid',
                'eccentricity'
            )
        )
    )
    
    size_bool = (props['equivalent_diameter_area'] >= min_spot_size) & (props['equivalent_diameter_area'] <= max_spot_size)
    e_bool = props['eccentricity'] < eccentricity

    hw = pi_size // 2 # half-window
    hwr = pi_size % 2 # half-window remainder
    props["y"] = np.rint(props["centroid-0"])
    props["x"] = np.rint(props["centroid-1"])
    row_bool = (props["y"] >= hw) & (props["y"] < image_frame.shape[0] - hw - hwr + 1)
    col_bool = (props["x"] >= hw) & (props["x"] < image_frame.shape[1] - hw - hwr + 1)

    frame_detections = props[size_bool & e_bool & row_bool & col_bool].reset_index()
    frame_detections["y"] = frame_detections["y"].astype('int')
    frame_detections["x"] = frame_detections["x"].astype('int')
    
    if len(frame_detections.shape) == 2: # at least one particle detected
        particles = [image_frame[r - hw : r + hw + hwr, c - hw : c + hw + hwr].ravel() for r, c in frame_detections[["y","x"]].values]
        particles = np.asarray(particles)
    else:
        frame_detections = pd.DataFrame.from_dict({"x": [], "y": [], "frame": []})
        particles = np.array([]).reshape(0, pi_size*pi_size)
    
    return frame_detections, particles


def log_filter(image, G_ft):
    """
    Laplace of Gaussian filter on image with 
    """
    return fftshift(irfft2(rfft2(image) * G_ft, s=image.shape))
    
def _log_setup(H, W, k, w):
    """
    Generate a Laplacian-of-Gaussian (LoG) transfer function
    for subsequent convolution with an image.

    args
    ----
        H, W    :   int, height and width of target image
        k       :   float, kernel sigma
        w       :   int, kernel size

    returns}
    -------
        2D ndarray, dtype complex128, the transfer function

    """
    S = 2 * (k**2)
    g = np.exp(-((np.indices((w, w)) - (w - 1) / 2) ** 2).sum(0) / S)
    g = g / g.sum()
    log_k = -ndi.laplace(g)
    return rfft2(pad(log_k, H, W))

def pad(I, H, W, mode='ceil'):
    """
    Pad an array with zeroes around the edges, placing 
    the original array in the center. 

    args
    ----
        I       :   2D ndarray, image to be padded
        H       :   int, height of final image
        W       :   int, width of final image
        mode    :   str, either 'ceil' or 'floor'. 'ceil'
                    yields convolution kernels that function
                    similarly to scipy.ndimage.uniform_filter.

    returns
    -------
        2D ndarray, shape (H, W)

    """
    H_in, W_in = I.shape
    out = np.zeros((H, W))
    if mode == "ceil":
        hc = np.ceil(H / 2 - H_in / 2).astype(int)
        wc = np.ceil(W / 2 - W_in / 2).astype(int)
    elif mode == "floor":
        hc = np.floor(H / 2 - H_in / 2).astype(int)
        wc = np.floor(W / 2 - W_in / 2).astype(int)
    out[hc : hc + H_in, wc : wc + W_in] = I
    return out

def threshold_dask(image_log, rois_bool, c):
    
    nframes = image_log.shape[0]
    image_th = da.array([image_log[fr] >= image_log[fr][rois_bool].mean() + c*image_log[fr][rois_bool].std() for fr in range(nframes)])
        
    return image_th

def insert_badframes(image, goodframes):
    new_image = np.zeros([len(goodframes), image.shape[1], image.shape[2]])
    new_image[goodframes] = image
    return new_image