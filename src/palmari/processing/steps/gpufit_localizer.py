import pandas as pd
import numpy as np
from scipy.ndimage import uniform_filter

import pygpufit.gpufit as gf

from .base import SubpixelLocalizer

class GpufitLocalizer(SubpixelLocalizer):

    widget_options = {'tolerance': 'FloatSpinBox',
                      'max_iterations': 'SpinBox',
                      'estimator': 'ComboBox'}
    
    widget_options = {
        'tolerance': {
            'min': 1e-8,
            'max': 1e-1,
            'step': 1e-8,
            'label': 'tolerance',
        },
        'max_iterations': {
            'min': 3,
            'max': 1000,
            'step': 1,
            'label': 'max iterations',
        },
        'estimator': {
            'label': 'estimator',
            'choices': [
                ('least squares estimator', gf.EstimatorID.LSE),
                ('maximum likelihood estimator', gf.EstimatorID.MLE)
            ]
        }
    }

    def __init__(self, 
                 tolerance: float = 1e-4,
                 max_iterations: int = 25,
                 estimator: int = gf.EstimatorID.LSE):
        
        # self.model = gf.ModelID.GAUSS_2D
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        # self.parameters_to_fit = np.ones(5, dtype=np.int32)
        self.estimator = estimator

    @property
    def name(self):
        return "Gpufit localizer"

    def movie_localization(self, mov: np.ndarray, detections: pd.DataFrame):

        if not hasattr(self, '_mov_dict'):
            self._mov_dict = {}
            print('mov_dict not found. Empty mov_dict created.')
        else:
            print('mov_dict found.')
            print(self._mov_dict.keys())
        mov_dict = self._mov_dict

        if 'particle_data' in mov_dict: # need particle data to use this localizer, add flexibility in future
            print('found particle_data')
            print(mov_dict['particle_data'].shape)
            localizations = fit_particle_data(particle_data=mov_dict['particle_data'],
                                              detections=detections,
                                              model_id=gf.ModelID.GAUSS_2D,
                                              tolerance=self.tolerance,
                                              max_number_iterations=self.max_iterations,
                                              parameters_to_fit=np.ones(5, dtype=np.int32),
                                              estimator_id=self.estimator)
        else: 
            print("didn't find particle data")
            localizations = detections
        
        return localizations
    
    # need this function because abstract class in base, but doesn't actually do anything
    def localize_frame(self, img: np.ndarray, detections: np.ndarray) -> pd.DataFrame:
        pass
    
def fit_particle_data(particle_data, # 2d array, n particles x n pixels
                      detections, # pd.DataFrame
                      weights=None,
                      model_id=gf.ModelID.GAUSS_2D,
                      tolerance=1e-4,
                      max_number_iterations=25,
                      parameters_to_fit=np.ones(5, dtype=np.int32),
                      estimator_id=gf.EstimatorID.LSE):
    """
    
    """
    if particle_data.dtype != np.float32:
        particle_data = particle_data.astype(np.float32)

    if parameters_to_fit.dtype != np.int32:
        parameters_to_fit = parameters_to_fit.astype(np.int32)

    n_particles, n_pixels = particle_data.shape
    window_size = int(np.sqrt(n_pixels))

    initial_parameters = np.zeros([n_particles, 5], dtype=np.float32) # amp, x0, y0, sigma, offset

    amp_init, sigma_init, offset_init, intensity_total, bgd_mean, noise_est = calc_initial_parameters(particle_data)
    initial_parameters[:,0] = amp_init
    initial_parameters[:,3] = sigma_init
    initial_parameters[:,4] = offset_init

    initial_parameters[:,1:3] = window_size // 2

    out = gf.fit(data=particle_data,
           weights=weights,
           model_id=model_id,
           initial_parameters=initial_parameters,
           tolerance=tolerance,
           max_number_iterations=max_number_iterations,
           parameters_to_fit=parameters_to_fit,
           estimator_id=estimator_id,
           user_info=None)

    # x0 : col -> y in palmari output
    # y0 : row -> x in palmari output
    fitpars = out[0] # 2d array: columns: amp, x0, y0, sigma, offset
    states = out[1] # 
    chi_squares = out[2]
    number_iterations = out[3]
    execution_time = out[4]

    origin_offset = window_size // 2 # Gpufit first value of window corresponds to (0,0)

    detections['y_detect'] = detections['y']
    detections['x_detect'] = detections['x']

    detections['amp'] = fitpars[:,0]
    detections['x'] = fitpars[:,1] + detections['x_detect'] - origin_offset
    detections['y'] = fitpars[:,2] + detections['y_detect'] - origin_offset
    detections['sigma'] = fitpars[:,3]
    detections['offset'] = fitpars[:,4]

    detections['fit_state'] = states
    detections['chi_squares'] = chi_squares
    detections['number_iterations'] = number_iterations
    detections['execution_time'] = execution_time

    detections['amp_init'] = amp_init
    detections['sigma_init'] = sigma_init
    detections['offset_init'] = offset_init
    detections['I_tot'] = intensity_total # intensities summed minus border_mean
    detections['border_mean'] = bgd_mean
    detections['noise_est'] = noise_est

    return detections

def calc_initial_parameters(particle_data):
    """
    Calculate initial parameters using the approach described in:
    Leutenegger, M., and M. Weber. 2021. Least-squares fitting of Gaussian spots on graphics processing units. arXiv.
    ---
    """
    n_particles, n_pixels = particle_data.shape
    image_size = int(np.sqrt(n_pixels))

    particle_images = particle_data.reshape(n_particles, image_size, image_size)
    smooth_images = uniform_filter(particle_images, size=(1,3,3), mode='wrap')

    offset = smooth_images.min(axis=(1,2))
    amp = smooth_images.max(axis=(1,2)) - offset

    threshold = amp * np.exp(-0.5) + offset

    M = (particle_data > threshold[:,None]).sum(axis=1)

    sigma = np.sqrt(M / np.pi)

    noise_est, bgd_mean = calc_noise_estimate(particle_images)

    intensity_total = particle_images.sum(axis=(1,2))
    intensity_total = intensity_total - bgd_mean*n_pixels

    return amp, sigma, offset, intensity_total, bgd_mean, noise_est

def calc_noise_estimate(particle_images):

    top_row = particle_images[:,0,:]
    bottom_row = particle_images[:,-1,:]
    left_col = particle_images[:,0,1:-1]
    right_col = particle_images[:,-1,1:-1]

    border_elements = np.concatenate([
        top_row,
        bottom_row,
        left_col,
        right_col
    ],
    axis=1)

    noise = np.std(border_elements, axis=1)
    bgd_mean = np.mean(border_elements, axis=1)

    return noise, bgd_mean