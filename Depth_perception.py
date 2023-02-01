#---------------------------------LIBRARIES---------------------------------
import numpy as np

from PIL import Image, ImageOps
from scipy import signal
from ipywidgets import interact, interactive, fixed

import matplotlib.pyplot as plt

#----------------------------------------------------------------------------
def generate_gabor(size, shift, sigma, rotation, phase_shift, frequency):
    radius = (int((size[0]/2.0), int((size[1]/2.0))))
    [x, y] = np.meshgrid(range(-radius[0], radius[0]), range(-radius[1], radius[1]))
    x = (x - int(shift[0])) * frequency
    y = (y - int(shift[1])) * frequency
    tmp = x * np.cos(rotation) + y * np.sin(rotation) + phase_shift
    radius = (int(size[0]/2.0), int(size[1]/2.0))
    [x, y] = np.meshgrid(range(-radius[0], radius[0]), range(-radius[1], radius[1]))
    
    x = x - int(shift[0])
    y = y - int(shift[1])
    x1 = x * np.cos(rotation) + y * np.sin(rotation)
    y1 = -x * np.sin(rotation) + y * np.cos(rotation)
    
    sinusoid = np.cos(tmp)
    
    gauss = np.e * np.exp(np.negative(0.5 * ((x1**2 / sigma[0]**2) + (y1**2 / sigma[1]**2))))
    return gauss * sinusoid

#----------------------------------------------------------------------------
def open_norm(path, verbose=False):
    raw = np.array(Image.open(path).convert("L"))
    norm = (raw - np.mean(raw)) / np.std(raw)
    
    if verbose:
        return raw, norm
    else:
        return norm
#-----------------------------------------------------------------------------
def linear_convolution(center, slide):
    if center.shape != slide.shape:
        return
    
    padded_slide = np.zeros((center.shape[0], center.shape[1] * 3))
    padded_slide[0:, center.shape[1]:center.shape[1] * 2] = center
    
    estimate = np.zeros([center.shape[1] * 2])
    
    for x in range(center.shape[1] * 2):
        estimate[x] = np.sum(padded_slide[0:, x:center.shape[1] + x] * slide)
        
    print(np.abs(np.argmax(np.abs(estimate)) - center.shape[1]))
    
    return np.abs(estimate)

#-----------------------------------------------------------------------------------
def double_convolve(normal, shifted, image, pupillary_distance):
    normal_convolved = signal.convolve2d(image, normal, boundary='symm', mode='same')
    shifted_convolved = signal.convolve2d(image, shifted, boundary='symm', mode='same')
    
    return_shape = image.shape
    
    realigned = np.zeros(return_shape)
    print(realigned.shape)
    
    normal_convolved = normal_convolved[0:, 0:-pupillary_distance]
    shifted_convolved = shifted_convolved[0:, pupillary_distance:]
    
    mul = normal_convolved * shifted_convolved
    realigned[0:, pupillary_distance:] = mul
    
    return np.abs(realigned)
    
#----------------------------------------------------------------------------------
def demo_gabor(shift_x, shift_y, sigma_x, sigma_y, rotation, phase_shift, frequency, pd):
    size = (64, 64)
    shift = (shift_x, shift_y)
    sigma = (sigma_x, sigma_y)
    
    static = generate_gabor(size, shift, sigma, rotation, phase_shift, frequency)
    
    plt.imshow(static, origin="lower")
    plt.show()
    
    n_shift = (shift_x + pd, shift_y)
    shifted = generate_gabor(size, n_shift, sigma, rotation, phase_shift, frequency)
    
    plt.imshow(shifted, origin="lower")
    plt.show()
    
    print(size, shift, sigma, rotation, phase_shift, frequency)
    print(size, n_shift, sigma, rotation, phase_shift, frequency)
    
    #Start explorative linear convolution
    linear_convolution(static, shifted)