
#-------------------------------------------libraries-----------------------------------------
import PIL
import numpy as np
import matplotlib.pyplot as plt

#---------------------------------------------------------------------------------------------

def generate_2d_pink_noise(size=(70, 70), frequency=1.0, seed=0):
    #seed random for reproducible results and generate a random 2D array
    np.random.seed(seed)
    rand_array = np.random.rand(size[0], size[1])
    
    plt.imshow(rand_array)
    plt.title("rand_array")
    plt.show()
    
    #Group 1: calculating features from randomness using discrete fourier transform
    rand_fft = np.fft.fft2(rand_array)
    rand_shifted = np.fft.fftshift(rand_fft)
    rand_phase = np.angle(rand_shifted)
    print("Rand_Array ", rand_array.shape)
    
    #creates 2D gradients for x and y
    half_cast = (int(size[0] / 2.0), int(size[1] / 2.0))
    [x_gradient, y_gradient] = np.meshgrid(range(-half_cast[0], half_cast[0]), range(-half_cast[1], half_cast[1]))
    print("X_Gradient ", x_gradient.shape)
    
    #Group 2: initalizing inverse frequency gradients, or pink noise.
    #Think Pythagorean Theorem
    radial_gradient = np.sqrt(x_gradient**2 + y_gradient**2)
    radial_gradient[np.where(radial_gradient == 0)] = 1 #set all 0 values to 1 to avoid division by 0
    frequency_gradient = radial_gradient**frequency
    inverse_gradient = 1.0 / frequency_gradient
    
    #Group 3: mixes the randomness and inverse gradient to activate certain areas
    noise = inverse_gradient * np.exp(1j * rand_phase).transpose()
    noise_shifted = np.fft.fftshift(noise)
    noise_fft = np.fft.ifft2(noise_shifted)
    pink_noise = np.real(noise_fft)
    norm_pink_noise = (pink_noise - np.mean(pink_noise)) / np.std(pink_noise)
    
    return norm_pink_noise