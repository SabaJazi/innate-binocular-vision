
#-------------------------------------------libraries-----------------------------------------
import PIL
import numpy as np
import matplotlib.pyplot as plt
import os
#-----------------------------------------making a pattern for stereogram -------------------------

def generate_2d_pink_noise(size=(70, 70), frequency=1.0, seed=0):
    #seed random for reproducible results and generate a random 2D array
    np.random.seed(seed)
    rand_array = np.random.rand(size[0], size[1])
    
    plt.imshow(rand_array, cmap='gray')
    plt.title("rand_array(pattern)")
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

#-----------------------------------making a depthmap for stereogram ----------------------------------------------
    #Function to create the circular depthmap for the autostereogram
def create_circular_depthmap(shape=(270, 270), center=None, radius=100):
    depthmap = np.zeros(shape, dtype=np.float)
    
    a = 100
    b = 170
    n = 270
    r = radius#50
    y, x = np.ogrid[-a:n-a, -b:n-b]
    
    mask = x * x + y * y <= r * r
    depthmap[mask] = 4
    
    return depthmap


#--------------------------------------making the stereogram from the pattern and depthmap------------------------------------------
#Function to create the autosterogram from the depthmap and pattern
def make_autostereogram(depthmap, pattern):
    autostereogram = np.zeros_like(depthmap, dtype=pattern.dtype)
    
    for r in np.arange(autostereogram.shape[0]):
        for c in np.arange(autostereogram.shape[1]):
            if c < pattern.shape[1]:
                autostereogram[r, c] = pattern[r % pattern.shape[0], c]
            else:
                shift = int(depthmap[r, c])
                autostereogram[r, c] = autostereogram[r, c - pattern.shape[1] + shift]
                
    return autostereogram

#-------------------------------# run codes: Create and display the depthmap----------------------------

depthmap = create_circular_depthmap(radius = 50)
plt.imshow(depthmap, cmap='gray')

plt.title("Depth Map")
plt.show()

#Generate the pink noise
pink_noise = generate_2d_pink_noise(size=(70,70))

#Create and display the autostereogram from the generated depthmap and pink noise
autostereogram = make_autostereogram(depthmap, pink_noise)
plt.imshow(autostereogram, cmap='gray')
plt.title("Autostereogram")
plt.show()

autostereogram = autostereogram[:-70, :]
rescaled = (255.0 / autostereogram.max() * (autostereogram - autostereogram.min())).astype(np.uint8)
folder_path = "output"

# Check if the folder exists, and if not, create it
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Save the file to the folder
sv = PIL.Image.fromarray(autostereogram, mode='L')
sv.save(os.path.join(folder_path, "shift5_70patch.png"))

rescaled = (depthmap[:-100, 100:]).astype(np.uint8)
sv = PIL.Image.fromarray(rescaled, mode='L')
sv.save("output/depthmap.png")

inv = np.invert(rescaled, dtype=np.uint8)
sv = PIL.Image.fromarray(inv)
sv.save("output/inverted_dm.png")