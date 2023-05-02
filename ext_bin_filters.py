# import FilterTools
# import magiceye
import scipy
import numpy as np
import random
import matplotlib.pyplot as plt

def generate_depth_map(x_size = 500, y_size=500):
  """ this function generates a random depth map from an array of zeros
  with circles of values 10 or 20 """
  im = np.zeros([x_size, y_size])
  num_circles = 3
  circle_radius = 50
  for i in range(0,num_circles):
    depth = (random.randint(2, 3)) * 10
    c_x = random.randint(x_size-circle_radius*2) + circle_radius
    c_y = random.randint(y_size-circle_radius*2) + circle_radius
    [x, y] = np.meshgrid(np.arange(x_size)-c_x,np.arange(y_size)-c_y)
    radius = np.sqrt(x**2 + y **2)
    im[np.where(radius < circle_radius)] = depth
  return im

def generate_2d_noise(x_size = 100, y_size=100, f=1):
  """ this functin generates a 2D array of 1/f noise """
  im = random.randint(x_size,y_size)
  imfft = scipy.fft.fftshift(scipy.fft.fft2(im))
  mag = abs(imfft)
  phase = scipy.angle(imfft)
  [x, y] = np.meshgrid(range(-x_size//2,x_size//2),range(-y_size//2,y_size//2))
  radius = np.sqrt(x**2 + y **2)
  radius[np.where(radius == 0)] = 1  # avoid division by 0
  filter = 1.0 / (radius ** f)
  newfft = filter * np.exp(1j*phase)
  im = np.real(scipy.fft.ifft2(scipy.fft.fftshift(newfft)))
  
  im -= im.mean()
  im /= im.std()
  
  # pylab.imshow(im)
  return im
  


def generate_2d_pink_noise(size=(70, 70), frequency=1.0, seed=0):
    #seed random for reproducible results and generate a random 2D array
    np.random.seed(seed)
    rand_array = np.random.rand(size[0], size[1])
    
    plt.imshow(rand_array)
    plt.title("2d pink noise rand_array(pattern)")
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



def gabor_filter_generator(p=[], w=10):
  """ generates a series of gabor filter patches from parameters p
  p - the gabor filter dictionary of arrays
  the variables are accessed by p[filt_num][0 or 1]['s_par']
  o - orientation in radians of the planar cosine wave
  c[2] - center position of the filter, first pixel at location 0
  s[2] - standard dev of the gaussian parallel & perp to modulation direction
  f - spatial frequency in radians / pixel 
  p - phase of the filter in radians cos=0, -sin=pi/2
  w - the width in pixels of the patch
  OUTPUT:
  a filter matrix: num of filters x 2 x w x w
  """
  
  if not any(p):
    p = pick_gabor_params()
    
  num_filters = (np.array(p).shape)[0]
  n_p = np.zeros([num_filters, 2, w, w], 'float')
#   print("P printing:", p)
  for f_i in range(0,num_filters):
    for lr in [0,1]:
      f = p[f_i][lr]
      print("f printing:", f)
      [X0, Y0] = np.meshgrid(range(0,w),range(0,w))

      X0 = X0 - f['c'][0]
      Y0 = Y0 - f['c'][1]
      # rotate the coordinate system
      X = X0 * np.cos(f['o']) + Y0 * np.sin(f['o'])
      Y = - X0 * np.sin(f['o']) + Y0 * np.cos(f['o'])
      temp_n_p = ( 1/(2*np.pi*f['s'][0]*f['s'][1]) *
        np.exp( -np.pi * (X**2 / f['s'][0]**2 + Y**2 / f['s'][1]**2) ) *
        np.real( np.exp ( 1j * (X*f['f'] + f['p']) )) )
      temp_n_p -= np.mean(temp_n_p)
      temp_n_p /= np.std(temp_n_p)
      n_p[f_i,lr,:,:] = temp_n_p
  return n_p


def pick_gabor_params(num_filters=100, w=10, s_min=2, s_range=2, num_layers = 2):
  """ this function returns randomly generated gabor filter parameters 
  where pairs of filters (if num_layers == 2) are quadrature pairs """
  params = []
  rnd = random.randint
  for f_i in range(0,num_filters,num_layers):
    o = rnd(0,100)*np.pi
    #c = [rnd()*w, rnd()*w]
    c = [w//2, w//2]
    s = [rnd(0,100)*s_range+s_min, rnd(0,100)*s_range*2+s_min*2]
    f = (2.5 + rnd(0,100)*(np.pi-2.5)) / s[0]
    p = rnd(0,100)*2*np.pi
    
    if num_layers == 1:
      params.append(dict(o=o, c=c, s=s, f=f, p=p))
    for lr in range(num_layers):
      if num_layers == 2:
        lr_dicts = [{},{}]
      else: # num_layers == 1:
        lr_dicts = {}
      if lr == 0:
        p = rnd(0,100)*2*np.pi
      else:
        p = (p+np.pi/2) % (2*np.pi)    
      lr_dicts[lr] = dict(o=o, c=c, s=s, f=f, p=p)
    if f_i+1 < num_filters:
      params.append(lr_dicts)
  return params
#   -------------------------------------------------------
# disp_mat=FilterTools.get_disparity_mat(patches, reorder=False)
# depthmap = generate_depth_map(500,500)
shiftLR = 70
true_depth_map = r'C:\vscode\innate-binocular-vision\innate-binocular-vision\output\inverted_dm.png'
depthmap=true_depth_map
# pattern = generate_2d_noise(70,70)
pattern = generate_2d_pink_noise(size=(70,70))
width=10
p = pick_gabor_params(100,width)
filts = gabor_filter_generator(p,width)
# disp_mat=FilterTools.get_disparity_mat(patches, reorder=False)
