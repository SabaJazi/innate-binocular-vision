from numpy import *
import scipy
import pickle
import pylab
from PIL import Image
def generate_2d_noise(x_size = 100, y_size=100, f=1):
  """ this functin generates a 2D array of 1/f noise """
  im = random.rand(x_size,y_size)
  imfft = fft.fftshift(fft.fft2(im))
  mag = abs(imfft)
  phase = angle(imfft)
  [x, y] = meshgrid(range(-x_size//2,x_size//2),range(-y_size//2,y_size//2))
  radius = sqrt(x**2 + y **2)
  radius[where(radius == 0)] = 1  # avoid division by 0
  filter = 1.0 / (radius ** f)
  newfft = filter * exp(1j*phase)
  im = real(fft.ifft2(fft.fftshift(newfft)))
  
  im -= im.mean()
  im /= im.std()
  
  # pylab.imshow(im)
  return im
  
def threshold_noise(f=1, thresh=0, levels = 2, show = False, image_width = 256):

  x_size = image_width
  y_size = image_width
  noise_im = generate_2d_noise(x_size, y_size, f)
  thresh_im = zeros(shape(noise_im))
  if thresh == 0:
    thresh += 1e-20
  for i in range(levels-1):
    # percentile of the threshold
    percentile = (i+1) * 100.0 / levels
    location = scipy.stats.scoreatpercentile(noise_im.flatten(),percentile)

    # using a soft threshold
    norm_term = 1.0 / (levels - 1)
    thresh_im += scipy.stats.norm.cdf(noise_im,loc=location,scale=thresh)
  
  if (show):
    pylab.figure()
    pylab.bone()
    pylab.imshow(thresh_im)
    pylab.title('f = ' + str(f))
    
  return thresh_im

def boxcount_dim(thresh_img, show=False, start_size = 10):
  """ this function computes the fractal dimension of a 2D thresholded image
  using the box-count method. """
  
  if(show):
    print ("warning: this function currently assumes circularly-symmetric boundary conditions")
  zo_img = thresh_img.copy()
  # set the nonzero locations to one
  zo_img[where(thresh_img > 0)] = 1
  edge_img = zeros(shape(zo_img))
  w = shape(thresh_img)[0]  
  h = shape(thresh_img)[1]
  if (w < 50) or (h < 50):
    return
  
  # finding boundary areas
  for x in range(0,w):
    for y in range(0,h):
      x2 = (x+1)%w
      y2 = (y+1)%h
      if (zo_img[x,y] != zo_img[x2,y]) or (zo_img[x,y] != zo_img[x,y2]):
        edge_img[x,y] = 1
      else:
        edge_img[x,y] = 0
  
  s_vals = range(start_size,0,-1)
  s_index = -1
  n_vals = zeros(shape(s_vals))
  for s in s_vals:
    s_index += 1
    n_cnt = 0
    for x in range(0, w-s+1, s):
      for y in range(0, h-s+1, s):
        patch = edge_img[x:x+s, y:y+s]
        if sum(patch) > 0:
          n_cnt += 1
    if (show):
      print ("s:", s, ", n's:", n_cnt, ", # boxes:", (w/s)*(h/s), ", % covered:",n_cnt * 1.0 / ( (w/s)*(h/s) ))
    n_vals[s_index] = n_cnt
  
  if(show):
    print ("estimates of box count fractal dimension")
    for i in range(0,len(s_vals)):
      print ("s:",s_vals[i], " dim:", log(n_vals[i]) / log(w/s_vals[i]))
    
  if show:
    import pylab
    pylab.figure()
    pylab.plot(s_vals, log(n_vals) / log(w/array(s_vals)))
    
  # return the estimate from the smallest box size in the list  
  return log(n_vals[-1]) / log(w/s_vals[-1])

def collect_noise_patches(num_patches = 5000, patch_width = 8, downsample=2, f_exp = 2, levels = 2):
  """ collects image patches, in the same way as the LGN model, for analysis """

  max_tries = num_patches * 20
  image_width = 200
  
  img_first_patch = 0 # the first patch number accepted from an image
  img_first_try = 0 # the first attempt to take a patch from the image
  patch_cnt = 0
  try_cnt = 0
  ds = downsample
  w = patch_width * ds
  d = w * w
  d_final = patch_width * patch_width
  avg_filt = ones([ds, ds],'float') / ds**2

  layer_patch = zeros([1,w,w],float)
  down_patch = zeros([1,patch_width,patch_width],'float')  
  patch = zeros([d,1],float)
  
  img_patches = zeros([d_final,num_patches],float)

  # change the image sampled from
  active = threshold_noise(f_exp,levels=levels, image_width = image_width)
  # normalizing the activity image
  active -= active.mean()
  active /= active.std()
      
  # collect the patches
  while patch_cnt < num_patches and try_cnt < max_tries:
    try_cnt += 1  # number of total patches attempted

    if (try_cnt - img_first_try) > 20 * 100 or \
      (patch_cnt - img_first_patch) > 100:
      # change the image sampled from
      active = threshold_noise(f_exp,levels=levels, image_width = image_width)
      # normalizing the activity image
      active -= active.mean()
      active /= active.std()
      
      img_first_patch = patch_cnt
      img_first_try = try_cnt
      print (float(patch_cnt)/num_patches )
    
    px = random.randint(0,image_width - w)
    py = random.randint(0,image_width - w)
        
    layer_patch[0,:,:] = active[px:px+w,py:py+w].copy()
    patch_std = layer_patch.std()
    
    if patch_std > 0.4:
      # create the patch vector
      # downsample the patch
      for x in range(patch_width):
        for y in range(patch_width):
          down_patch[0,x,y] = sum(avg_filt * layer_patch[0,ds*x:ds*x+ds,ds*y:ds*y+ds])      
      patch = reshape(down_patch, d_final)     
      patch = patch - mean(patch)         
      img_patches[:,patch_cnt] = patch.copy()
      patch_cnt += 1
  return img_patches

def generate_depth_map(x_size = 500, y_size=500):
  """ this function generates a random depth map from an array of zeros
  with circles of values 10 or 20 """
  im = zeros([x_size, y_size])
  num_circles = 3
  circle_radius = 50
  for i in range(0,num_circles):
    depth = (random.randint(2) + 1) * 10
    c_x = random.randint(x_size-circle_radius*2) + circle_radius
    c_y = random.randint(y_size-circle_radius*2) + circle_radius
    [x, y] = meshgrid(arange(x_size)-c_x,arange(y_size)-c_y)
    radius = sqrt(x**2 + y **2)
    im[where(radius < circle_radius)] = depth
  return im

def make_autostereogram(depth_map, pattern, shiftLR=70):
  """ this function creates an autostereogram from a depth map and a pattern to apply """
  # make depthmap an integer array
  depth_map = array(depth_map.round(),'int')
  x_size = depth_map.shape[0]+shiftLR
  y_size = depth_map.shape[1]
  pattern = tile(pattern, (1,y_size//pattern.shape[1]+1))
  pattern = tile(pattern, (shiftLR//pattern.shape[0]+1,1))
  pattern = pattern[0:shiftLR,0:y_size]

  im = zeros([x_size, y_size],'float')
  im[0:shiftLR,:] = pattern
  for y in range(y_size):
    p_x = 0 # the pattern offset
    p_row = pattern[:,y].copy()
    shape(p_row)
    old_depth = 0
    for x in range(shiftLR,x_size):
      depth = depth_map[x-shiftLR,y]
      depth_change = depth-old_depth
      old_depth = depth

      # change p_row if there is a depth change
      if depth_change != 0:
        old_width = len(p_row)
        new_width = old_width - depth_change
        p_row = tile(p_row, (2,))            
        if depth_change > 0:
          # shift the pixels left at p_x
          p_row[p_x:old_width] = p_row[p_x+depth_change:old_width+depth_change]
          p_row = p_row[0:new_width]
        elif depth_change < 0:
          p_row = concatenate((p_row[0:p_x],p_row[p_x:p_x-depth_change],p_row[p_x:old_width]))

      p_x += 1
      p_x = p_x % len(p_row)
      im[x,y] = p_row[p_x]
  return im
      
      
  """
  background = tile(pattern,(x_size//shiftLR+1,1))
  background = background[0:x_size,:]
  
  im = background.copy()
  for x in range(depth_map.shape[0]):
    for y in range(depth_map.shape[1]):
      im[x,y] = background[x+depth_map[x,y],y]
  return im
  """
  
import V1Tools
import FilterTools
from PIL import Image

def test_autostereogram(save_stereogram=True):
  shiftLR = 70
  depthmap = generate_depth_map(500,500)
  pylab.figure(3)
  pylab.imshow(depthmap.transpose())
  pattern = generate_2d_noise(70,70)
  a = make_autostereogram(depthmap, pattern, shiftLR)
  pylab.figure(1)
  pylab.imshow(a.transpose())
  
  if save_stereogram:
    norm_a = array(255 * (a - a.min()) / (a.max() - a.min()), 'int8')
    pilImage = Image.fromarray(norm_a.transpose(), 'L')
    pilImage.save('autostereogram.png')
  
  state = V1Tools.read_imgs(a)
  width=10
  p = FilterTools.pick_gabor_params(100,width)
  filts = FilterTools.gabor_filter_generator(p,width)
  V1Tools.convolve_filters(state,filts,filts,5)
  
  # with the correct shiftLR
  output_map = V1Tools.get_depth_mat(state, 70)
  pylab.figure(2)
  pylab.imshow(output_map.transpose())

  # with a different shiftLR
  output_map = V1Tools.get_depth_mat(state, 60)
  pylab.figure(5)
  pylab.imshow(output_map.transpose())

def test_picturegram(inputimage):
  import V1Tools
  import FilterTools
  state = V1Tools.read_img(img_in="data/" + inputimage + ".png",adaptShiftLR=True)
  width=10
  print(state['shiftLR'])
  p = FilterTools.pick_gabor_params(100,width)
  filts = FilterTools.gabor_filter_generator(p,width)
  V1Tools.convolve_filters(state,filts,filts,5)
  output_map = V1Tools.get_depth_mat(state)
  pylab.figure(2)
  pylab.imshow(output_map.transpose())
