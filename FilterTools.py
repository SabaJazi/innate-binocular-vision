from numpy import *
import scipy
import pickle
import pylab

def convert_patches(old_patches, convert_dim=4, num_layers = 2):
  """ this function converts between a filter matrix with columns as filters and
  a 'num_of_filters x 2 x width x width' array """
  # testing if already converted
  if len(old_patches.shape) == convert_dim:
    return old_patches.copy()  
  if convert_dim == 4:
    patch_num = old_patches.shape[1]    
    w = int(sqrt( old_patches.shape[0] / num_layers ))
    new_patches = zeros([patch_num, num_layers,w,w],'float')    
    for i in range(0,patch_num):
      reshaped = old_patches[:,i].reshape((num_layers,w,w)).copy() 
      # normalize each filter to be zero mean and unit variance
      reshaped = reshaped - reshaped.mean()
      new_patches[i,:,:,:] = reshaped/reshaped.std()  
  elif convert_dim == 2:
    patch_num = old_patches.shape[0]    
    data_dim = old_patches.shape[1] * old_patches.shape[2] * old_patches.shape[3]
    new_patches = zeros([data_dim, patch_num],'float')
    for i in range(0,patch_num):
      new_patches[:,i] = old_patches[i,:,:,:].reshape(data_dim)
  return new_patches

def show_patches_mat(pre_patches, show_patch_num = 16, display=True, num_layers=2):
  """ this function generates a 2D array to display image patches """

  # make sure the patches are in the right format
  patches = convert_patches(pre_patches, 2, num_layers) 

  tot_patches = patches.shape[1]
  data_dim = patches.shape[0]
  patch_width = sqrt(data_dim / num_layers)
  
  # extract show_patch_num patches
  disp_patch = zeros([data_dim, show_patch_num], float)
  for i in range(0,show_patch_num):
    #if i < 5:
      # the first 5 patches
      #patch_i = i
    #elif i < 10:
      # spread the samples in the middle
    patch_i = i * tot_patches // show_patch_num
    #else:
      # the last 5 patches
      #patch_i = tot_patches - (show_patch_num - i)
  
    patch = patches[:,patch_i].copy()
    pmax  = patch.max()
    pmin = patch.min()
    # fix patch range from min to max to 0 to 1
    if max > min: 
      patch = (patch - pmin) / (pmax - pmin)
    disp_patch[:,i] = patch.copy()

  bw = 5    # border width
  pw_y = patch_width
  pw_x = patch_width * num_layers + (num_layers-1)*bw
  
  patches_y = int(sqrt(show_patch_num))
  patches_x = int(ceil(float(show_patch_num) / patches_y))
  patch_img = disp_patch.max() * ones([(pw_x + bw) * patches_x - bw,
    patches_y * (pw_y + bw) - bw], float)
  for i in range(0,show_patch_num): 
    y_i = int(i / patches_y)
    x_i = i % patches_y
    reshaped = disp_patch[:,i].reshape((num_layers,patch_width,patch_width))
    full_patch = zeros([pw_x, pw_y], float)
    full_patch[0:patch_width,:] = reshaped[0,:,:].copy()
    if num_layers == 2:
      full_patch[patch_width + bw - 1:2 * patch_width + bw, :] = reshaped[1,:,:].copy()
    patch_img[x_i*(pw_x+bw):x_i*(pw_x+bw)+pw_x,y_i*(pw_y+bw):y_i*(pw_y+bw)+pw_y] = full_patch
  
  if display:
    pylab.bone()
    pylab.imshow(patch_img.T, interpolation='nearest')
    pylab.axis('off')
  return patch_img

def collect_natural_patches(img_folder, num_patches = 5000, patch_width = 8, downsample=2):
  """ collects image patches, in the same way as the LGN model, for analysis 
  the natural images are from a specific folder of 13 .tiff files"""

  import Image
  max_tries = num_patches * 50
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
  nat_img_cnt = 1  
  active = Image.open(img_folder + '/' + str(nat_img_cnt) + '.tiff')
  active = asarray(active, 'double').transpose()  
  # normalizing the activity image
  active -= active.mean()
  active /= active.std()
      
  # collect the patches
  while patch_cnt < num_patches and try_cnt < max_tries:
    try_cnt += 1  # number of total patches attempted

    if (try_cnt - img_first_try) > 50 * 100 or \
      (patch_cnt - img_first_patch) > num_patches/12:
      # change the image sampled from
      nat_img_cnt += 1
      active = Image.open(img_folder + '/' + str(nat_img_cnt) + '.tiff')
      active = asarray(active, 'double').transpose()        
      # normalizing the activity image
      active -= active.mean()
      active /= active.std()
      
      img_first_patch = patch_cnt
      img_first_try = try_cnt
      print (float(patch_cnt)/num_patches) 
    
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

def calculate_patch_disparities(patches):
  """ this function calculates the disparities between the left and right
  sides of patches, and returns the information as a 
  num_patches x num_disparities matrix """

def get_disparity_mat(patches, reorder=False):
  """ this function returns the matrix which maps filter activities to
  depth judgements.
  matrix dimensions are disparity (in pixels) x filter num
  Each column is for a particular filter where each row represents
  a different depth.  The center row is 0 disparity. """
  
  conv_patches = convert_patches(patches, 4)
  filter_num = conv_patches.shape[0] # number of filters
  pw = conv_patches.shape[2] # patch width
  disp_mat = zeros([pw + 1, filter_num],'float')
  for c in range(0,filter_num):
    # note the patches have zero mean and unit energy
    l_patch = conv_patches[c,0,:,:]
    r_patch = conv_patches[c,1,:,:]
    for r in range(-pw/2,pw/2+1):
      l_start = max(0,-r)
      l_end = min(pw,pw-r)
      r_start = max(0,r)
      r_end = min(pw,pw+r)
      l_overlap = l_patch[l_start:l_end,:]
      r_overlap = r_patch[r_start:r_end,:]
      corr = corrcoef(l_overlap.flatten(), r_overlap.flatten())[0,1]
      #corr = (l_overlap * r_overlap).sum()
      # correction for parts that don't overlap
      #corr *= pw/float(l_end-l_start)
      disp_mat[pw/2+r,c] = corr
      
  if reorder:
    max_shift = (disp_mat.shape[0]-1) / 2    
    reorder_indices = argsort(disp_mat.argmax(0))
    disp_mat = disp_mat[:,reorder_indices]
  return disp_mat

def get_disparities(patches):
  """ this function returns a histogram of disparities for
      a population of patches """
  disp_mat = get_disparity_mat(patches, reorder=False)
  num_of_disparities = disp_mat.shape[0]
  num_of_patches = disp_mat.shape[1]

  disp_list = disp_mat.argmax(0)
  disparity_weights = zeros(num_of_disparities)
  print ("new list")
  for i in range(num_of_patches):
    # number of standard deviations above the mean
    # stdevs = (disp_mat[disp_list[i],i] - mean(disp_mat[:,i])) / std(disp_mat[:,i])
    if not isnan(disp_mat[disp_list[i],i]):
      # and abs(disp_mat[disp_list[i],i]) > 0.01:
      #if i < 50:
        #print "disp: ", disp_list[i]
        #print "mean: ", mean(disp_mat[:,i]), "std: ", std(disp_mat[:,i]), \
        #  "val: ", disp_mat[disp_list[i],i], "stds: ", stdevs
      disparity_weights[disp_list[i]] += 1
  
  # if you're summing disparities curves instead of just taking the max...
  # disparity_weights = abs(disp_mat).sum(1)

  disparities = arange(0,num_of_disparities) - (num_of_disparities-1)/2
  return disparities, disparity_weights

def depth_map_loss(true_map, computed_map):
  """ this function calculates an error between depth maps """
  assert true_map.shape == computed_map.shape, "map shapes do not match" + str(true_map.shape) + str(computed_map.shape)
  false_positives = len(where(computed_map[where(true_map==0)]!=0)[0])
  print (false_positives)
  false_negatives = len(where(true_map[where(computed_map==0)]!=0)[0])
  print (false_negatives)
  #fix
  return false_positives + false_negatives    

import V1Tools
import magiceye
def depth_function_test(s_vals, true_depth_map, stereogram, shiftLR, width):
  print (s_vals)
  p = pick_gabor_params(50,width,s_min=s_vals[0], s_range=s_vals[1])
  filters = gabor_filter_generator(p,width)
  state = V1Tools.read_img(stereogram, shiftLR=shiftLR)
  V1Tools.convolve_filters(state,filters,filters,5)
  computed_map = V1Tools.get_depth_mat(state)
  pylab.figure(4)
  pylab.imshow(computed_map.transpose())
  pylab.figure(5)
  pylab.imshow(true_depth_map.transpose())
  return depth_map_loss(true_depth_map, computed_map)
  
def find_best_deviation():
  true_depth_map = magiceye.generate_depth_map();
  
  pylab.figure(1)
  pylab.imshow(true_depth_map.transpose())
  shiftLR = 70
  width = 10
  pattern = magiceye.generate_2d_noise(shiftLR,shiftLR)
  stereogram = magiceye.make_autostereogram(true_depth_map, pattern, shiftLR)
  true_depth_map = true_depth_map[0::5,0::5] # for downsizing
  true_depth_map = true_depth_map[width//2:-width//2+1,width//2:-width//2+1]
  pylab.figure(2)
  pylab.imshow(stereogram.transpose())
  
  x0 = array([0.1,0.1])
  args = (true_depth_map, stereogram, shiftLR, width)
  #fix
  x_opt = scipy.optimize.fmin_powell(depth_function_test, x0, args, maxfun=15, maxiter=10, retall=True)
  print (x_opt)
  return x_opt



def pick_gabor_params(num_filters=100, w=10, s_min=2, s_range=2, num_layers = 2):
  """ this function returns randomly generated gabor filter parameters 
  where pairs of filters (if num_layers == 2) are quadrature pairs """
  params = []
  rnd = random.rand
  for f_i in range(0,num_filters,num_layers):
    o = rnd()*pi
    #c = [rnd()*w, rnd()*w]
    c = [w//2, w//2]
    s = [rnd()*s_range+s_min, rnd()*s_range*2+s_min*2]
    f = (2.5 + rnd()*(pi-2.5)) / s[0]
    p = rnd()*2*pi
    
    if num_layers == 1:
      params.append(dict(o=o, c=c, s=s, f=f, p=p))
    for lr in range(num_layers):
      if num_layers == 2:
        lr_dicts = [{},{}]
      else: # num_layers == 1:
        lr_dicts = {}
      if lr == 0:
        p = rnd()*2*pi
      else:
        p = (p+pi/2) % (2*pi)    
      lr_dicts[lr] = dict(o=o, c=c, s=s, f=f, p=p)
    if f_i+1 < num_filters:
      params.append(lr_dicts)
  return params


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
  num_filters = shape(p)[0]
  n_p = zeros([num_filters, 2, w, w], 'float')
  
  for f_i in range(0,num_filters):
    for lr in [0,1]:
      f = p[f_i][lr]
      [X0, Y0] = meshgrid(range(0,w),range(0,w))
      X0 = X0 - f['c'][0]
      Y0 = Y0 - f['c'][1]
      # rotate the coordinate system
      X = X0 * cos(f['o']) + Y0 * sin(f['o']);
      Y = - X0 * sin(f['o']) + Y0 * cos(f['o']);
      temp_n_p = ( 1/(2*pi*f['s'][0]*f['s'][1]) *
        exp( -pi * (X**2 / f['s'][0]**2 + Y**2 / f['s'][1]**2) ) *
        real( exp ( 1j * (X*f['f'] + f['p']) )) )
      temp_n_p -= mean(temp_n_p)
      temp_n_p /= std(temp_n_p)
      n_p[f_i,lr,:,:] = temp_n_p
  return n_p

def test_gabor_filter_generator(layers = 1):
  n_p = gabor_filter_generator(num_layers = 2)

    
def gabor_energy_rem (x, patch):
  """ this function returns the percent energy unexplained by
  a gabor fit to the patch using the parameters in p.

  parameters in x are...
   x[0] = p.o - orientation in radians of the planar cosine wave
   x[1],x[2] = p.c - center position of the filter, first pixel at location 1
   x[3],x[4] = p.s - standard dev of the gaussian
     in the direction parallel and perpendicular to modulation direction
   x[5] = p.f - spatial frequencies in radians / pixel 
   x[6] = p.p - phase of the filter in radians cos=0, -sin=pi/2

   patch should be normalized and square
  """

  w = shape(patch)[0]
  
  # bounds for the algorithm
  # for [f.o f.c(1) f.c(2) f.s(1) f.s(2) f.f f.p];
  lb = [-pi, 0, 0, 0, 0, pi/w, 0]
  ub = [pi, w, w, w, w, pi, pi]    

  if any(x > ub) or any(x < lb):
    return inf

  p = dict(o=x[0], c=[x[1],x[2]], s=[x[3],x[4]], f=[5], p=[6])
  gabor = gabor_filter_generator(p, w)
  
  # compute the relative energy of 
  # the difference between the gabor and the original patch
  gp_diff = gabor - patch;
  rem = sum(gp_diff ** 2) / sum(patch ** 2)
  return rem

def wcov(x,y, weights):
  """
  calculates the weighted covariance matrix for a set of points
  """
  
  if not len(x) == len(y) == len(weights):
    print ("vector lengths are not equal in function wcov")
    exit(0)
  num_points = len(x)
  
  # normalize the wieghts
  norm_weights = weights / sum(weights)
  
  mean_x = sum(norm_weights * x)
  mean_y = sum(norm_weights * y)
  
  cov_mat = zeros(2,2)
  cov_mat[0,0] = sum(norm_weights * (x - mean_x) * (x - mean_x))
  cov_mat[0,1] = cov_mat[1,0] = \
    sum(norm_weights * (x - mean_x) * (y - mean_y))
  cov_mat[1,1] = sum(norm_weights * (y - mean_y) * (y - mean_y))
  cov_mat /= 1 - sum(norm_weights * norm_weights)
  
  return cov_mat

def gabor_fit(o_p, thresh):

  """
  this function fits gabor functions to the image patches in o_p
  INPUT
  o_p - the image patches, 'x' by 'x' by 'num of patches'
  thresh - threshold to stop trying to fit a gabor to the patch 
  OUTPUT
  p - the gabor filter parameters as an array of dictionaries
    p.o - orientation in radians of the planar cosine wave and gaussian envelope
    p.c[0] p.c[1] - center position of the filter, first pixel at location 1
    p.s[0] p.s[1] - standard dev of the gaussian parallel and perpendicular
      to sine modulation
    p.f - spatial frequencies in radians / pixel along the orientation
    p.p - phase of the filter in radians cos=0, -sin=pi/2
  n_p - the fit gabor filters in order of best to worst fit
  o_p - the original filters in the order of n_p
  """
"""
  w = shape(o_p)[0]  # assuming the filters are square
  filt_num = shape(o_p)[2]

  try_max = 10  # number of attempts to fit a filter if above threshold

  # bounds for the algorithm - same as in gabor_rem
  # for [f.o f.c(1) f.c(2) f.s(1) f.s(2) f.f f.p];
  lb = [-pi, 0, 0, 0, 0, pi/w, 0]
  ub = [pi, w, w, w, w, pi, pi]

  p = dict(o=0, c=[1,1], s=[1,1], f=0.5, p=0)
  p = list(repeat(p, filt_num))

  n_p = p.copy()

  # normalizing the original filters
  for o_i in range(filt_num):
    o_p[:,:,o_i] -= mean(o_p[:,:,o_i])
    o_p[:,:,o_i] /= std(o_p[:,:,o_i])
  
  # making initial guesses for the filters
  print 'Initializing gabor filter fits...'

  for filt = range(filt_num)
    # guess the center by finding the mean energy
    tf = o_p[:,:,filt]; # temporary filter
    
    # estimating the orientation, phase, and spatial frequency of the filter
    # using only the max and min filter positions
    # note: this is very crude
    index = argmax(tf)
    max_pos[0] = index // w
    max_pos[1] = index % w
    index = argmin(tf)
    min_pos[0] = index // w
    min_pos[1] = index % w
      
    # estimating the orientation
    pos_diff = min_pos - max_pos
    p[filt].o = math.atan( pos_diff[1] / (pos_diff[0] + eps) )

    # estimating the center as the location of the mean, weighted by energy
    # and the covariance matrix of the gaussian envelope
    energy = tf ** 2
    [X Y] = meshgrid(range(w),range(w))
    v = wcov(flatten(X), flatten(Y),flatten(energy))

    p[filt].c = (min_pos + max_pos) / 2  # center between min and max
    # rotate the covariance matrix back to align the orientation with the x axis
    R = array([ [cos(p[filt].o),  sin(p[filt].o)], [-sin(p[filt].o), cos(p[filt].o)] ])
    v_rot = dot(R,dot(v,R.T))
    p[filt].s[0] = sqrt(v_rot[0,0]) 
    p[filt].s[1] = sqrt(v_rot[1,1])
  
    # estimating the frequency
    m2m_distance = sqrt(sum((min_pos - max_pos) * (min_pos - max_pos)))
    p[filt].f = 2*pi / (2 * m2m_distance);
  
    # estimating the phase
    p[filt].p = pi/2

  # optimize the gabor patch fitting
  for filt = 1:filt_num
    print 'optimizing filter', filt, '/', filt_num
    patch = o_p[:,:,filt]
    gp = p[filt]  # initial "educated" guess
  
    # setup for iteration
    try_cnt = 0;
    x0 = [gp.o gp.c[0] gp.c[1] gp.s[0] gp.s[1] gp.f gp.p]  
    x0_rem = thresh + 1000  # not the best fit
    x1 = x0.copy()
    x1_rem = thresh + 1000  # not the best fit
    while ((x0_rem > thresh) and (try_cnt < try_max) ):
      try_cnt = try_cnt + 1
      fmin(gabor_energy_rem, x1, args = (patch), disp = 1)
        # old matlab minimization code
        #  x1 = fmincon(@(x) gabor_energy_rem(x,patch), ...
        #    x1,[],[],[],[],lb,ub,[],options);
        #
        # new python function specification
        #  fmin(func, x0, args = (), xtol = 0.0001, ftol = 0.0001, maxiter = None, maxfun = None, full_output = 0, disp = 1, retall = 0, callback = None)

      # check result against old one
      x1_rem = gabor_energy_rem(x1,patch)
      if (x1_rem < x0_rem):  # if it's a better fit, replace the old one
        x0_rem = x1_rem
        x0 = x1
      end
      # make a random guess
      x1 = (ub - lb) * rand(size(ub)) + lb
      # give some "educated guess" help
      x1[0] = gp.o
      x1[1] = gp.c[0]
      x1[2] = gp.c[1]
      x1[5] = gp.f
    
    # the right patch has been found
    f = p[filt]  # to initialize f
    f.o = x0[0];  f.c[0] = x0[1];  f.c[1] = x0[2]
    f.s[0] = x0[3];  f.s[1] = x0[4];  f.f = x0[5]; f.p = x0[6]
    p[filt] = f

  # generating the gabors
  n_p = gabor_generator(p, w);

  # finding the fit remainder
  rem = zeros(filt_num)
  for filt in range(filt_num):
    f = p[filt]
    rem[filt] = gabor_energy_rem(
      [f.o, f.c[0], f.c[1], f.s[0], f.s[1], f.f, f.p], o_p[:,:,filt])

  # reorder the filters based on quality of fit
  I = argsort(rem)
  o_p = o_p[:,:,I]
  n_p = n_p[:,:,I]
  p = p[I]

  return p, n_p, o_p


def gabor_summary(p, o_p, thresh = 0, 
  sumstat = {'orient_band_median', 'freq_band_median', 'aspect_ratio'},
  showfigs=0):
  # computes and displays summary statistics for gabor fitted patches
  # shows the following, which are in sumstat
  # INPUT
  # if sumstat is undefined, all stats are computed
  # p, o_p - gabor parameters and original patchs
  # sumstat - list of statistics to compute
  # showfigs - boolean variable showing figures (1) or just computing p_values (0)
  # OUTPUT
  # p_val - the p-value that the statistic matches the physiological parameter value
  # stat - the mean of the parameter chosen
  
  # orientation half-bandwidth
  orient_band_median = 20
  # frequency full-bandwidth
  freq_band_median = 1.4
  aspect_ratio = 2
  
  p_val = zeros(len(sumstat), 1)
  
  filt_num = shape(o_p)[2]
  rem = zeros(filt_num)
  
  # finding the remainder of filters to fit
  for filt in range(filt_num):
    f = p[filt]
    rem[filt] = gabor_energy_rem(
      [f.o f.c[0] f.c[1] f.s[0] f.s[1] f.f f.p], o_p[:,:,filt])
  
  # reorder the filters based on quality of fit
  I = argsort(rem)
  o_p = o_p[:,:,I]
  p = p[I]
  
  # showing the quality of fits
  if (thresh == 0): # thresh wasn't set
    thresh = rem[I[int(filt_num//2)]]  # accept half
  
  # remove filters that are above threshold
  # (they should remain in order)
  junk_p = p[rem > thresh].copy()
  junk_o_p = o_p[:,:,rem > thresh].copy()
  p = p[rem <= thresh]
  o_p = o_p[:,:,rem <= thresh]
  filt_num = len(p)
  
  if (showfigs): # display a histogram of fit qualities
    pylab.figure()
    pylab.hist(rem, sqrt(len(p)))
    pylab.title('quality of fits')
  
  if (showfigs):  # display best and worst filters
    # display range of patches below threshold
    show_max = min(100, filt_num)
    if (show_max > 1):
      patch_sel = int( (range(show_max) + 1e-6) / (show_max-1) * (filt_num-1) + 1)
      pylab.figure()
      pylab.title('ordered fit patches')
      FilterTools.show_patches_mat(o_p[:,:,patch_sel], num_layers=1)
      # display the best fit gabors
      pylab.figure()
      pylab.title('ordered fit gabors')
      FilterTools.show_patches_mat(gabor_generator(p(patch_sel), 16), num_layers=1)
    
    # display range of patches above threshold
    show_max = min(100, len(junk_p))
    if (show_max > 1):
      patch_sel = int( (range(show_max) + 1e-6) / (show_max-1) * (len(junk_p)-1) + 1)
      pylab.figure()
      pylab.title('ordered fit patches')
      FilterTools.show_patches_mat(junk_p[:,:,patch_sel], num_layers=1)
      # display the best fit gabors
      pylab.figure()
      pylab.title('ordered fit gabors')
      FilterTools.show_patches_mat(gabor_generator(junk_p(patch_sel), 16), num_layers=1)
  
  if ('orient_band_median' in sumstat):
    # half-amplitude orientation bandwidth
    # and half-amplitude spatial frequency bandwidth  
    o_bandwidth = zeros(len(p)) 
    for x = range(len(p)):
      c = sqrt(log(2)/pi)
      o_bandwidth[x] = 2 * (180/pi) * atan( c / (p[x].s[1] * p[x].f/(2*pi)) )
  
    # testing the orientation bandwidth
    print 'median of the orient bandwidth: ', median(o_bandwidth)
    print 'std of the orient bandwidth: ', std(o_bandwidth)
    print 'physiological median: ', orient_band_median
    
    if (showfigs):
      pylab.figure()
      pylab.hist(o_bandwidth, sqrt(len(p)))
      pylab.xlim([0, 90])  
      pylab.title('orientation bandwidth in deg (natural: median = 20)')
  
  if ('freq_band_median' in sumstat):
    # half-amplitude spatial frequency bandwidth  
    s_bandwidth = zeros(len(p))  
    for x = range(len(p))
      c = sqrt(log(2)/pi)
      s_bandwidth[x] = 2 * c / (p[x].s[0] * p[x].f/(2*pi))
    
    # testing the frequency bandwidth bandwidth
    p_val_i = sumstat.index('freq_band_median')
    print 'median of the spat freq bandwidth: ', median(s_bandwidth)
    print 'std of the spat freq bandwidth: ', std(s_bandwidth)
    print 'physiological median :', freq_band_median
      
    if (showfigs):
      pylab.figure()
      pylab.hist(s_bandwidth, sqrt(len(p)))
      pylab.xlim([0, 2])
      pylab.title('spatial frequency bandwidth (natural = 1-1.5)')
  
  if ('aspect_ratio' in sumstat):
    # standard deviation of the gaussian envelope
    sx = zeros(len(p))
    sy = zeros(len(p))
    aspect = zeros(len(p))
    for x = range(len(p))
      sx[x] = p[x].s[0]
      sy[x] = p[x].s[1]
      aspect[x] = sy[x] / sx[x]
    
    # testing the aspect ratio
    p_val_i = sumstat.index('aspect_ratio')
    print 'mean of the aspect ratio: ', mean(aspect)
    print 'std of the aspect ratio: ', std(aspect)
    print 'test of mean :', aspect_ratio
     
    if (showfigs):
      pylab.figure()
      pylab.plot(sx, sy, 'b*');
      # hold on;
      # plot([min(sx) max(sx)],[2*min(sx) 2*max(sx)], 'r-');  
      # hold off;
      pylab.xlim([0, 10])
      pylab.ylim([0, 10])
      pylab.title('Standard deviations of the gaussian envelope')
      pylab.xlabel('direction along modulation')
      pylab.ylabel('direction perpendicular to modulation')
"""