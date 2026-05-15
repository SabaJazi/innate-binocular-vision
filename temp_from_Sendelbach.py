from PIL import Image, ImageOps
import pylab
import hashlib
import progressbar
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
from random import randint

from scipy import signal
from scipy.interpolate import griddata
from sklearn.decomposition import FastICA
from sklearn.feature_extraction import image as skimage
from ipywidgets import interact, interactive, fixed




def generate_gabor(size, shift, sigma, rotation, phase_shift, frequency):
    radius = (int((size[0]/2.0)), int((size[1]/2.0)))
    [x, y] = np.meshgrid(range(-radius[0], radius[0]), range(-radius[1], radius[1])) # a BUG is fixed in this line
    x = x - int(shift[0])
    y = y - int(shift[1])
    x = x * frequency
    y = y * frequency
    tmp = x * np.cos(rotation) + y * np.sin(rotation) + phase_shift
    radius = (int(size[0]/2.0), int(size[1]/2.0))
    [x, y] = np.meshgrid(range(-radius[0], radius[0]), range(-radius[1], radius[1])) # a BUG is fixed in this line

    x = x - int(shift[0])
    y = y - int(shift[1])
    x1 = x * np.cos(rotation) + y * np.sin(rotation)
    y1 = -x * np.sin(rotation) + y * np.cos(rotation)

    sinusoid = np.cos(tmp)

    gauss = np.e * np.exp(np.negative(0.5 * ((x1**2 / sigma[0]**2) + (y1**2 / sigma[1]**2))))
    gauss = gauss / 2*np.pi * sigma[0] * sigma[1]

    gabor = gauss * sinusoid
    return gabor

def open_norm(path,verbose=False):
    raw = np.array(Image.open(path).convert("L"))
    norm = (raw - np.mean(raw)) / np.std(raw)

    if verbose:
        return raw, norm
    else:
        return norm

def linear_convolution(center, slide):
    if (center.shape != slide.shape):
        return
    padded_slide = np.zeros((center.shape[0],center.shape[1]*3))
    padded_slide[0:,center.shape[1]:center.shape[1]*2] = center
    #plt.imshow(padded_slide,origin="lower")
    #plt.show()
    estimate = np.zeros([center.shape[1]*2])
    for x in range(center.shape[1]*2):
        dot = np.sum(padded_slide[0:,0+x:center.shape[1]+x] * slide)
        estimate[x] = dot
    #plt.plot(estimate)
    #plt.show()
    return np.abs(estimate)

def double_convolve(normal, shifted, image, pupillary_distance):

    #CHECKOUT https://github.com/maweigert/gputools
    #probably VERY advantageous to switch over to GPU for convolutions!

    normal_convolved = signal.convolve2d(image, normal, boundary='symm', mode='same')
    shifted_convolved = signal.convolve2d(image, shifted, boundary='symm', mode='same')

    return_shape = image.shape

    realigned = np.zeros(return_shape)




    normal_convolved = normal_convolved[0:,0:-pupillary_distance]
    shifted_convolved = shifted_convolved[0:,pupillary_distance:]




    diff = np.subtract(normal_convolved, shifted_convolved)
    mul = normal_convolved * shifted_convolved
    #plt.imshow(mul,cmap="nipy_spectral")
    #plt.show()

    #REMOVE BELOW COMMENTS TO THRESH SUBHALF VALUES
    #low_values_flags = mul <= 0 #mul.max()*0.5  # Where values are low
    #mul[low_values_flags] = 0  # All low values set to 0
    realigned[0:,pupillary_distance:] = mul
    return np.abs(mul)

def scale_disparity(activity_map, disparity_map):
    scaled_disparity = np.zeros([activity_map.shape[0],activity_map.shape[1],disparity_map.shape[0]])
    scaled_disparity[:,:] = disparity_map
    for x in range(activity_map.shape[0]):
        for y in range(activity_map.shape[1]):
            scaled_disparity[x,y] = activity_map[x,y] * scaled_disparity[x,y]

    return scaled_disparity






import random
def distance(x0, y0, x1, y1):
  return np.sqrt(pow(x0-x1,2) + pow(y0-y1,2))

class LGN:
  """
  this class defines a model which generates binocular spontaneous activity
  """

  def __init__(self, width = 128, p = 0.5, r = 1.0, t = 1, trans = 0.0,
    make_wave = True, num_layers=2, random_seed=0):
    random.seed(random_seed)
    self.width = width
    self.p = p
    self.r = r
    self.t = t
    self.trans = trans
    self.num_layers = num_layers
    if make_wave:
      self.reset_wave()

  def reset_wave(self):
    """ create another random wave """
    # setting up the network
    w = self.width
    self.recruitable = np.random.rand(self.num_layers, w, w) < self.p
    self.tot_recruitable = len(np.where(self.recruitable)[0])
    self.tot_recruitable_active = 0
    self.tot_active = 0
    self.active = np.zeros([self.num_layers,w,w],bool)
    self.active_neighbors = np.zeros([self.num_layers,w,w],int)
    self.activated = []; # the recently active nodes

    if self.tot_recruitable > 0:
      while self.fraction_active() < 0.2:
        self.activate()

  def fraction_active(self):
    """ returns the fraction of potentially recruitable cells which are active """
    if self.tot_recruitable > 0:
      return float(self.tot_recruitable_active) / self.tot_recruitable
    else:
      return nan

  def propagate(self):
    """ propagate the activity if a valid node has been activated """
    # activated only has recruitable and currently inactive members
    while len(self.activated) > 0:
      act_l, act_x, act_y = self.activated.pop()
      self.active[act_l,act_x,act_y] = True
      self.tot_active += 1
      self.tot_recruitable_active += 1
      for l in range(self.num_layers):
        for x in range(int(act_x-self.r),int(act_x+self.r+1)):
          for y in range(int(act_y-self.r),int(act_y+self.r+1)):
            if distance(act_x,act_y,x,y) <= self.r:
              xi = x % self.width
              yi = y % self.width
              if l != act_l: # spread the activity across layers
                if np.random.rand() < self.trans: # transfer the activity
                  self.active_neighbors[l, xi,yi] += 1
              else: # if it is the same layer
                self.active_neighbors[l, xi,yi] += 1
              if self.active_neighbors[l, xi,yi] == self.t and \
                not self.active[l, xi,yi]:
                if self.recruitable[l, xi,yi]:
                  self.activated.append([l, xi,yi])
                else: # activate the node but don't propagate the activity
                  self.active[l,xi,yi] = True
                  self.tot_active += 1

  def activate(self):
    """ activate a random potentially active node """
    if self.fraction_active() > 0.95:
      return

    # pick a random point
    while True:
      l = np.random.randint(0,self.num_layers)
      x = np.random.randint(0,self.width)
      y = np. random.randint(0,self.width)
      if (self.recruitable[l,x,y] and not self.active[l,x,y]):
        break
    self.activated.append([l,x,y])
    self.propagate()

  def correlation(self):
    """ returns the correlation between the left and right images """
    # the total number of activations in common
    # same_count = len(where(self.active[0,:,:] == self.active[1,:,:])[0])
    # return float(same_count) / (self.width * self.width)

    # create an activity matrix of 0's and 1's (instead of True and False)
    if self.num_layers < 2:
      print("monocular models cannot have correlations between eye layers")
      return 0
    w = self.width
    active01 = np.zeros([2,w,w],int)
    active01[where(self.active)] = 1

    mean0 = active01[0,:,:].mean()
    mean1 = active01[1,:,:].mean()
    std0 = active01[0,:,:].std()
    std1 = active01[1,:,:].std()
    cov = ((active01[0,:,:] - mean0) * (active01[1,:,:] - mean1)).mean()
    return cov / (std0 * std1)

  def make_img_mat(self, show_img=True):
    """ return a matrix of 1's and 0's showing the activity in both layers """
    img_array = np.zeros([self.num_layers,self.width,self.width])
    border_width = 10 if self.num_layers > 1 else 0
    w = self.width
    for l in range(self.num_layers):
        img = np.zeros([w, w], float)
        for x in range(0,w-1):
            for y in range(0,w-1):
                if self.active[l,x,y]:
                    img[x,y] = 1

        img_array[l] = img
        #plt.imshow(img)
        #plt.show()

    return img_array






def generate_patches(num_patches, patch_size, lgn_width, lgn_p, lgn_r, lgn_t, lgn_a):
    half_comp = patch_size**2
    patch_count = 0

    while (patch_count < num_patches):
        L = LGN(width = lgn_width, p = lgn_p, r = lgn_r, t = lgn_t, trans = lgn_a, make_wave = True, num_layers=2)
        layer_activity = L.make_img_mat()
        patches_1 = np.array(skimage.extract_patches_2d(layer_activity[0], (patch_size, patch_size)))
        patches_2 = np.array(skimage.extract_patches_2d(layer_activity[1], (patch_size, patch_size)))
        reshaped_patches_1 = patches_1.reshape(-1, patches_1.shape[1]*patches_1.shape[1])
        reshaped_patches_2 = patches_2.reshape(-1, patches_2.shape[1]*patches_2.shape[1])
        composite_patches = np.concatenate((reshaped_patches_1,reshaped_patches_2),axis=1)
        blacklist = []
        for x in range(composite_patches.shape[0]):
            if composite_patches[x][:half_comp].std() == 0.0 or composite_patches[x][half_comp:].std() == 0.0:
                blacklist.append(x)
        composite_patches = np.delete(composite_patches, np.array(blacklist), axis=0)
        if (patch_count == 0):
            patch_base = composite_patches
        else:
            patch_base = np.append(patch_base, composite_patches, axis=0)
        patch_count = patch_base.shape[0]

    return patch_base[:num_patches]




def perform_ica(num_components, patches):
    # Run ICA on all the patches and return generated components
    ica_instance = FastICA(n_components=num_components, random_state=1,max_iter=1000, whiten='unit-variance') # note, sensitive to n_components
    icafit = ica_instance.fit(patches)
    ica_components = icafit.components_
    return ica_components


def generate_filters(num_filters, num_components, num_patches, patch_size, lgn_width, lgn_p, lgn_r, lgn_t, lgn_a):
    print("GENERATING FILTERS")
    bar = progressbar.ProgressBar(max_value=num_filters)
    filter_count = 0
    while (filter_count < num_filters):
        patches = generate_patches(num_patches, patch_size, lgn_width, lgn_p, lgn_r, lgn_t, lgn_a)
        filters = perform_ica(num_components, patches)
        if (filter_count == 0):
            filter_base = filters
        else:
            filter_base = np.append(filter_base, filters, axis=0)
        filter_count = filter_base.shape[0]
        if (filter_count < num_filters):
            bar.update(filter_count)
        else:
            bar.update(num_filters)

    return filter_base[:num_filters]

def unpack_filters(filters):
    half_filter = int(filters.shape[1]/2)
    filter_dim = int(np.sqrt(filters.shape[1]/2))
    first_eye = filters[:, 0:half_filter].reshape(-1,filter_dim,filter_dim)
    second_eye = filters[:, half_filter:].reshape(-1,filter_dim,filter_dim)
    return (first_eye, second_eye)

def linear_disparity(first_eye, second_eye):
    disparity_map = np.empty([first_eye.shape[0],first_eye.shape[1]*2])
    for index in range(first_eye.shape[0]):
        disparity = linear_convolution(first_eye[index], second_eye[index])
        disparity_map[index] = disparity
    return disparity_map


def normalize_disparity(disparity_map):
    sum_disparity = np.sum(disparity_map, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        normalized_disparity = disparity_map / sum_disparity
    return normalized_disparity


def generate_activity(autostereogram, asg_patch_size, first_eye, second_eye, disparity_map):
    print("\nCALCULATING ACTIVITY")
    bar = progressbar.ProgressBar(max_value=first_eye.shape[0])
    for index in range(first_eye.shape[0]):
        #make this more elegant
        convolution = double_convolve(first_eye[index], second_eye[index], autostereogram, asg_patch_size)
        scaled_activity = scale_disparity(convolution,disparity_map[index])
        if index == 0:
            summed_activity = scaled_activity
        else:
            summed_activity = summed_activity + scaled_activity
        bar.update(index)
    bar.update(first_eye.shape[0])
    return summed_activity



def estimate_depth(activity):
    print("\nESTIMATING DEPTH")
    depth_estimate = np.zeros([activity.shape[0],activity.shape[1]])
    bar = progressbar.ProgressBar(max_value=activity.shape[0])
    for x in range(activity.shape[0]):
        for y in range(activity.shape[1]):
            peak = np.abs(np.nanargmax(activity[x,y])-int(activity.shape[2]/2))
            peak = np.nanargmax(activity[x,y])
            depth_estimate[x,y] = peak
        bar.update(x)
    return depth_estimate

# num_filters, num_components, num_patches, patch_size, lgn_width, lgn_p, lgn_r, lgn_t, lgn_a
test = generate_filters(1000, 5, 7000, 8, 128, 0.128, 3, 2, 0.2)
t = unpack_filters(test)
dm = linear_disparity(t[0],t[1])
nd = normalize_disparity(dm)


# auto = open_norm("/content/shift5_70patch.png",verbose=False)
auto = open_norm("C:\\Users\\19404\\innate-binocular-vision\\original_shift5_70patch.png",verbose=False)
asg_patch_sz = 70
act = generate_activity(auto, asg_patch_sz, t[0], t[1], nd)
de = estimate_depth(act)
plt.imshow(de, cmap="Greys_r")
plt.show()

# dm = np.array(Image.open("/content/dm.png").convert("L"))
dm = np.array(Image.open("C:\\Users\\19404\\innate-binocular-vision\\original_dm.png").convert("L"))
dm = dm[0:200,:]
de = de[0:200,:]
corr = np.corrcoef(de.flatten(),dm.flatten())[0,1]
print("corr: ", corr)