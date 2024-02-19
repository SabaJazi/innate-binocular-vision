from PIL import Image, ImageOps, ImageFilter
import time
import json
import pylab
import hashlib
import progressbar
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
from random import randint
import progressbar

from scipy import signal
from scipy.interpolate import griddata
from sklearn.decomposition import FastICA
from sklearn.feature_extraction import image as skimage
from ipywidgets import interact, interactive, fixed
%matplotlib inline

def distance(x0, y0, x1, y1):
    return np.sqrt(pow(x0-x1, 2) + pow(y0-y1, 2))

class LGN:
    """
    this class defines a model which generates binocular spontaneous activity
    """

    def __init__(self, width=128, p=0.5, r=1.0, t=1, trans=0.0,
        make_wave=True, num_layers=2, random_seed=0):
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
        self.allcells = (self.num_layers * w * w)
        self.recruitable = np.random.rand(self.num_layers, w, w) < self.p
        self.tot_recruitable = len(np.where(self.recruitable)[0])
        self.tot_recruitable_active = 0
        self.tot_active = 0
        self.active = np.zeros([self.num_layers, w, w], bool)
        self.active_neighbors = np.zeros([self.num_layers, w, w], int)
        self.activated = []  # the recently active nodes

        if self.tot_recruitable > 0:
            # changed active threshold from 20% to 1%
            while self.fraction_active() < 0.20:
                self.activate()

    def fraction_active(self):
        """ returns the fraction of potentially recruitable cells which are active """
        if self.tot_recruitable > 0:
            return float(self.tot_recruitable_active) / self.tot_recruitable
        else:
            return float('NaN')

    def propagate(self):
        """ propagate the activity if a valid node has been activated """
        # activated only has recruitable and currently inactive members
        while len(self.activated) > 0:
            act_l, act_x, act_y = self.activated.pop()
            self.active[act_l, act_x, act_y] = True
            self.tot_active += 1
            self.tot_recruitable_active += 1
            for l in range(self.num_layers):
                for x in range(int(act_x-self.r), int(act_x+self.r+1)):
                    for y in range(int(act_y-self.r), int(act_y+self.r+1)):
                        if distance(act_x, act_y, x, y) <= self.r:
                            xi = x % self.width
                            yi = y % self.width
                            if l != act_l:  # spread the activity across layers
                                if np.random.rand() < self.trans:  # transfer the activity
                                    self.active_neighbors[l, xi, yi] += 1
                            else:  # if it is the same layer
                                self.active_neighbors[l, xi, yi] += 1
                            if self.active_neighbors[l, xi, yi] == self.t and not self.active[l, xi, yi]:
                                if self.recruitable[l, xi, yi]:
                                    self.activated.append([l, xi, yi])
                                else:  # activate the node but don't propagate the activity
                                    self.active[l, xi, yi] = True
                                    self.tot_active += 1

    def activate(self):
        """ activate a random potentially active node """
        if self.fraction_active() > 0.95:
            return

        # pick a random point
        while True:
            l = np.random.randint(0, self.num_layers)
            x = np.random.randint(0, self.width)
            y = np. random.randint(0, self.width)
            if (self.recruitable[l, x, y] and not self.active[l, x, y]):
                break
        self.activated.append([l, x, y])
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
        active01 = np.zeros([2, w, w], int)
        active01[np.where(self.active)] = 1

        mean0 = active01[0, :, :].mean()
        mean1 = active01[1, :, :].mean()
        std0 = active01[0, :, :].std()
        std1 = active01[1, :, :].std()
        cov = ((active01[0, :, :] - mean0) *
               (active01[1, :, :] - mean1)).mean()
        return cov / (std0 * std1)

    def make_img_mat(self, show_img=True):
        """ return a matrix of 1's and 0's showing the activity in both layers """
        percentage_active = float(self.active.sum()) / self.allcells
        print('\npercent active: ', percentage_active)
        if percentage_active < 0.05:
            print('LGN: activity less than low bound\n')
            raise ValueError('LGN: activity less than low bound')
        if percentage_active > 0.99:
            print('LGN: activity greater than high bound\n')
            raise ValueError('LGN: activity greater than high bound')


        img_array = np.zeros([self.num_layers, self.width, self.width])
        w = self.width

        for l in range(self.num_layers):
            img = np.zeros([w, w], float)
            conv = 0
            for x in range(0, w-1):
                for y in range(0, w-1):
                    if self.active[l, x, y]:
                        img[x, y] = 1
                        normal = np.array([[1,1,1],[1,0,1],[1,1,1]])
                        #  here is where things get slowly
                        conv2d = signal.convolve2d(img, normal, boundary='symm', mode='same')
                        thresh = 4.0
                        conv2d[np.where(conv2d < thresh)]  = 0
                        conv2d[np.where(conv2d >= thresh)]  = 1
                        conv = conv2d

            img_array[l] = conv
            # plt.imshow(img)
            # plt.show()

        return img_array

def save_array(input_array, path):
    cast_array = (255.0 / input_array.max() * (input_array - input_array.min())).astype(np.uint8)
    save_image = Image.fromarray(cast_array)
    # colorized_image = ImageOps.colorize(save_image, (0,0,0), (0,255,0))
    # colorized_image.save(path)
    save_image.save(path)
    # print("SAVING ESTIMATED DEPTHMAP TO: %s" % (path))

def save_LRactivity(layer_activity,patch_count,ident_hash):
  Path(parent_path+"/images/activity/{}/l".format(ident_hash)).mkdir(parents=True, exist_ok=True)
  Path(parent_path+"/images/activity/{}/r".format(ident_hash)).mkdir(parents=True, exist_ok=True)

  filename_r = "r{}.png".format(patch_count)
  filename_l = "l{}.png".format(patch_count)

  # generated_activity = layer_activity
  # activity_differences = np.abs(layer_activity[0]-layer_activity[1])
  save_array(layer_activity[0], 
             parent_path+"/images/activity/{}/r/".format(ident_hash)+filename_r)
 
  save_array(layer_activity[1], 
             parent_path+"/images/activity/{}/l/".format(ident_hash)+filename_r)
  
  print("SAVING ACTIVITY TO: %s" % (parent_path+"/images/activity/{}".format(ident_hash)))


def save_LRfilters(first_eye,second_eye,ident_hash):
  Path(parent_path+"/images/filters/{}/l".format(ident_hash)).mkdir(parents=True, exist_ok=True)
  Path(parent_path+"/images/filters/{}/r".format(ident_hash)).mkdir(parents=True, exist_ok=True)

  # for filter_count in range(first_eye.shape[0]):
  for filter_count in range(16):

    filename_r = "r{}.png".format(filter_count)
    filename_l = "l{}.png".format(filter_count)

    save_array(first_eye[filter_count], 
                parent_path+"/images/filters/{}/r/".format(ident_hash)+filename_r)

    save_array(second_eye[filter_count], 
                parent_path+"/images/filters/{}/l/".format(ident_hash)+filename_r)
    
    print("SAVING FILTERS TO: %s" % (parent_path+"/images/filters/{}".format(ident_hash)+filename_r))


def generate_ident_hash(num_filters, num_components, num_patches, patch_size, lgn_width, lgn_p, lgn_r, lgn_t, lgn_a, current_time):
    input_string = "%f%f%f%f%f%f%f%f%f%f" % (num_filters, num_components, num_patches, patch_size, lgn_width, lgn_p, lgn_r, lgn_t, lgn_a, current_time)
    output_hash = hashlib.sha256(input_string.encode('utf-8')).hexdigest()
    return output_hash[:20]

def open_norm(path,verbose=False):
    raw = np.array(Image.open(path).convert("L"))
    norm = (raw - np.mean(raw)) / np.std(raw)

    if verbose:
        return raw, norm
    else:
        return norm

def calculate_optimal_p(t, r, a):
    p = t / (((np.pi * (r**2)/2))*(1+a))
    return p

def double_convolve(normal, shifted, image, pupillary_distance):

    #CHECKOUT https://github.com/maweigert/gputools
    #probably VERY advantageous to switch over to GPU for convolutions!

    normal_convolved = signal.convolve2d(
        image, normal, boundary='symm', mode='same')
    shifted_convolved = signal.convolve2d(
        image, shifted, boundary='symm', mode='same')

    return_shape = image.shape

    realigned = np.zeros(return_shape)

    normal_convolved = normal_convolved[0:,0:-pupillary_distance]
    shifted_convolved = shifted_convolved[0:,pupillary_distance:]

    diff = np.subtract(normal_convolved, shifted_convolved)
    mul = normal_convolved * shifted_convolved
    #plt.imshow(mul,cmap="nipy_spectral")
    #plt.show()

    #REMOVE BELOW COMMENTS TO THRESH SUBHALF VALUES
    low_values_flags = mul < 0 #mul.max()*0.5  # Where values are low
    mul[low_values_flags] = 0  # All low values set to 0
    realigned[0:,pupillary_distance:] = mul
    return np.abs(mul)

def scale_disparity(activity_map, disparity_map):
    scaled_disparity = np.zeros(
        [activity_map.shape[0],activity_map.shape[1],disparity_map.shape[0]])
    scaled_disparity[:,:] = disparity_map
    for x in range(activity_map.shape[0]):
        for y in range(activity_map.shape[1]):
            scaled_disparity[x,y] = activity_map[x,y] * scaled_disparity[x,y]

    return scaled_disparity

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
            peak = int(np.abs(np.nanargmax(activity[x,y])-int(activity.shape[2]/2)))
            #peak = np.nanargmax(activity[x,y])
            depth_estimate[x,y] = peak
        bar.update(x)
    return depth_estimate
def generate_patches(num_patches, patch_size, lgn_width, lgn_p, lgn_r,
                     lgn_t, lgn_a,ident_hash):
    half_comp = patch_size**2
    patch_count = 0

    while (patch_count < num_patches):
        # L = LGN(width = lgn_width, p = lgn_p, r = lgn_r, t = lgn_t, trans = lgn_a, make_wave = True, num_layers=2, random_seed=randint(1,100))
        L = LGN(width = lgn_width, p = lgn_p, r = lgn_r, t = lgn_t, trans = lgn_a, make_wave = True, num_layers=2)

        # layer_activity = L.make_img_mat()
        try:
            layer_activity = L.make_img_mat()
            save_LRactivity(layer_activity,patch_count,ident_hash)
        except ValueError as err:
            raise err
        # print(err.args)
        patches_1 = np.array(skimage.extract_patches_2d(layer_activity[0], (patch_size, patch_size)))
        patches_2 = np.array(skimage.extract_patches_2d(layer_activity[1], (patch_size, patch_size)))
        reshaped_patches_1 = patches_1.reshape(-1,patches_1.shape[1]*patches_1.shape[1])
        reshaped_patches_2 = patches_2.reshape(-1,patches_2.shape[1]*patches_2.shape[1])
        composite_patches = np.concatenate((reshaped_patches_1, reshaped_patches_2), axis=1)
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

    return (patch_base[:num_patches], layer_activity)

def perform_ica(num_components, patches):
    # Run ICA on all the patches and return generated components
    # added whiten='unit-variance'to prevent warning, not sure if it is the correct one, need to research
    ica_instance = FastICA(n_components=num_components, random_state=1,max_iter=1000000,whiten='unit-variance') # note, sensitive to n_components
    icafit = ica_instance.fit(patches)
    ica_components = icafit.components_
    return ica_components

def generate_filters(num_filters, num_components, num_patches, patch_size,
                     lgn_width, lgn_p, lgn_r, lgn_t, lgn_a,ident_hash):
    print("GENERATING FILTERS")
    bar = progressbar.ProgressBar(max_value=num_filters)
    filter_count = 0
    while (filter_count < num_filters):
        # patches = generate_patches(num_patches, patch_size, lgn_width, lgn_p, lgn_r, lgn_t, lgn_a)
        try:
            patches = generate_patches(
            num_patches, patch_size, lgn_width, lgn_p, lgn_r, lgn_t,
            lgn_a,ident_hash)
        except ValueError as err:
            raise err
        filters = perform_ica(num_components, patches[0])
        if (filter_count == 0):
            filter_base = filters
        else:
            filter_base = np.append(filter_base, filters, axis=0)
        filter_count = filter_base.shape[0]
        print(filter_count, end=" ")
        if (filter_count < num_filters):
          bar.update(filter_count)
        else:
          bar.update(num_filters)

    return (filter_base[:num_filters], patches[0], patches[1])
    # return filter_base[:num_filters]

def unpack_filters(filters,ident_hash):
    half_filter = int(filters.shape[1]/2)
    filter_dim = int(np.sqrt(filters.shape[1]/2))
    first_eye = filters[:, 0:half_filter].reshape(-1,filter_dim,filter_dim)
    second_eye = filters[:, half_filter:].reshape(-1,filter_dim,filter_dim)
    save_LRfilters(first_eye,second_eye,ident_hash)
    return (first_eye, second_eye)

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

def linear_disparity(first_eye, second_eye):
    disparity_map = np.empty([first_eye.shape[0],first_eye.shape[1]*2])
    for index in range(first_eye.shape[0]):
        disparity = linear_convolution(first_eye[index], second_eye[index])
        disparity_map[index] = disparity
    return disparity_map

def normalize_disparity(disparity_map):
    with np.errstate(divide='ignore', invalid='ignore'):
        #normalize_disparity = (disparity_map - np.mean(disparity_map, axis=0)) / np.std(disparity_map)
        normalized_disparity = (disparity_map / np.mean(disparity_map, axis=0))

        #sum_normalized_disparity = np.sum(normalized_disparity, axis=0)
        #double_normalized_disparity = normalized_disparity / sum_normalized_disparity
    return normalized_disparity

def run_experiment(num_filters, num_components, num_patches, patch_size, lgn_width, lgn_p, lgn_r, lgn_t, lgn_a, autostereogram, asg_patch_size, groundtruth, experiment_folder):
    ident_hash = generate_ident_hash(num_filters, num_components, num_patches, patch_size, lgn_width, lgn_p, lgn_r, lgn_t, lgn_a, time.time())

    res  = generate_filters(num_filters, num_components, num_patches, patch_size, lgn_width, lgn_p, lgn_r, lgn_t, lgn_a,ident_hash)
    # next 3 lines add too much time to the process, avoid such code!
    # filters = res[0]
    # patches = res[1].reshape(-1, patch_size,patch_size)
    # lgn = res[2]
    split_filters = unpack_filters(res[0],ident_hash)
    disparity_map = linear_disparity(split_filters[0],split_filters[1])


    #normalized_disparity = disparity_map

    normalized_disparity = normalize_disparity(disparity_map)
    #plt.hist(disparity_distribution(normalized_disparity))
    #plt.show()

    activity = generate_activity(autostereogram, asg_patch_size, split_filters[0], split_filters[1], normalized_disparity)

    depth_estimate = estimate_depth(activity)
    correlation = np.corrcoef(depth_estimate.flatten(),groundtruth.flatten())[0,1]
    current_time = time.localtime()
    # image_path = "%s/%s.png" % (experiment_folder, ident_hash)
    image_path = "%s/images/depthmaps/%s.png" % (experiment_folder, ident_hash)

    # data_path = "%s/%s.json" % (experiment_folder, ident_hash)
    data_path = "%s/json/%s.json" % (experiment_folder, ident_hash)

    save_array(depth_estimate, image_path)
    params = {
        "num_filters": num_filters,
        "num_components": num_components,
        "num_patches": num_patches,
        "patch_size": patch_size,
        "lgn_width": lgn_width,
        "lgn_p": lgn_p,
        "lgn_r": lgn_r,
        "lgn_t": lgn_t,
        "lgn_a": lgn_a,
        "corr": np.abs(correlation),
        "time": time.strftime('%a, %d %b %Y %H:%M:%S GMT', current_time),
        "id": ident_hash
    }
    with open(data_path, 'w') as file:
        file.write(json.dumps(params))

    return params


from datetime import date
 
# Returns the current local date
today = date.today()
print("Today date is: ", today)

from pathlib import Path
# parent_path="/content"
main_path="/content/drive/MyDrive/lab/LGN-IBV/results/"

Path(main_path+str(today)).mkdir(parents=True, exist_ok=True)
parent_path="/content/drive/MyDrive/lab/LGN-IBV/results/"+str(today)

auto = open_norm(main_path+"shift5_70patch.png",verbose=False)
gt = np.array(Image.open(main_path+"/dm.png").convert("L"))

#
# order of variables:
# num_filters, num_components, num_patches, patch_size,
# lgn_width, lgn_p, lgn_r, lgn_t, lgn_a, autostereogram, asg_patch_size,
# groundtruth, experiment_folder
#
# check if these folders are made, and if not, creat them before running the code
Path(parent_path+"/images").mkdir(parents=True, exist_ok=True)
Path(parent_path+"/json").mkdir(parents=True, exist_ok=True)
Path(parent_path+"/images/depthmaps").mkdir(parents=True, exist_ok=True)


r = 2
pshift = 0.01
a=0.1
t=2

p = calculate_optimal_p(t,r,a) + pshift

x = run_experiment(100, 20, 10000, 8, 128, p, r, t, a, auto, 70, gt, parent_path)