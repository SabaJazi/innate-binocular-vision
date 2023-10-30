"""
Innate Binocular Vision (ibv) code to run the experiments.
"""

from PIL import Image, ImageOps
import time
import json
import hashlib
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import FastICA
from sklearn.feature_extraction import image as skimage
# from google.cloud import storage
from colorama import Fore,Style
import logging

def calculate_optimal_p(t, r, a):
    p = t / (((np.pi * (r**2)/2))*(1+a))
    return p


def generate_gabor(size, shift, sigma, rotation, phase_shift, frequency):
    radius = (int((size[0]/2.0)), int((size[1]/2.0)))
    # a BUG is fixed in this line
    [x, y] = np.meshgrid(range(-radius[0], radius[0]),
                         range(-radius[1], radius[1]))
    x = x - int(shift[0])
    y = y - int(shift[1])
    x = x * frequency
    y = y * frequency
    tmp = x * np.cos(rotation) + y * np.sin(rotation) + phase_shift
    radius = (int(size[0]/2.0), int(size[1]/2.0))
    # a BUG is fixed in this line
    [x, y] = np.meshgrid(range(-radius[0], radius[0]),
                         range(-radius[1], radius[1]))

    x = x - int(shift[0])
    y = y - int(shift[1])
    x1 = x * np.cos(rotation) + y * np.sin(rotation)
    y1 = -x * np.sin(rotation) + y * np.cos(rotation)

    sinusoid = np.cos(tmp)

    gauss = np.e * \
        np.exp(np.negative(
            0.5 * ((x1**2 / sigma[0]**2) + (y1**2 / sigma[1]**2))))
    gauss = gauss / 2*np.pi * sigma[0] * sigma[1]

    gabor = gauss * sinusoid
    return gabor


def open_norm(path, verbose=False):
    raw = np.array(Image.open(path).convert("L"))
    norm = (raw - np.mean(raw)) / np.std(raw)

    if verbose:
        return raw, norm
    else:
        return norm


def linear_convolution(center, slide):
    if (center.shape != slide.shape):
        return
    padded_slide = np.zeros((center.shape[0], center.shape[1]*3))
    padded_slide[0:, center.shape[1]:center.shape[1]*2] = center
    # plt.imshow(padded_slide,origin="lower")
    # plt.show()
    estimate = np.zeros([center.shape[1]*2])
    for x in range(center.shape[1]*2):
        dot = np.sum(padded_slide[0:, 0+x:center.shape[1]+x] * slide)
        estimate[x] = dot
    # plt.plot(estimate)
    # plt.show()
    return np.abs(estimate)


def double_convolve(normal, shifted, image, pupillary_distance):

    # CHECKOUT https://github.com/maweigert/gputools
    # probably VERY advantageous to switch over to GPU for convolutions!

    normal_convolved = signal.convolve2d(
        image, normal, boundary='symm', mode='same')
    shifted_convolved = signal.convolve2d(
        image, shifted, boundary='symm', mode='same')

    return_shape = image.shape

    realigned = np.zeros(return_shape)

    normal_convolved = normal_convolved[0:, 0:-pupillary_distance]
    shifted_convolved = shifted_convolved[0:, pupillary_distance:]

    mul = normal_convolved * shifted_convolved
    # plt.imshow(mul,cmap="nipy_spectral")
    # plt.show()

    # REMOVE BELOW COMMENTS TO THRESH SUBHALF VALUES
    low_values_flags = mul < 0  # mul.max()*0.5  # Where values are low
    mul[low_values_flags] = 0  # All low values set to 0
    realigned[0:, pupillary_distance:] = mul
    return np.abs(mul)


def scale_disparity(activity_map, disparity_map):
    scaled_disparity = np.zeros(
        [activity_map.shape[0], activity_map.shape[1], disparity_map.shape[0]])
    scaled_disparity[:, :] = disparity_map
    for x in range(activity_map.shape[0]):
        for y in range(activity_map.shape[1]):
            scaled_disparity[x, y] = activity_map[x, y] * \
                scaled_disparity[x, y]

    return scaled_disparity



# In[4]:

def generate_patches(num_patches, patch_size, lgn_width, lgn_p, lgn_r, lgn_t, lgn_a):
    print(Fore.GREEN + 'generate_patches:')
    print(Style.RESET_ALL)

    half_comp = patch_size**2
    patch_count = 0
    
    while (patch_count < num_patches):
        L = LGN(width=lgn_width, p=lgn_p, r=lgn_r, t=lgn_t, trans=lgn_a,
                make_wave=True, num_layers=2, random_seed=random.randint(1, 100))
        try:
            layer_activity = L.make_img_mat(patch_count)
        except ValueError as err:
            raise err

        patches_1 = np.array(skimage.extract_patches_2d(layer_activity[0], (patch_size, patch_size)))
        # plt.imshow(patches_1[0])
        # plt.title('patch sample from layer 1')
        # plt.show()
        patches_2 = np.array(skimage.extract_patches_2d(layer_activity[1], (patch_size, patch_size)))
        # plt.imshow(patches_2[0])
        # plt.title('patch sample from layer 2')
        # plt.show()
        reshaped_patches_1 = patches_1.reshape(-1,patches_1.shape[1]*patches_1.shape[1])
        reshaped_patches_2 = patches_2.reshape(-1,patches_2.shape[1]*patches_2.shape[1])
        composite_patches = np.concatenate((reshaped_patches_1, reshaped_patches_2), axis=1)

        # removing the patches that don't have variance in pixel values
        blacklist = []
        for x in range(composite_patches.shape[0]):
            if composite_patches[x][:half_comp].std() == 0.0 or composite_patches[x][half_comp:].std() == 0.0:
                blacklist.append(x)
        composite_patches = np.delete(composite_patches, np.array(blacklist), axis=0)
        # plt.imshow(composite_patches)
        # plt.title('Composit patch')
        # plt.show()
        if (patch_count == 0):
            patch_base = composite_patches
        else:
            patch_base = np.append(patch_base, composite_patches, axis=0)
        
        patch_count = patch_base.shape[0]
    
    return (patch_base[:num_patches], layer_activity)


# In[5]:

def perform_ica(num_components, patches):

    print(Fore.GREEN + 'perform_ica:')
    print(Style.RESET_ALL)
    # Run ICA on all the patches and return generated components
    # note, sensitive to n_components
    ica_instance = FastICA(n_components=num_components,
                           random_state=1, max_iter=1000000, whiten='standard')
    icafit = ica_instance.fit(patches)
    ica_components = icafit.components_
    return ica_components


# In[6]:

def generate_filters(num_filters, num_components, num_patches, patch_size, lgn_width, lgn_p, lgn_r, lgn_t, lgn_a):
    print(Fore.RED + 'generate_filters:')
    print(Style.RESET_ALL)

    filter_count = 0
    filter_base=[]
    while (filter_count < num_filters):
        # try:
        patches = generate_patches(
        num_patches, patch_size, lgn_width, lgn_p, lgn_r, lgn_t, lgn_a)
    # except ValueError as err:
            # raise err
        #actually bases not filters
        filters = perform_ica(num_components, patches[0])
        # print(filters)
        if (filter_count == 0):
            print(filters)
            # filter_base = np.append(filter_base,filters, axis=0)
            filter_base = filters

            filter_count = filter_base.shape[0]
            print('filter count:' ,filter_count)

        else:
            # filter_base = np.append(filter_base, filters, axis=0)
            filter_base = np.vstack([filter_base,filters])
            filter_count = filter_base.shape[0]
            print('filter count:' ,filter_count)
            print('filters(base) array shape: ',filter_base.shape )


    return (filter_base[:num_filters], patches[0], patches[1])


# In[7]:

def unpack_filters(filters):
    print(Fore.RED + 'Unpacking filters:')
    print(Style.RESET_ALL)

    half_filter = int(filters.shape[1]/2)
    filter_dim = int(np.sqrt(filters.shape[1]/2))
    first_eye = filters[:, 0:half_filter].reshape(-1, filter_dim, filter_dim)
    second_eye = filters[:, half_filter:].reshape(-1, filter_dim, filter_dim)
    return (first_eye, second_eye)


# In[8]:

def linear_disparity(first_eye, second_eye):
    print(Fore.RED + 'linear_disparity:')
    print(Style.RESET_ALL)

    disparity_map = np.empty([first_eye.shape[0], first_eye.shape[1]*2])
    for index in range(first_eye.shape[0]):
        disparity = linear_convolution(first_eye[index], second_eye[index])
        disparity_map[index] = disparity
    return disparity_map


# In[9]:

def normalize_disparity(disparity_map):
    with np.errstate(divide='ignore', invalid='ignore'):
        #normalize_disparity = (disparity_map - np.mean(disparity_map, axis=0)) / np.std(disparity_map)
        normalized_disparity = (disparity_map / np.mean(disparity_map, axis=0))

        #sum_normalized_disparity = np.sum(normalized_disparity, axis=0)
        #double_normalized_disparity = normalized_disparity / sum_normalized_disparity
    return normalized_disparity


# In[10]:

def generate_activity(autostereogram, asg_patch_size, first_eye, second_eye, disparity_map):
    for index in range(first_eye.shape[0]):
        # make this more elegant
        convolution = double_convolve(
            first_eye[index], second_eye[index], autostereogram, asg_patch_size)
        scaled_activity = scale_disparity(convolution, disparity_map[index])
        if index == 0:
            summed_activity = scaled_activity
        else:
            summed_activity = summed_activity + scaled_activity
    return summed_activity


# In[11]:

def estimate_depth(activity):
    depth_estimate = np.zeros([activity.shape[0], activity.shape[1]])
    for x in range(activity.shape[0]):
        for y in range(activity.shape[1]):
            peak = int(
                np.abs(np.nanargmax(activity[x, y])-int(activity.shape[2]/2)))
            #peak = np.nanargmax(activity[x,y])
            depth_estimate[x, y] = peak
    return depth_estimate


# In[12]:

def save_array(input_array, path):
    cast_array = (255.0 / input_array.max() *
                  (input_array - input_array.min())).astype(np.uint8)
    save_image = Image.fromarray(cast_array)
    save_image.save(path)



# In[14]:

# In[15]:

def disparity_distribution(disparity_map):
    print(Fore.RED + 'disparity_distribution:')
    print(Style.RESET_ALL)

    dist = np.empty([disparity_map.shape[0]])
    for x in range(disparity_map.shape[0]):
        peak = np.abs(np.nanargmax(
            disparity_map[x])-int(disparity_map.shape[1]/2))
        dist[x] = int(peak)
    return dist


# In[16]:

def run_experiment(num_filters, num_components, num_patches, patch_size, lgn_width, lgn_p, lgn_r, lgn_t, lgn_a, autostereogram, asg_patch_size, groundtruth, experiment_folder):
    print(Fore.RED + 'run_experiment:')
    print(Style.RESET_ALL)
    autostereogram = open_norm(autostereogram,verbose=False)
    groundtruth = np.array(Image.open(groundtruth).convert("L"))

    filters = generate_filters(num_filters, num_components, num_patches,
                               patch_size, lgn_width, lgn_p, lgn_r, lgn_t, lgn_a)
    split_filters = unpack_filters(filters)
    disparity_map = linear_disparity(split_filters[0], split_filters[1])

    plt.hist(disparity_distribution(disparity_map))
    plt.title('disparity histogram')
    plt.show()

    #normalized_disparity = disparity_map

    normalized_disparity = normalize_disparity(disparity_map)
    plt.hist(disparity_distribution(normalized_disparity))
    plt.title('normalized disparity histogram')

    plt.show()

    activity = generate_activity(autostereogram, asg_patch_size,
                                 split_filters[0], split_filters[1], normalized_disparity)
    depth_estimate = estimate_depth(activity)
    correlation = np.corrcoef(depth_estimate.flatten(),
                              groundtruth.flatten())[0, 1]
    current_time = time.localtime()
  
    save_array(depth_estimate, "im.png")

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
        "id": 1 #ident_hash
    }


    return params



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

    def make_img_mat(self,p_c, show_img=True):
        # print(Fore.RED + 'make_img_mat:')
        # print(Style.RESET_ALL)

        """ return a matrix of 1's and 0's showing the activity in both layers """
        percentage_active = float(self.active.sum()) / self.allcells
        
        print(p_c,': percentage_active:'  ,percentage_active)
        if percentage_active < 0.05:
            print('LGN: activity less than low bound')
            raise ValueError('LGN: activity less than low bound')
        if percentage_active > 0.99:
            print('LGN: activity greater than high bound')
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
                        # Defines a 3x3 convolution kernel
                        normal = np.array([[1,1,1],[1,0,1],[1,1,1]])
                        conv2d = signal.convolve2d(img, normal, boundary='symm', mode='same')
                        thresh = 4.0
                        conv2d[np.where(conv2d < thresh)]  = 0
                        conv2d[np.where(conv2d >= thresh)]  = 1
                        conv = conv2d

            img_array[l] = conv
            # Next line shows each of activity patern 1 by 1
            # plt.imshow(img)
            # plt.title("LGN activity 64x64 layer {}".format(l+1))

            # plt.show()

            # plt.imshow(conv)
            # plt.title("Convolved LGN activity layer {}".format(l+1))

            # plt.show()

        return img_array


def local_experiment(experiment_subparameters, patch_max, filter_max):
    print(Fore.BLUE + 'local_experiment:')
    print(Style.RESET_ALL)

    current_dir = os.getcwd()
    depthmap_path =  experiment_subparameters["depthmap_path"]
    # depthmap_path = os.path.join(current_dir, experiment_subparameters["depthmap_path"])

    autostereogram_path = experiment_subparameters["autostereogram_path"]
    # autostereogram_path = os.path.join(current_dir,experiment_subparameters["autostereogram_path"])

    autostereogram = open_norm(autostereogram_path, verbose=False)
    groundtruth = np.array(Image.open(depthmap_path).convert("L"))

    # try:
    res = generate_filters(experiment_subparameters["num_filters"], experiment_subparameters["num_components"], experiment_subparameters["num_patches"],
                            experiment_subparameters["patch_size"],
                            experiment_subparameters["lgn_size"],
                            experiment_subparameters["lgn_parameters"]['lgn_a'], 
                            experiment_subparameters["lgn_parameters"]['lgn_r'] ,experiment_subparameters["lgn_parameters"]['lgn_p'], experiment_subparameters["lgn_parameters"]['lgn_t'])
        # except ValueError as err:
    #     raise err

    filters = res[0]
    patches = res[1].reshape(-1, experiment_subparameters["patch_size"], experiment_subparameters["patch_size"])
    lgn = res[2]

    split_filters = unpack_filters(filters)


    disparity_map = linear_disparity(split_filters[0], split_filters[1])
    normalized_disparity = normalize_disparity(disparity_map)
    activity = generate_activity(autostereogram, experiment_subparameters["autostereogram_patch"], split_filters[0], split_filters[1], normalized_disparity)
    depth_estimate = estimate_depth(activity)


    correlation = np.corrcoef(depth_estimate.flatten(), groundtruth.flatten())[0, 1]

    experiment_subparameters["correlation"] = correlation
    return experiment_subparameters
# -----------------------------------run-----------------------------------

experiment_subparameters = {
    "depthmap_path": r"C:\vscode\innate-binocular-vision\innate-binocular-vision\dm.png",
    "autostereogram_path": r"C:\vscode\innate-binocular-vision\innate-binocular-vision\autostereogram.png",
    # "num_filters": 2000,
    "num_filters": 200,
    "num_components": 20,
    "num_patches": 1000,
    "patch_size": 8,
    "lgn_size": 64,
    # "lgn_parameters":[[0.5, 1.5 , 10], [4, 4, 1],[1, 4, 8], [0.05 ,0.05, 1]],
    "lgn_parameters":
        {
            "lgn_a": 0.5,
            "lgn_r": 2.0,
            "lgn_p": 0.592,
            "lgn_t": 1.0,
            "name": "a0.05_r1.00_p0.592_t1.00"
        },
    "lgn_dump": r"C:\vscode\innate-binocular-vision\innate-binocular-vision",
    "filter_dump": r"C:\vscode\innate-binocular-vision\innate-binocular-vision",
    "patch_dump": r"C:\vscode\innate-binocular-vision\innate-binocular-vision",
    "autostereogram_patch":r"C:\vscode\innate-binocular-vision\innate-binocular-vision",
    "activity_dump":r"C:\vscode\innate-binocular-vision\innate-binocular-vision",
    "correlation":r"C:\vscode\innate-binocular-vision\innate-binocular-vision"
   
}
# [[0.5 1.5 10], [4 4 1] ,[1 4 8], [0.05 0.05 1]]
# Set the maximum values for patch and filter 
# (original path_max=100000, original filter=200)
patch_max = 10000
filter_max = 20

# Call the local_experiment function

result = local_experiment(experiment_subparameters, patch_max, filter_max)

# Print the result or perform other actions as needed
print("Experiment result:", result)