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

def generate_patches(num_patches, patch_size, lgn_width, lgn_p, lgn_r, lgn_t, lgn_a):
    #here we creat LGN patern patches
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



def generate_filters(num_filters, num_components, num_patches, patch_size, lgn_width, lgn_p, lgn_r, lgn_t, lgn_a):
    print(Fore.RED + 'generate_filters:')
    print(Style.RESET_ALL)

    filter_count = 0
    filter_base=[]
    while (filter_count < num_filters):
        # try:
        patches = generate_patches(
        num_patches, patch_size, lgn_width, lgn_p, lgn_r, lgn_t, lgn_a)

    return (filter_base[:num_filters], patches[0], patches[1])


def save_array(input_array, path):
    cast_array = (255.0 / input_array.max() *
                  (input_array - input_array.min())).astype(np.uint8)
    save_image = Image.fromarray(cast_array)
    save_image.save(path)


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
           #---- just to visualize---
        plt.imshow(self.recruitable[0],cmap="gray")
        plt.title("recruitable nodes layer 1")
        plt.colorbar()
        plt.show()
        #-------------------
        self.tot_recruitable = len(np.where(self.recruitable)[0])
        self.tot_recruitable_active = 0
        self.tot_active = 0
        self.active = np.zeros([self.num_layers, w, w], bool)
        #---- just to visualize---
        plt.imshow(self.active[0],cmap="gray")
        plt.title("Start LGN activity layer 1")
        plt.colorbar()
        plt.show()
        #-------------------
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
            #---- just to visualize---
            # plt.imshow(self.active[0],cmap="gray")
            # plt.title("activated nodes in layer 1")
            # plt.colorbar()
            # plt.show()
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # 1 row, 2 columns

            # Plot the first image on the first subplot
            axes[0].imshow(self.active[0], cmap="gray")
            axes[0].set_title("Layer 1")
            axes[0].set_axis_off()  # Optional: Turn off axes

            # Plot the second image on the second subplot
            axes[1].imshow(self.active[1], cmap="gray")
            axes[1].set_title("Layer 2")
            axes[1].set_axis_off()  # Optional: Turn off axes

            # Show the plot
            plt.show()
            #-------------------
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

            plt.imshow(conv)
            plt.title("Convolved LGN activity layer {}".format(l+1))

            plt.show()

        return img_array


def local_experiment(experiment_subparameters, patch_max, filter_max):
    print(Fore.BLUE + 'local_experiment:')
    print(Style.RESET_ALL)

    res = generate_filters(experiment_subparameters["num_filters"], experiment_subparameters["num_components"], experiment_subparameters["num_patches"],
                            experiment_subparameters["patch_size"],
                            experiment_subparameters["lgn_size"],
                            experiment_subparameters["lgn_parameters"]['lgn_a'], 
                            experiment_subparameters["lgn_parameters"]['lgn_r'] ,experiment_subparameters["lgn_parameters"]['lgn_p'], experiment_subparameters["lgn_parameters"]['lgn_t'])
 
    filters = res[0]
    patches = res[1].reshape(-1, experiment_subparameters["patch_size"], experiment_subparameters["patch_size"])
    lgn = res[2]



    return experiment_subparameters
# -----------------------------------run-----------------------------------

experiment_subparameters = {
    "depthmap_path": r"C:\vscode\innate-binocular-vision\innate-binocular-vision\dm.png",
    "autostereogram_path": r"C:\vscode\innate-binocular-vision\innate-binocular-vision\autostereogram.png",
    # "num_filters": 2000,
    "num_filters": 200,
    "num_components": 75,
    "num_patches": 5000,
    "patch_size": 16,
    "lgn_size": 64,
    # "lgn_parameters":[[0.5, 1.5 , 10], [4, 4, 1],[1, 4, 8], [0.05 ,0.05, 1]],
    "lgn_parameters":
        # {
        #     "lgn_a": 0.5,
        #     "lgn_r": 2.0,
        #     "lgn_p": 0.592,
        #     "lgn_t": 1.0,
        #     "name": "a0.05_r1.00_p0.592_t1.00"
        # },
        {
            "lgn_a": 0.5,
            "lgn_r": 3.0,
            "lgn_p": 0.48,
            "lgn_t": 5.0,
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