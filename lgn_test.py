
from PIL import Image, ImageOps
import time
import json
import hashlib
import numpy as np
import random
import os

from scipy import signal
from sklearn.decomposition import FastICA
from sklearn.feature_extraction import image as skimage
import matplotlib.pyplot as plt

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
        print('percent active: ', percentage_active)
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

# ----------------------------------------------------
# L = LGN(width=64, p=0.15, r=4, t=5, trans=0.0,
#             make_wave=True, num_layers=2)

# generated_activity = L.make_img_mat()
# activity_differences = np.abs(generated_activity[0]-generated_activity[1])
# # print(activity_differences)
# ----------------------------------------
# p_lin = np.linspace(0.0,0.5,5)
# for a in p_lin:
#     L = LGN(width=256, p=0.15, r=4, t=5, trans=a,
#             make_wave=True, num_layers=2)

#     generated_activity = L.make_img_mat()
#     activity_differences = np.abs(generated_activity[0]-generated_activity[1])
#     differences = np.abs(generated_activity[0]-generated_activity[1])
#     fig, (layer_1, layer_2, diff) = plt.subplots(1, 3, sharex=True)
    
#     layer_1.axis("off")
#     layer_2.axis("off")
#     diff.axis("off")

#     layer_1.imshow(generated_activity[0], cmap="Greys_r")
#     layer_2.imshow(generated_activity[1], cmap="Greys_r")
#     diff.imshow(activity_differences, cmap="Reds_r")
    
#     filename = "p_0.15-r_4.0-t_5.0_a_{}.png".format(a)

#     plt.savefig(filename,dpi=300)

#     plt.show()
# ------------------------------------------
p_lin = np.linspace(0.0,0.5,5)
t_lin = np.linspace(3,8,6)
print('p line' , p_lin)
print('t line', t_lin)
fig, subs = plt.subplots(6, 6, sharex=True)
for idx_p, p in enumerate(p_lin):
    for idx_t, t in enumerate(t_lin):
        print('p = ',p,'t = ',t)
        L =LGN(width=256, p=p, r=4, t=t, trans=0.5,
                make_wave=True, num_layers=2)

        try:
            generated_activity = L.make_img_mat()
        except ValueError as err:
            if (str(err) == "LGN: activity greater than high bound"):
                generated_activity =  np.full((2,256, 256), 1.0)
                generated_activity[0][0][0] = 0


            else:
                generated_activity =  np.full((2,256, 256), 0.0)
                generated_activity[0][0][0] = 1


        subs[idx_p][idx_t].axis("off")
        subs[idx_p][idx_t].set_aspect(aspect=100)

        generated_activity = generated_activity[:,:,:-5]
        subs[idx_p][idx_t].imshow(generated_activity[0], cmap="winter")


filename = "lgn_pt_variance.png"
plt.savefig(filename,dpi=300)