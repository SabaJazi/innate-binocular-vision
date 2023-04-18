#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pylab as py
#%matplotlib inline
from IPython.display import Image, Audio
import math
import PIL
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import rcParams
from matplotlib import pylab
from scipy import ndimage
import matplotlib
import wave
import sys
import random as random


# this function collects patches from black and white images
def collectPatchesBW(numPatches, patchWidth, filePath):
    maxTries = numPatches * 50
    firstPatch = 0 # the first patch number accepted from an image
    firstTry = 0 # the first attempt to take a patch from the image
    patchCount = 0 # number of collected patches
    tryCount = 0 # number of attempted collected patches
    numPixels = patchWidth * patchWidth
    patchSample = np.zeros([patchWidth,patchWidth],'double')
    patch = np.zeros([numPixels,1],'double')
    imgPatches = np.zeros([numPixels,numPatches],'double')
    # chooses the image that we're sampling from
    imgCount = 1
    image = PIL.Image.open(filePath + str(imgCount) + '.jpg')
    imageHeight, imageWidth, imageChannels = matplotlib.pyplot.imread(filePath + str(imgCount) + '.jpg').shape
    image = image.convert('L')
    image = np.asarray(image, 'double').transpose()
    # normalizing the image
    image -= image.mean()
    image /= image.std()
    while patchCount < numPatches and tryCount < numPatches:
        tryCount += 1
        if (tryCount - firstTry) > maxTries/2 or (patchCount - firstPatch) > numPatches/2:
        # change the image sampled from to the next in the folder
            imgCount += 1
            image = PIL.Image.open(filePath + str(imgCount) + '.jpg')
            imageHeight, imageWidth, imageChannels = matplotlib.pyplot.imread(filePath + str(imgCount) + '.jpg').shape
            image = image.convert('L')
            image = np.asarray(image, 'double').transpose()
            # normalizing the image
            image -= image.mean()
            image /= image.std()
            firstPatch = patchCount
            firstTry = tryCount
        #starts patch collection in a random space
        px = np.random.randint(0,imageWidth - patchWidth)
        py = np.random.randint(0,imageHeight - patchWidth)
        patchSample = image[px:px+patchWidth,py:py+patchWidth].copy()
        patchStd = patchSample.std()
        if patchStd > 0.0: # > 0 to remove blank/uninteresting patches for speed
            # create the patch vector
            patch = np.reshape(patchSample, numPixels)
            patch = patch - np.mean(patch)
            imgPatches[:,patchCount] = patch.copy()
            patchCount += 1
    return imgPatches

#this function displays black and white image patches
def showPatchesBW(prePatches, showPatchNum = 16, display=True):
    patches = prePatches
    totalPatches = patches.shape[1]
    dataDim = patches.shape[0]
    patchWidth = int(np.round(np.sqrt(dataDim)))
    # extract show_patch_num patches
    displayPatch = np.zeros([dataDim, showPatchNum], float)
    # NORMALIZE PATCH LUMINANCE VALUES
    for i in range(0,showPatchNum):
        #patch_i = i * totalPatches // showPatchNum
        patch_i = i
        patch = patches[:,patch_i].copy()
        pmax  = patch.max()
        pmin = patch.min()
        # fix patch range from min to max to 0 to 1
        if pmax > pmin:
            patch = (patch - pmin) / (pmax - pmin)
        displayPatch[:,i] = patch.copy()
    bw = 5    # border width
    pw = patchWidth
    patchesY = int(np.sqrt(showPatchNum))
    patchesX = int(np.ceil(float(showPatchNum) / patchesY))
    patchImg = displayPatch.max() * np.ones([(pw + bw) * patchesX - bw, patchesY * (pw + bw) - bw], float)
    for i in range(0,showPatchNum):
        y_i = i // patchesY
        x_i = i % patchesY
        reshaped = displayPatch[:,i].reshape((pw,pw))
        fullPatch = np.zeros([pw, pw], float)
        fullPatch[0:pw,:] = reshaped[:,:].copy()
        patchImg[x_i*(pw+bw):x_i*(pw+bw)+pw,y_i*(pw+bw):y_i*(pw+bw)+pw] = fullPatch

    if display:
        py.bone()
        py.imshow(patchImg.T, interpolation='nearest')
        py.axis('off')
    return


# this function collects patches from color images
def collectPatchesColor(numPatches, patchWidth, filePath):
    maxTries = numPatches * 50
    imageWidth = 200
    firstPatch = 0  # the first patch number accepted from an image
    firstTry = 0  # the first attempt to take a patch from the image
    patchCount = 0  # number of collected patches
    tryCount = 0  # number of attempted collected patches
    numPixels = 3 * patchWidth * patchWidth
    patchSample = np.zeros([3, patchWidth, patchWidth], float)
    patch = np.zeros([numPixels], float)
    imgPatches = np.zeros([numPixels, numPatches], float)
  # this chooses the image we're sampling from, starting from the first image
    imgCount = 1
    image = PIL.Image.open(filePath + str(imgCount) + '.jpg')
    image = np.asarray(image, 'double').transpose()
    image = image[0:3, :, :]
  # normalizing the image
    image -= image.mean()
    image /= image.std()
  # collect the patches from images in file
    while patchCount < numPatches and tryCount < maxTries:
        tryCount += 1  # number of total patches attempted
        if tryCount - firstTry > maxTries / 2 or patchCount             - firstPatch > numPatches / 2:
            imgCount += 1  # this switches to the next image we're sampling from!
            image = PIL.Image.open(filePath + str(imgCount) + '.jpg')
            image = np.asarray(image, 'double').transpose()
            image = image[0:3, :, :]
            image -= image.mean()
            image /= image.std()
            firstPatch = patchCount
            firstTry = tryCount
        px = np.random.randint(0, imageWidth - patchWidth)
        py = np.random.randint(0, imageWidth - patchWidth)
        patchSample = image[:, px:px + patchWidth, py:py
                            + patchWidth].copy()
        patch_std = patchSample.std()
        if patch_std > 0.0:  # > 0 to remove blank/uninteresting patches for speed
      # create the patch vector
            patch = np.reshape(patchSample, numPixels)
            patch = patch - np.mean(patch)
            imgPatches[:, patchCount] = patch.copy()
            patchCount += 1
    return imgPatches

#this function displays color image patches
def showPatchesColor(prePatches, showPatchNum=16, display=True):
    patches = prePatches
    totalPatches = patches.shape[1]
    dataDim = patches.shape[0]
    patchWidth = int(np.round(np.sqrt(dataDim/3)))
  # extract showPatchNum patches
    displayPatch = np.zeros([dataDim, showPatchNum], float)
    # NORMALIZE PATCH LUMINANCE VALUES
    for i in range(0, showPatchNum):
        #patch_i = i * totalPatches // showPatchNum
        patch_i = i
        patch = patches[:, patch_i].copy()
        pmax = patch.max()
        pmin = patch.min()
    # fix patch range from min to max to 0 to 1
        if pmax > pmin:
            patch = (patch - pmin) / (pmax - pmin)
        displayPatch[:, i] = patch.copy()
    bw = 5  # border width
    pw = patchWidth
    patchesY = int(np.sqrt(showPatchNum))
    patchesX = int(np.ceil(float(showPatchNum) / patchesY))
    patchImg = displayPatch.max() * np.ones([3, (pw + bw) * patchesX
            - bw, patchesY * (pw + bw) - bw], float)
    for i in range(0, showPatchNum):
        y_i = i // patchesY
        x_i = i % patchesY
    # reshape patch sizing
        reshaped = displayPatch[:, i].reshape((3, pw, pw))
        fullPatch = np.zeros([3, pw, pw], float)
        fullPatch[0:3, 0:pw, 0:pw] = reshaped[:, :, :].copy()
        patchImg[:, x_i * (pw + bw):x_i * (pw + bw) + pw, y_i * (pw
                 + bw):y_i * (pw + bw) + pw] = fullPatch
    if display:
    # displays the patches
        py.imshow(patchImg[:, :, :].T, interpolation='nearest')
        py.axis('off')
    return


#this function collects patches from audio clips
def collectPatchesAudio(file):
    spf = wave.open('audio/' + file + '.wav','r')
    # extract raw audio from .wav file
    signal = spf.readframes(-1)
    signal = np.frombuffer(signal, 'Int16')
    fs = spf.getframerate()
    print(fs)
    numSamples = len(signal)
    width = 100
    ds = 3
    numPatches = 100000
    audioPatches = np.zeros((numPatches,width))
    for i in range(numPatches):
        x_start = np.random.randint(0,numSamples-ds*width-2)
        audioPatches[i,:] = signal[x_start:x_start+ds*width-1:ds]
    return(audioPatches)

#this function displays audio patches
def showPatchesAudio(patches):
    width = 100
    cnt = 0
    for cnt in range(20):
        plt.subplot(5, 5, cnt+1)
        frame = pylab.gca()
        frame.axes.get_xaxis().set_ticklabels([])
        frame.axes.get_yaxis().set_ticklabels([])
        plt.plot(range(width),patches[cnt,:])
    plt.show()


def collectPatchesBinocular(numPatches, patchWidth, filePath):
    maxTries = numPatches * 50
    firstPatch = 0 # the first patch number accepted from an image
    firstTry = 0 # the first attempt to take a patch from the image
    patchCount = 0 # number of collected patches
    tryCount = 0 # number of attempted collected patches
    numPixels = patchWidth * patchWidth
    patchSample = np.zeros([patchWidth,patchWidth],'double')
    patchSampleL = np.zeros([numPixels,numPatches],'double')
    patchSampleR = np.zeros([numPixels,numPatches],'double')
    patch = np.zeros([numPixels,1],'double')
    #imgPatches = np.zeros([numPixels,numPatches],'double')
    imgPatchesL = np.zeros([numPixels,numPatches],'double')
    imgPatchesR = np.zeros([numPixels,numPatches],'double')
    # chooses the image that we're sampling from
    imgCount = 1
    image = PIL.Image.open(filePath + str(imgCount) + '.jpg')
    Height = image.height
    Width = image.width
    imageHeight, imageWidth, imageChannels = matplotlib.pyplot.imread(filePath + str(imgCount) + '.jpg').shape
    image = image.convert('L')

    left1 = 0
    top1 = 0
    right1 = 1060 // 2
    bottom1 = 496

    left2 = 1060 // 2
    top2 = 0
    right2 = 1060
    bottom2 = 496

    image_left = image.crop((left1, top1, right1, bottom1))
    image_right = image.crop((left2, top2, right2, bottom2))
    left_width = image_left.width
    right_width = image_right.width

    ## no error yet
    imageL = np.asarray(image_left, 'double').transpose()
    # normalizing the image
    imageL -= imageL.mean()
    imageL /= imageL.std()

    imageR = np.asarray(image_right, 'double').transpose()
    # normalizing the image
    imageR -= imageR.mean()
    imageR /= imageR.std()

    #########################################################

    image = np.asarray(image, 'double').transpose()
    # normalizing the image
    image -= image.mean()
    image /= image.std()

    px = np.random.randint(0, left_width - patchWidth)
    py = np.random.randint(0, Height - patchWidth)

    '''
    lx = np.random.randint(0,left_width // 2)
    ly = np.random.randint(0, imageHeight // 2)
    rx = np.random.randint(0,right_width // 2)
    ry = np.random.randint(0, imageHeight // 2)
    '''

    Patches = []


    while patchCount < numPatches and tryCount < numPatches:
        tryCount += 1
        px = np.random.randint(0, left_width - patchWidth)
        py = np.random.randint(0, Height - patchWidth)

        patchSampleL = imageL[px:px+patchWidth,py:py+patchWidth].copy()
        patchSampleR = imageR[px:px+patchWidth,py:py+patchWidth].copy()
        patchStdL = patchSampleL.std()
        patchStdR = patchSampleR.std()

        # create the patch vector
        patchL = np.reshape(patchSampleL, numPixels)
        patchL = patchL - np.mean(patchL)
        imgPatchesL[:,patchCount] = patchL.copy()

        patchR = np.reshape(patchSampleR, numPixels)
        patchR = patchR - np.mean(patchR)
        imgPatchesR[:,patchCount] = patchR.copy()
        patchCount += 1
    Patches.append(imgPatchesL)
    Patches.append(imgPatchesR)

    return np.array(Patches)


def showPatchesBinocular(prePatchesL, prePatchesR, showPatchNum, display=True):
    patchesL = prePatchesL
    patchesR = prePatchesR
    totalPatches = 500
    dataDim = 256
    patchWidth = int(np.round(np.sqrt(dataDim)))
    # extract show_patch_num patches
    displayPatchL = np.zeros([dataDim, showPatchNum], float) #array of zeros for left
    displayPatchR = np.zeros([dataDim, showPatchNum], float) # array of zeros for right
    # loops through patches to print 16 of them
    for i in range(0,showPatchNum):
        #patch_i = i * totalPatches // showPatchNum
        patch_i = i # both start in same position
        patchL = patchesL[:,patch_i].copy() #copys left patch position
        patchR = patchesR[:,patch_i].copy() #copys right patch position
        pmaxL = patchL.max() #normalize left
        pminL = patchL.min()

        pmaxR = patchR.max() #normalize right
        pminR = patchR.min()
        # fix patch range from min to max to 0 to 1 for left
        if pmaxL > pminL:
            patchL = (patchL - pminL) / (pmaxL - pminL)
        displayPatchL[:,i] = patchL.copy()

        # fix patch range from min to max to 0 to 1 for right
        if pmaxR > pminR:
            patchR = (patchR - pminR) / (pmaxR - pminR)
        displayPatchR[:,i] = patchR.copy()
    bw = 5    # border width
    pw = patchWidth # both have same patch width
    # same X and Y for left and right
    patchesY = int(np.sqrt(showPatchNum))
    patchesX = int(np.ceil(float(showPatchNum) / patchesY))
    patchImgL = displayPatchL.max() * np.ones([(pw + bw) * patchesX - bw, patchesY * (pw + bw) - bw], float)
    patchImgR = displayPatchR.max() * np.ones([(pw + bw) * patchesX - bw, patchesY * (pw + bw) - bw], float)

    # THIS IS WHERE IT NEEDS TO BE FIXED (THE LOOP)
    '''
    the first loop below cycles through the left images and adds them to a subplot
    then the next loop does the right images and adds them to the same subplot

    the issue: not being able to tell if the patches are in the correct order
                (left the right paired together in each location) and not getting
                32 total patches (16 from each side)
    probable solution: changes axis so the subplots dont go on top of each other
    '''

    k = 0
    fig = plt.figure(figsize=(2,16))
    ##############  LOOP #####################################
    for i in range(0,showPatchNum):
        k = k+1
        y_i = i // patchesY
        x_i = i % patchesY
        reshapedL = displayPatchL[:,i].reshape((pw,pw))
        fullPatchL = np.zeros([pw, pw], float)
        fullPatchL[0:pw,:] = reshapedL[:,:].copy()
        patchImgL[x_i*(pw+bw):x_i*(pw+bw)+pw,y_i*(pw+bw):y_i*(pw+bw)+pw] = fullPatchL

        reshapedR = displayPatchR[:,i].reshape((pw,pw))
        fullPatchR = np.zeros([pw, pw], float)
        fullPatchR[0:pw,:] = reshapedR[:,:].copy()
        patchImgR[x_i*(pw+bw):x_i*(pw+bw)+pw,y_i*(pw+bw):y_i*(pw+bw)+pw] = fullPatchR

        patches_array = np.concatenate((fullPatchL, fullPatchR), axis = 1)
        ax = plt.subplot(16,1,k)

        if k == 1:
          ax.title.set_text('Left       Right')
        for i in range(0, showPatchNum):
          py.axis('off')
          ax.imshow(patches_array, interpolation='nearest',cmap=py.get_cmap('gray'))


    return


def showFiltersBinocular(leftComponents, rightComponents, NumFilters):
    patchesL = leftComponents
    patchesR = rightComponents
    totalPatches = 500
    dataDim = 256
    patchWidth = int(np.round(np.sqrt(dataDim)))
    # extract show_patch_num patches
    displayPatchL = np.zeros([dataDim, NumFilters], float) #array of zeros for left
    displayPatchR = np.zeros([dataDim, NumFilters], float) # array of zeros for right
    # loops through patches to print 16 of them
    for i in range(0,NumFilters):
        #patch_i = i * totalPatches // showPatchNum
        patch_i = i # both start in same position
        patchL = patchesL[:,patch_i].copy() #copys left patch position
        patchR = patchesR[:,patch_i].copy() #copys right patch position
        pmaxL = patchL.max() #normalize left
        pminL = patchL.min()

        pmaxR = patchR.max() #normalize right
        pminR = patchR.min()
        # fix patch range from min to max to 0 to 1 for left
        if pmaxL > pminL:
            patchL = (patchL - pminL) / (pmaxL - pminL)
        displayPatchL[:,i] = patchL.copy()

        # fix patch range from min to max to 0 to 1 for right
        if pmaxR > pminR:
            patchR = (patchR - pminR) / (pmaxR - pminR)
        displayPatchR[:,i] = patchR.copy()
    bw = 5    # border width
    pw = patchWidth # both have same patch width
    # same X and Y for left and right
    patchesY = int(np.sqrt(NumFilters))
    patchesX = int(np.ceil(float(NumFilters) / patchesY))
    patchImgL = displayPatchL.max() * np.ones([(pw + bw) * patchesX - bw, patchesY * (pw + bw) - bw], float)
    patchImgR = displayPatchR.max() * np.ones([(pw + bw) * patchesX - bw, patchesY * (pw + bw) - bw], float)


    k = 0
    fig = plt.figure(figsize=(2,16))
    ##############  LOOP #####################################
    for i in range(0,NumFilters):
        k = k+1
        y_i = i // patchesY
        x_i = i % patchesY
        reshapedL = displayPatchL[:,i].reshape((pw,pw))
        fullPatchL = np.zeros([pw, pw], float)
        fullPatchL[0:pw,:] = reshapedL[:,:].copy()
        patchImgL[x_i*(pw+bw):x_i*(pw+bw)+pw,y_i*(pw+bw):y_i*(pw+bw)+pw] = fullPatchL

        reshapedR = displayPatchR[:,i].reshape((pw,pw))
        fullPatchR = np.zeros([pw, pw], float)
        fullPatchR[0:pw,:] = reshapedR[:,:].copy()
        patchImgR[x_i*(pw+bw):x_i*(pw+bw)+pw,y_i*(pw+bw):y_i*(pw+bw)+pw] = fullPatchR

        patches_array = np.concatenate((fullPatchL, fullPatchR), axis = 1)
        ax = plt.subplot(16,1,k)

        if k == 1:
          ax.title.set_text('Left       Right')
        for i in range(0, NumFilters):
          py.axis('off')
          ax.imshow(patches_array, interpolation='nearest',cmap=py.get_cmap('gray'))


    return
