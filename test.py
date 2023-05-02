
# import ica_helper_methods
import matplotlib.pyplot as plt
from PIL import Image
import sklearn.decomposition
from sklearn import decomposition
import numpy as np
import pylab as py

import matplotlib
# =====================================================
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
    image = Image.open(filePath  + '.png')
    Height = image.height
    Width = image.width
    imageHeight, imageWidth = matplotlib.pyplot.imread(filePath  + '.png').shape
    # print( imageHeight, imageWidth)
    image = image.convert('L')

    left1 = 0
    top1 = 0
    right1 = 270 // 2
    bottom1 = 200

    left2 = 270 // 2
    top2 = 0
    right2 = 270
    bottom2 = 200

    image_left = image.crop((left1, top1, right1, bottom1))
    image_right = image.crop((left2, top2, right2, bottom2))
    left_width = image_left.width
    right_width = image_right.width
    
    ## no error yet
    imageL = np.asarray(image_left, 'double').transpose()
   
    # normalizing the image
    imageL=imageL - (np.mean(imageL))
    imageL =imageL/(np.std(imageL) )

    imageR = np.asarray(image_right, 'double').transpose()
    # normalizing the image
    imageR=imageR - (np.mean(imageR))
    imageR =imageR/(np.std(imageR) )

    #########################################################

    image = np.asarray(image, 'double').transpose()
    # normalizing the image
    image=image - (np.mean(image))
    image =image/(np.std(image) )

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


# ---------------------------------------------------

def showPatchesBinocular(prePatchesL, prePatchesR, showPatchNum, display=True):
    patchesL = prePatchesL
    patchesR = prePatchesR
    totalPatches = 500
    dataDim = 256
    # dataDim = 64
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
    fig = plt.figure(figsize=(8,8))
    ##############  LOOP #####################################
    for i in range(0,int(showPatchNum/4)):
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
        # I concatenate some 1s between concatenation of left and right patches for better visuals
        patches_array = np.concatenate((fullPatchL,np.ones([pw, pw], float), fullPatchR), axis = 1)
        # number of pair that are shown
        ax = plt.subplot(4,1,k)

        if k == 1:
          ax.title.set_text('Left                 Right')
        for i in range(0, showPatchNum):
          py.axis('off')
          ax.figure.set_size_inches(4, 4)
          ax.imshow(patches_array, interpolation='nearest',cmap=py.get_cmap('gray'))
    plt.show()
        

    return
# ---------------------------------------------------

patchesBinocular = collectPatchesBinocular(50000, 16, r'C:\vscode\innate-binocular-vision\innate-binocular-vision\data\data')
left = patchesBinocular[0]
right = patchesBinocular[1]

showPatchesBinocular(left, right, 16)
# print(left)