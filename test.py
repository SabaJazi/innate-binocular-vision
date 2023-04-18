
import ica_helper_methods
import matplotlib.pyplot as plt
import sklearn.decomposition
from sklearn import decomposition
import numpy as np
import pylab as py
# ---------------------------------------------------

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
# ---------------------------------------------------

patchesBinocular = ica_helper_methods.collectPatchesBinocular(50000, 16, 'C:/Users/19404/innate-binocular-vision/')

left = patchesBinocular[0]
right = patchesBinocular[1]

showPatchesBinocular(left, right, 16)
# print(left)