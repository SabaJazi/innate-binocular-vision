from numpy import *
import scipy
import scipy.signal
import scipy.optimize
# import Image
from PIL import Image
import pickle
import os
import FilterTools
import pylab
import matplotlib.pyplot

# create a class to process V1 images
# created by Mark V. Albert, July 2008

img1_x = 0
img2_x = 0

def read_imgs(imgL="data/magiceye3.png", imgR=None, 
  state={}):
  """ reads in either a filename or an array as a stereogram
  returns a state variable for V1Tools """
  
  # if it's an autostereogram, make the left and right eye view the same
  if imgR == None:
    imgR = imgL
  
  if isinstance(imgL,str): #img_in is an image file
    im_L = Image.open(imgL)
    im_R = Image.open(imgR)
    imgL_normed = asarray(im_L).transpose() / 255.0
    imgR_normed = asarray(im_R).transpose() / 255.0
    
  else: # image is an array
    im_L = imgL.copy()
    im_R = imgR.copy()
    imgL_normed = (im_L - im_L.min()) / (im_L.max() - im_L.min())
    imgR_normed = (im_R - im_R.min()) / (im_R.max() - im_R.min())    

  state['img_mat'] = array([imgL_normed, imgR_normed])
  return state

def click1(event):
       img1_x = event.xdata
       print ('x of figure 1 is', event.xdata)

def click2(event):
       img2_x = event.xdata
       print( 'x of figure 2 is', event.xdata)
   
def binoc_test(img_in1 = "binoc/001_L.png", img_in2 = "binoc/001_R.png"):
  img1 = Image.open(img_in1)
  img2 = Image.open(img_in2)
  img1 = img1.transpose(Image.FLIP_TOP_BOTTOM)
  img2 = img2.transpose(Image.FLIP_TOP_BOTTOM)
  ##ivals = array(img1.getdata(),shape=(img1.size[1],img1.size[0]))
  pylab.figure(1)
  pylab.imshow(img1)
  matplotlib.pyplot.connect('button_press_event',click1)
  pylab.figure(2)
  pylab.imshow(img2)
  matplotlib.pyplot.connect('button_press_event',click2)

def adapt_shiftLR(img_mat, debug=False):
  """ this function finds the appropriate shiftLR for the autostereogram """
  slice_size = 10
  corrslide = [corrcoef(img_mat[0:slice_size,:].flatten(),
	img_mat[x:(x+slice_size),:].flatten())[0,1]
	for x in range(slice_size,min(600,img_mat.shape[0]-slice_size))]
  if debug:
    import pylab
    pylab.plot(corrslide)
  shiftLR = argmax(corrslide)+slice_size
  print ("New shift for autostereogram is %d" % shiftLR)
  return shiftLR
  
def convolve_filters(state, filters, bases, down):
  """ this function sets the filter activity for the left and right """
  state['downsample'] = down
  width = state['img_mat'].shape[1]
  height = state['img_mat'].shape[2]
  
  # downsample the image
  down_img_mat = [[],[]]
  for i in [0,1]:
    im = Image.fromarray(array(state['img_mat'][i].transpose()*255,uint8),'L')  
    im = im.resize((width//down,height//down),Image.ANTIALIAS)
    down_img_mat[i] = asarray(im).transpose()/255.0
  state['down_img_mat'] = array(down_img_mat)  
    # down_img_mat is a 2 x width//down x height//down array

  state['v1_filters'] = FilterTools.convert_patches(filters, 4)
  state['v1_bases'] = FilterTools.convert_patches(bases, 4)
  
  num_of_filters = state['v1_filters'].shape[0]
  filter_width = state['v1_filters'].shape[2]
  state['filtered_img'] = zeros([num_of_filters,2, 
	shape(state['down_img_mat'])[1] - filter_width + 1,
	shape(state['down_img_mat'])[2] - filter_width + 1], 'float') 
  num_of_filters = state['v1_filters'].shape[0]
  for f in range(0,num_of_filters):
    print(float(f) / num_of_filters)
    for lr in range (0,2):
      state['filtered_img'][f,lr,:,:] =  \
      scipy.signal.convolve2d(state['v1_filters'][f,lr,:,:],
      state['down_img_mat'][lr], 'valid')

def save_state(state, filename='data/V1Tools_state.dat'):
  outfile = open(filename,'wb')
  pickle.dump(state, outfile, -1)
  outfile.close()

def load_state(filename='data/V1Tools_state.dat'):
  infile = open(filename,'rb')
  state = pickle.load(infile)
  infile.close()
  return state

def get_filter_prod_activity(state, shiftLR):
   """ returns a matrix of combined L and R filter activity over the whole
	  image - num of filters x xdim x ydim compensating for shiftLR """     
   pix_shift = int(round(shiftLR / state['downsample'])) #dividing shift by downsample
   min_L_x = max(0,-pix_shift) #starting point for left filtered
   min_R_x = max(0,pix_shift) #starting point for right filtered
   # state.filtered_img.shape[2] is the filtered image width after convolution
   size_x = state['filtered_img'].shape[2] - abs(pix_shift)
   L_filtered_shifted = state['filtered_img'][:,0,min_L_x:min_L_x+size_x,:]
   R_filtered_shifted = state['filtered_img'][:,1,min_R_x:min_R_x+size_x,:]
   filter_prod_activity = L_filtered_shifted * R_filtered_shifted
   return filter_prod_activity

def get_depth_mat(state, shiftLR=140, max_activity_method=False, adaptShiftLR=False):
  """ this function creates a depth judgement for an image from the 
	  V1 shift matrix and filter product activity """
	  
  if adaptShiftLR:
	  shiftLR = adapt_shiftLR(state['img_mat'][0])
	
  filter_prod_activity = get_filter_prod_activity(state, shiftLR)
  disp_mat = FilterTools.get_disparity_mat(state['v1_bases'])
  
  width = filter_prod_activity.shape[1]
  height = filter_prod_activity.shape[2]
  max_shift = (disp_mat.shape[0]-1) / 2
  depth_mat = zeros([width, height], 'float')

  for x in range(0,width):
    for y in range(0,height):
      if not max_activity_method:
        depth_vec = dot(disp_mat, filter_prod_activity[:,x,y])
        max_index = argmax(depth_vec)
      else:  
        max_index = argmax(disp_mat[:,argmax(filter_prod_activity[:,x,y])])
    depth = max_index - max_shift
    depth_mat[x,y] = depth
  
  return depth_mat
  