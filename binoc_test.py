from numpy import *
import scipy
import pickle
import pylab
import V1Tools
import FilterTools
from matplotlib.widgets import Slider

##Run on left and right images: instantiate class with: import binoc
##input filename ### in ###_L.png, ###_R.png
##shows slider to change shiftLR

class binoc:
  def __init__(self,inputimage):

    ##V1Tools.binoc_test("binoc/" + inputimage + "_L.png","binoc/" + inputimage + "_R.png")
    state = V1Tools.read_imgs("binoc/" + inputimage + "_L.png","binoc/" + inputimage + "_R.png")

    width=10
    ##state['shiftLR'] = V1Tools.img2_x - V1Tools.img1_x

    p = FilterTools.pick_gabor_params(100,width)
    filts = FilterTools.gabor_filter_generator(p,width)
    V1Tools.convolve_filters(state,filts,filts,down=3)
    self.state = state

  ##redraw the figure with the L-R separation shiftLR
  def choose_sep(self,shiftLR):
    state = self.state	
    output_map = V1Tools.get_depth_mat(state, shiftLR)
    return output_map.transpose()
    ##pylab.imshow(output_map.transpose())
  """
  def test_binoc2(inputimage,shiftLR):
    ##old method for testing binocular image
    ##V1Tools.binoc_test("binoc/" + inputimage + "_L.png","binoc/" + inputimage + "_R.png")
    state = V1Tools.read_imgs("binoc/" + inputimage + "_L.png","binoc/" + inputimage + "_R.png")
    state['shiftLR'] = shiftLR

    width=10
    ##state['shiftLR'] = V1Tools.img2_x - V1Tools.img1_x

    p = FilterTools.pick_gabor_params(100,width)
    filts = FilterTools.gabor_filter_generator(p,width)
    V1Tools.convolve_filters(state,filts,filts,down=5)
  
    output_map = V1Tools.get_depth_mat(state)
    pylab.figure(2)
    pylab.bone()
    pylab.imshow(output_map.transpose())
    """

##prompts for image number, of image pair ###_L.png,###_R.png
filename = input("Filename: ")
convolved = binoc(filename)
parametermin, parametermax = 0,100
pylab.figure(4)
pylab.subplots_adjust(bottom=0.15)
axparameter  = pylab.axes([0.125, 0.10, 0.775, 0.03] )
slider_parameter = Slider(axparameter, 'ShiftLR', parametermin, parametermax)  
pylab.subplot(111)
pylab.bone()
im = convolved.choose_sep(10)
pylab.imshow(im)

##update the slider bar
def update(val):
	global parameter
	global im
	parameter = slider_parameter.val                  		
	im = convolved.choose_sep(parameter)
	pylab.imshow(im)
	
slider_parameter.on_changed(update)

pylab.show()










