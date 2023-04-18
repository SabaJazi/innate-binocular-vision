from numpy import *
import pylab

# create a percolation network
# created by Mark V. Albert, May 2008

def distance(x0, y0, x1, y1):
  return sqrt( pow(x0-x1,2) + pow(y0-y1,2) )

class LGN: 
  """
  this class defines a model which generates binocular spontaneous activity
  """
  
  def __init__(self, width = 128, p = 0.5, r = 1.0, t = 1, trans = 0.0,
    make_wave = True, num_layers=2):
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
    self.recruitable = random.rand(self.num_layers, w, w) < self.p
    self.tot_recruitable = len(where(self.recruitable)[0])
    self.tot_recruitable_active = 0
    self.tot_active = 0
    self.active = zeros([self.num_layers,w,w],bool)
    self.active_neighbors = zeros([self.num_layers,w,w],int)
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
                if random.rand() < self.trans: # transfer the activity
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
      l = random.randint(0,self.num_layers)
      x = random.randint(0,self.width)
      y = random.randint(0,self.width)
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
      print ("monocular models cannot have correlations between eye layers")
      return 0
    w = self.width
    active01 = zeros([2,w,w],int)
    active01[where(self.active)] = 1
    
    mean0 = active01[0,:,:].mean()
    mean1 = active01[1,:,:].mean()
    std0 = active01[0,:,:].std()
    std1 = active01[1,:,:].std()
    cov = ((active01[0,:,:] - mean0) * (active01[1,:,:] - mean1)).mean()
    return cov / (std0 * std1)
    
    
  def make_img_mat(self, show_img=True):
    """ return a matrix of 1's and 0's showing the activity in both layers """
    border_width = 10 if self.num_layers > 1 else 0
    w = self.width
    img = zeros([w, self.num_layers*w+border_width], float)
    for l in range(self.num_layers):
      for x in range(0,w):
        for y in range(0,w):
          if self.active[l,x,y]:
            img[y,l*(w+border_width) + x] = 1
    if border_width > 0:  # if it's not monocular
      # making the border between the two layers
      img[:,w:(w+border_width)] = 0.5
    
    if (show_img):
      pylab.bone()
      pylab.imshow(img)
    
    return img