'''Author: Tyler Bolles
   Date: 5/29/2020
   Purpose: This program is a demonstration of a method which tranforms picture
            data into tabular data representing the height of water in an 
            experimental wave tank. An animation was added for demonstration
            purposes, showing the result of the algorithm against (nearly) raw 
            pictures. As written, the accompanying data folder 'June26j' must be
            in the same directory as this file in order to run. '''
from scipy import ndimage as sciImg
import numpy as np
import matplotlib.pyplot as plt
import imageio
import pandas as pd

def get_image(name, blur = True, thresh = None):
    ''' reads in image and cleans it a bit '''
    img = np.linalg.norm(imageio.imread(name), axis = 2)
    if thresh: img = img * (img > thresh)
    if blur: img = sciImg.filters.gaussian_filter(img, (1,2)) 
    return img
	
def get_height_from_img(img, box_size):
    ''' extract y-coordinates of surface for each column of raw image by 
        converting pixel brightness along the column into a probability of
        the surface being there and then taking expectation '''
    yInds = np.arange(len(img[:,0]))
    M = img / (img.sum(axis = 0)) # normalize into a probability density function
    yPix = M.T.dot(yInds) # take expectation w.r.t. pdf for each column
    yAvg = [np.nanmedian([yPix[i * box_size:(i + 1) * box_size]])
            for i in range(int(num_x_pixels / box_size))] # moving median (average)
    return np.asarray(yAvg)

def get_file_name(i):
	# makes file name from picture number. example: picture_1234.png
    if i < 10:
        suffix = '_000'+str(i)
    elif 100 > i >= 10:
        suffix = '_00'+str(i)
    elif 1000 > i >= 100:
        suffix = '_0'+str(i)
    else:
        suffix = '_'+str(i)
    name = 'June26j' + '/' + 'June26j' + suffix + '.png'
    return name

num_x_pixels = 1920
box_size = 5 # make a divisor of length of image. we will ultimately return 
             # a moving average of the raw data with a box size of this many
pixels_to_cm_ratio = 3.125 / 99.0 
num_images = 100
start_img = 1
t  = np.linspace(0, num_images/60., num_images) # time in seconds
x = np.arange(0, num_x_pixels, box_size) # columns of the raw image which will be 
                                         # returned as physical data after the 
                                         # moving average. in units of pixels 

images = [] # store images to speed up the demo animation. 

# the output array of water heights 
height = np.zeros((int(num_x_pixels/box_size), num_images)) 

for i in range(num_images):
    name = get_file_name(i + start_img)
    img = get_image(name, True, 180) 
    height[: , i ] = get_height_from_img(img, box_size) 
    images += [img[:, x]] # save the reduced image

# animation demo
ax  = plt.subplot(1,1,1)
for i in range(0,100, 6):
    ax.cla()
    ax.imshow(images[i], cmap='gray', aspect='auto') # show raw image 
                                                     # this is very time consuming
    ax.plot(height[:, i], color = 'r', lw = 2) # result superimposed in image
    plt.pause(0.001)

height = -height # convert to physical units
mean_height = np.mean(height, axis = 1)
for ind in np.arange(int(num_x_pixels/box_size)):
    height[ind, :] -= mean_height[ind] # water heights have mean zero by physical law. 
                                   # this operation controls for slight angles in camera.
height *= pixels_to_cm_ratio # convert to centimeters
height_df = pd.DataFrame(height.T, index = t, columns = x*pixels_to_cm_ratio)
height_df.to_csv('SurfaceHeightDemo.csv', float_format = '%.4e')



