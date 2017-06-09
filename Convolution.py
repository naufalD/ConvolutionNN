import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import cv2

'test'

class ConvolutionNet(object):
    def __init__(self):
        pass

def convolve(input_data, filter,stride=1):
    shape_x = input_data.shape[1]-(filter.shape[1]-1)
    shape_y = input_data.shape[2]-(filter.shape[2]-1)
    output = np.zeros((input_data.shape[0],shape_x,shape_y))
    tic = time.time()
    for y in range(shape_y):
        for x in range(shape_x):
            output[input_data.shape[0]-1,x,y] = np.sum(input_data[:,x:x+filter.shape[1],y:y+filter.shape[1]]*filter)
    toc = time.time()
    print 'Took : ', toc-tic, 's\n'
    return output
#
def test_convolve():
    image1 = cv2.imread('C:/Users/cooloryeti/Desktop/house.jpg')
    image1 = np.asarray(image1, dtype='float')
    image1 = np.array([image1])
    print image1.shape


    filter = np.array([[[[-1,-1,-1],[0,0,0],[1,1,1]],
                        [[-4,-4,-4],[0,0,0],[4,4,4]],
                        [[-1,-1,-1],[0,0,0],[1,1,1]],]])

    image2 = convolve(image1,filter) 

    image1 = image1[0,:,:,:]
    image2 = image2[0,:,:]
    image3 = np.maximum(image2,0)
    image3 = np.minimum(image3,255)
    image4 = np.minimum(image3,255)

    plt.subplot(4,2,1)
    plt.imshow(cv2.cvtColor(image1.astype('uint8'), cv2.COLOR_BGR2RGB))
    plt.subplot(4,2,2)
    plt.imshow(image2.astype('uint8'), cmap='Greys_r')
    plt.subplot(4,2,3)
    plt.imshow(image3.astype('uint8'), cmap='Greys_r')
    plt.subplot(4,2,4)
    plt.imshow(image4.astype('uint8'), cmap='Greys_r')    

    plt.show()

#
test_convolve()