import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import cv2

'test'

class ConvolutionLayer(object):
    def __init__(self, input_data, filter_size):
        self.filters = np.random.randn(1,filter_size,filter_size,input_data.shape[3]) 
        self.bias = 1
        self.filter_gradient = np.zeros([filter_size, filter_size])
        self.convolution_gradient = np.zeros(input_data.shape[1:])
    #
    def convolve(self, input_data, filter, stride=1):
        filter = np.rot90(np.rot90(filter))
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
    def gradients(self, input_data, backprop_gradients):
        self.filter_gradient = self.convolve(input_data, backprop_gradients)
        filter = self.filters
        filter[1:4] = np.pad(filter[1:],1,'constant')
        self.convolution_gradient = self.convolve(np.pad)
#
class FCLayer(object):
    def __init__(self, output_size, type = 'relu'):
        self.Weights = np.random.randn()
        self.bias = 0
#
def test_convolve():
    image1 = cv2.imread('C:/Users/cooloryeti/Desktop/house.jpg')
    image1 = np.asarray(image1, dtype='float')
    image1 = np.array([image1])
    layer = ConvolutionLayer(image1, 3)


    filter = np.array([[[[-1,-1,-1],[0,0,0],[1,1,1]],
                        [[-4,-4,-4],[0,0,0],[4,4,4]],
                        [[-1,-1,-1],[0,0,0],[1,1,1]],]])

    image2 = layer.convolve(image1,filter) 

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
a = np.array([[[[1,1,1],
            [1,1,1],
            [1,1,1]]]])
print a.shape
a = np.pad(a, 1, 'constant')
print a.shape