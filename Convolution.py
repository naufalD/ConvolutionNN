import numpy as np
import matplotlib.pyplot as plt

class ConvolutionNet(object):
    def __init__(self):
        pass

def convolve(input_data, filter,stride=1):
    shape_x = input_data.shape[1]-(filter.shape[1]-1)
    shape_y = input_data.shape[2]-(filter.shape[2]-1)
    print input_data.shape[1]
    print filter.shape[1]-1
    output = np.zeros((input_data.shape[0],shape_x,shape_y))
    for y in range(shape_y):
        for x in range(shape_x):
            output[input_data.shape[0]-1,x,y] = np.sum(input_data[:,x:x+filter.shape[1],y:y+filter.shape[1]]*filter)
    return output
                

input_data = np.array([[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]])
filters = np.array([[[1,1,1],[1,1,1],[1,1,1]]])
print convolve(input_data,filters)