__author__='Pabi'

import numpy as np
import pandas as pd

class Naive_Conv_NeuralNet_Layer(object):
    """
    A numpy based, naive implementation of a vanilla convolutional neural net layer.
    Multiple Conv layers + pooling(optional) + some final output transformations define a full Conv Net
    Details of the mathematics can be found at the from  textbook 'Deep Learning' by Goodman, Bengio et al.
    Implementation based also on stanford convoltional networks course work:
    http://cs231n.github.io/convolutional-networks/#conv
    """

    def __init__(self,input_feature_map_dim,no_filters,filter_map_dim=3,stride_len=1,zero_padding=1,rgb=True):
        """
        :param input_feature_map_dim:
        :param no_filters:
        :param filter_map_dim:
        :param stride_len:
        :param zero_padding:
        :return:
        """
        self.width_X,self.height_Y,self.num_channels_D=input_feature_map_dim
        self.no_filters=no_filters
        self.filter_map_dim=filter_map_dim
        self.stride_len=stride_len
        self.zero_padding=zero_padding
        self.bias=self.filter_matrix_initialization(method='Gaussian')
        self.filter_map = {}
        if self.rgb == True:
            self.n_channels=3
        else:
            self.n_channels=1
        for map_indx in range(1,self.filter_map_dim):
            self.filter_map['filter_matrix_{}'.format(str(map_indx))]=self.filter_matrix_initialization(method='Gaussian')

        # Initialize weights to be applied to input_feature_map. Sample from N(0,1) as intialization weights

        # Output volume sizes
        #TODO: write a function to check if out dim are int types. If not adjust zero padding
        self.Output_Width=(self.width_X-self.filter_map_dim+2*self.zero_padding)/(self.stride_len+1)
        self.Output_Height=(self.height_Y-self.filter_map_dim+2*self.zero_padding)/(self.stride_len+1)


    def Naive_Conv_forwardpass(self,input_feature_map):
        """
        Implements the forward pass convolution operation on a input feature map
        Simplest (most inefficient method.) Move filter by stride length across image and perform conv operation
        :param input_feature_map: input_feature map from previous layer
        :return: ReLu transformation non linearity of features after conv op.
        """


    def im2col(self,X,filter_dim):
        """
        :param X: filter_dim X filter_dim area of current image to be converted to 1 X filter_dim^2 vector
        :param filter_dim:
        :return: vector of dimension 1 by filter_dim^2
        """
        flat_vect = np.reshape(X,-1)
        return flat_vect

    def convolution_op(self,image_col,filter_col):
        conv_result=np.sum(np.dot(image_col,filter_col))
        return conv_result
    def filter_matrix_initialization(self,method='Gaussian'):
        return np.random.normal(loc=0,scale=1,size=[self.filter_map_dim,self.filter_map_dim])

class ReLu(object):

    def __inti__(self):
        return ""
    def forward_ReLU(self,X):
        
