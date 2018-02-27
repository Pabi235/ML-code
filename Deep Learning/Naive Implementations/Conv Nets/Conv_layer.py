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

    def __init__(self,input_feature_map_dim,no_filters,filter_map_dim=3,stride_len=1,zero_padding=1,k=3):
        """
        :param input_feature_map_dim:
        :param no_filters:
        :param filter_map_dim:
        :param stride_len:
        :param zero_padding:
        :return:
        """
        self.width_X,self.height_Y,self.num_channels_D=input_feature_map_dim
        self.n_filters=no_filters
        self.k=k
        self.filter_map_dim=filter_map_dim
        self.stride_len=stride_len
        self.zero_padding=zero_padding
        self.filter_map = {}
        self.bias={}
        self.
        for map_indx in range(0,self.filter_map_dim):
            self.filter_map['filter_tensor_{}'.format(str(map_indx))]=self.filter_matrix_initialization(method='Gaussian')
            self.bias['bias_{}'.format(str(map_indx))]=np.random.normal(loc=0,scale=1,size=1)
        # Initialize weights to be applied to input_feature_map. Sample from N(0,1) as intialization weights

        # Output volume sizes
        #TODO: write a function to check if out dim are int types. If not adjust zero padding
        self.Output_Width=(self.width_X-self.filter_map_dim+2*self.zero_padding)/(self.stride_len+1)
        self.Output_Height=(self.height_Y-self.filter_map_dim+2*self.zero_padding)/(self.stride_len+1)


    def Naive_Conv_forwardpass(self,X):
        """
        Forward pass of conv net implementing output map generating function.
        Returns an tensor of self.Output_Width X self.Output_Height X self.n_filters. Which is happily sent off to the
        next poor bugger layer of the CNN architecture
        Not happy too(livid actually)  with the efficiency and endless loops but c'est la vie. This is for education purposes.
        :param X: Image/feature map from previous layer/(just the picture I guess)
        :return: tensor of vol : self.Output_Width X self.Output_Height X self.n_filters
        """
        output_tensor = np.zeros((self.n_filters,self.Output_Height,self.Output_Width))
        for filter_k in range(0,self.n_filters):
            krnl_col=self.im2col(self.filter_map['filter_tensor_{}'.format(str(filter_k))])
            for wdth_idnx in range(0,self.Output_Width):
                for hgt_indx in range(0,self.Output_Height):

                    img_area=X[:,wdth_idnx:(wdth_idnx+self.stride_len),hgt_indx:(hgt_indx+self.stride_len)]
                    trn_img_col=self.im2col(img_area)
                    output_tensor[filter_k,hgt_indx,wdth_idnx] = self.convolution_op(trn_img_col,krnl_col)+self.bias['bias_{}'.format(str(filter_k))]
        return output_tensor



    def im2col(self,X):
        """
        :param X: filter_dim X filter_dim area of current image to be converted to 1 X filter_dim^2 vector
        :param filter_dim:
        :return: vector of dimension 1 by filter_dim^2
        """
        if X.shape == (self.filter_map_dim,self.filter_map_dim):
            flat_vect = np.reshape(X,-1)
            return flat_vect



    def convolution_op(self,image_col,filter_col):
        conv_result=np.sum(np.dot(image_col,filter_col))
        return conv_result
    def filter_matrix_initialization(self,method='Gaussian'):
        return np.random.normal(loc=0,scale=1,size=[self.num_channels_D,self.filter_map_dim,self.filter_map_dim])


class ReLu(object):

    def __inti__(self):
        return ""
    def forward_ReLU(self,X):
        
