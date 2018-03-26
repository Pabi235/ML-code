__author__='Pabi'

import numpy as np
import pandas as pd

class Data(object):
    """
    This class is used for all data storage purposes
    Data is stored as 3D volumes. This includes wieghts and input data
    """
    def __init__(self,width,height,depth,weight_init='undefined'):
        self.width=width
        self.height=height
        self.depth=depth
        n_weight_vals=self.height * self.depth * self.depth
        self.data_mtx= np.reshape(np.zeros(n_weight_vals),newshape=(self.depth,self.height,self.width))
        self.delta_data_mtx= np.reshape(np.zeros(n_weight_vals),newshape=(self.depth,self.height,self.width))
        self.vectorized_rand_norm=np.vectorize(self.generate_normal_rand,otypes=[np.float])

        # weight initialization
        # if weight initialization not set then use Xavier method for initializing weights
        'TODO: provide link'
        if weight_init != 'undefined':

            weight_std = np.sqrt(n_weight_vals)
            self.data_mtx = self.vectorized_rand_norm(data_mtx,weight_std)

        else:
            self.data_mtx=weight_init

    def get_data(self,width_indx,heigh_indx,depth_indx):
        pass
        #weight_patch=
    def set_data(self,i,j,depth,val):
        self.data_mtx[depth,i,j]=val
        pass

    def get_gradient(self):
        pass
    def set_gradient(self):
        pass

    def generate_normal_rand(self,matx_elmt):
        n_weight_vals=self.height * self.depth * self.depth
        return matx_elmt + np.random.normal(loc=0,scale=np.sqrt(n_weight_vals),size=1)


class Naive_Conv_NeuralNet_Layer(object):
    """
    A numpy based, naive implementation of a vanilla convolutional neural net layer.
    Multiple Conv layers + pooling(optional) + some final output transformations define a full Conv Net
    Details of the mathematics can be found at the from  textbook 'Deep Learning' by Goodman, Bengio et al.
    Implementation based also on stanford convolutional networks course work:
    http://cs231n.github.io/convolutional-networks/#conv
    """

    def __init__(self,input_volume,input_feature_map_dim,no_filters,filter_map_dim=3,stride_len=1,zero_padding=1,k=3):
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
        self.input_vol = input_volume.copy()
        self.k=k
        self.filter_size =filter_map_dim
        self.stride_len=stride_len
        self.zero_padding=zero_padding
        self.filter_vol = Data(width=self.filter_size,height=self.filter_size,depth=self.input_vol.depth)
        self.filter_map = [Data(width=self.filter_size,height=self.filter_size,depth=self.input_vol.depth) for i in range(self.n_filters)]
        self.bias_vol= Data(width=self.filter_size,height=self.filter_size,depth=1)

        # Initialize weights to be applied to input_feature_map. Sample from N(0,1) as intialization weights

        # Output volume sizes
        #TODO: write a function to check if output dim are int types. If not adjust with appropriate zero padding
        self.Output_Width=(self.input_vol.width-self.filter_size+2*self.zero_padding)/(self.stride_len+1)
        self.Output_Height=(self.input_vol.height-self.filter_size+2*self.zero_padding)/(self.stride_len+1)
        self.output_Tensor = Data(width=self.Output_Width,height=self.Output_Height,depth=self.n_filters)

    def Naive_Conv_forwardpass(self):
        """
        Forward pass of conv net implementing output map generating function.
        Returns an tensor of self.Output_Width X self.Output_Height X self.n_filters. Which is happily sent off to the
        next poor bugger layer of the CNN architecture
        Not happy too(livid actually)  with the efficiency and endless loops but c'est la vie. This is for education purposes.
        :param X: Image/feature map from previous layer/(just the picture I guess)
        :return: tensor of vol : self.Output_Width X self.Output_Height X self.n_filters
        """

        for filter_k in range(0,self.n_filters):
            filter_col = self.im2col( self.filter_map[filter_k].data_mtx)
            for wdth_indx in range(0,self.Output_Width):
                for hgt_indx in range(0,self.Output_Height):

                    img_area=self.input_vol.data_mtx[:,wdth_indx:(wdth_indx+self.stride_len),hgt_indx:(hgt_indx+self.stride_len)]
                    trn_img_col=self.im2col(img_area)
                    self.output_Tensor.data_mtx[filter_k,hgt_indx,wdth_indx] = self.convolution_op(trn_img_col,filter_col)+self.bias_vol
        return self.output_Tensor

    def Naive_Conv_backwardpass(self,X):
        """
        :param X:
        :return:
        """
        #  this fucking backward pass ...sigh
        # need to make two backward pass gradient calculations
        # gradient w.r.t filter weights which will be used for weight updates.
        # gradient w.r.t current image representation/ current layer which will be used as gradient flow to lower layers

        for filter_j in range(0,len(self.filter_map)):
            filter_vol=self.filter_map[filter_j]
            



    def im2col(self,X):
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



class ReLu(object):

    def __inti__(self):
        return ""
    def forward_ReLU(self,X):

class Loss_layer(object):


def generate_normal_rand(matx_input,y):
    """
    :param matx_input: a matrix of zeros
    :param y: variance of the neuron. Mean always assumed to be 0
    :return:
    """
    return matx_input + np.random.normal(loc=0,scale=np.sqrt(y),size=1)

generate_normal_rand(0,4)

data_mtx= np.reshape(np.zeros(300),newshape=(3,10,10))
def vectorized_rand_norms():
    np.vectorize(generate_normal_rand,otypes=[np.float])

vectorized_rand_norm=np.vectorize(generate_normal_rand,otypes=[np.float])
p=vectorized_rand_norm(data_mtx,4)

#p=vectorized_rand_norm(data_mtx)
