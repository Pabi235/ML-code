__author__='Pabi'

import numpy as np
import pandas as pd


class Data(object):
    """
    This class is used for all data storage purposes
    Data is stored as 3D volumes. This includes wieghts and input data
    """

    def __init__(self, width, height, depth, weight_init='undefined'):
        self.width = width
        self.height = height
        self.depth = depth
        n_weight_vals = self.height * self.width * self.depth
        self.data_mtx = np.reshape(np.zeros(n_weight_vals), newshape=(self.depth, self.height, self.width))
        self.delta_data_mtx = np.reshape(np.zeros(n_weight_vals), newshape=(self.depth, self.height, self.width))
        self.vectorized_rand_norm = np.vectorize(self.generate_normal_rand, otypes=[np.float])

        # weight initialization
        # if weight initialization not set then use Xavier method for initializing weights
        'TODO: provide link'
        if weight_init != 'undefined':

            weight_std = np.sqrt(n_weight_vals)
            self.data_mtx = self.vectorized_rand_norm(self.data_mtx)

    def get_data(self, width_indx, heigh_indx, depth_indx):
        pass
        # weight_patch=
    def get_shape(self):
        return self.data_mtx.shape

    def set_data_elmt(self, i, j, depth, val):
        self.data_mtx[depth, i, j] = val

    def set_padded_mtx(self,input_mtx):
        self.padded_mtx = input_mtx

    def set_data_mtx(self,input_mtx):
        self.data_mtx = input_mtx
        self.depth , self.width,self.height = input_mtx.shape
        self.delta_data_mtx = np.reshape(np.zeros(self.depth*self.height*self.width), newshape=(self.depth, self.height, self.width))

    def get_gradient(self, x, y, depth):
        return self.delta_data_mtx[depth, x, y]

    def set_gradient(self, x, y, depth, grad_val):
        self.delta_data_mtx[depth, x, y] = grad_val

    def add_gradient(self, x, y, depth, grad_val):
        self.delta_data_mtx += grad_val

    def generate_normal_rand(self, matx_elmt):
        n_weight_vals = self.height * self.depth * self.depth
        return np.float(matx_elmt + np.random.normal(loc=0, scale=np.sqrt(n_weight_vals), size=1))


class Naive_Conv_NeuralNet_Layer(object):
    """
    A numpy based, naive implementation of a vanilla convolutional neural net layer.
    Multiple Conv layers + pooling(optional) + some final output transformations define a full Conv Net
    Details of the mathematics can be found at the from  textbook 'Deep Learning' by Goodman, Bengio et al.
    Implementation based also on stanford convolutional networks course work:
    http://cs231n.github.io/convolutional-networks/#conv
    Code is inspired by karpathy's javascript implementation of a different deep learning layers. Its amazing. So check it out.
    All(most since I looked at alot of different code in coming to try understand this stuff) goes to ya. Thanks for making your code public dude
    https://github.com/karpathy/convnetjs
    """

    def __init__(self, input_volume, no_filters, filter_map_dim=3, stride_len=1, zero_padding=1,
                 k=3,weight_init='False'):
        """
        :param input_feature_map_dim:
        :param no_filters:
        :param filter_map_dim:
        :param stride_len:
        :param zero_padding:
        :return:
        """
        self.num_channels_D, self.width_X, self.height_Y= input_volume.get_shape()
        self.n_filters = no_filters
        self.input_vol = input_volume          # need to change this to a copy of the input in the future  .copy()
        self.k = k
        self.filter_size = filter_map_dim
        self.stride_len = stride_len
        self.zero_padding = zero_padding
        self.filter_map = [Data(width=self.filter_size, height=self.filter_size, depth=self.input_vol.depth,weight_init='normal') for i in
                           range(self.n_filters)]
        self.bias_vol = Data(width=1, height=1, depth=self.input_vol.depth)
        self.zero_pad_image()  # zero pad once since zero pad parm > 0
        # Initialize weights to be applied to input_feature_map. Sample from N(0,1) as intialization weights
        if weight_init == 'True':
            pass
        # Output volume sizes
        # TODO: write a function to check if output dim are int types. If not adjust with appropriate zero padding
        while (self.width_X-self.filter_size+ 2 * self.zero_padding)/self.stride_len % 1 != 0.0:
            print('yay')
            self.zero_pad_image()
        else:
            self.Output_Width = int((self.width_X - self.filter_size + 2 * self.zero_padding) / (self.stride_len + 1))
            self.Output_Height = int((self.height_Y - self.filter_size + 2 * self.zero_padding) / (self.stride_len + 1))
            self.output_Tensor = Data(width=self.Output_Width, height=self.Output_Height, depth=self.n_filters)

    def filter_val_init(self,filter_vol):
        for filter_indx in range(len(self.filter_map)):
            self.filter_map[filter_indx].set_data_mtx(filter_vol[filter_indx,:,:])

    def im2col(self, X):
        """
        :param X: filter_dim X filter_dim area of current image to be converted to 1 X filter_dim^2 vector
        :param filter_dim:
        :return: vector of dimension 1 by filter_dim^2
        """
        flat_vect = np.reshape(X, -1)
        return flat_vect

    def convolution_op(self, image_col, filter_col):
        conv_result = np.sum(np.dot(image_col, filter_col))
        return conv_result

    def zero_pad(X):
        return np.pad(X,pad_width=(1,1),mode='constant',constant_values=0)

    def zero_pad_image(self):
        input_img =self.input_vol.data_mtx.copy()
        input_img_list = input_img.tolist()
        padded = False
        while self.zero_padding < self.filter_size and padded==False:
            if (self.width_X-self.filter_size+ 2 * self.zero_padding)/self.stride_len % 1 == 0.0 :
                for j in range(0,input_img.shape[0]):
                    input_img_list[j] = zero_pad(input_img_list[j])
                    print(input_img_list)
                padded = True
            else:
                self.zero_padding += 1
        self.input_vol.set_padded_mtx(np.asarray(input_img_list))

    def set_weights(self,input_vol=[]):
        for j in range(len(self.filter_map)):
            self.filter_map[j].set_data_mtx(input_vol[j,:,:])

    def Naive_forwardpass(self):
        """
        Forward pass of conv net implementing output map generating function.
        Returns an tensor of self.Output_Width X self.Output_Height X self.n_filters. Which is happily sent off to the
        next poor bugger layer of the CNN architecture
        Not happy too(livid actually)  with the efficiency and endless loops but c'est la vie. This is for education purposes.
        :param X: Image/feature map from previous layer/(just the picture I guess)
        :return: tensor of vol : self.Output_Width X self.Output_Height X self.n_filters
        """

        for filter_k in range(0, self.n_filters):
            filter_col = self.im2col(self.filter_map[filter_k].data_mtx)
            for hgt_indx in range(0, self.Output_Height):
                for wdth_indx in range(0, self.Output_Width):
                    wdth_start_index = wdth_indx * self.stride_len
                    wdth_end_index= wdth_start_index + self.filter_size
                    hgt_start_index = hgt_indx * self.stride_len
                    hgt_end_index = hgt_start_index + self.filter_size
                    trn_img_area = self.input_vol.padded_mtx[:, wdth_start_index:wdth_end_index,
                                   hgt_start_index:hgt_end_index]
                    trn_img_col = self.im2col(trn_img_area)
                    self.output_Tensor.data_mtx[filter_k, hgt_indx, wdth_indx] = self.convolution_op(trn_img_col,
                                                                                                     filter_col) + np.sum(self.bias_vol.data_mtx)
        return self.output_Tensor


# hopefully correct implementation of conv back prop
    def Naive_backwardpass(self):
        """
        :param X:
        :return:
        """
        # this fucking backward pass ...sigh
        # need to make two backward pass gradient calculations
        # gradient w.r.t filter weights which will be used for weight updates.
        # gradient w.r.t current image representation/ current layer which will be used as gradient flow to lower layers

        for filter_j in range(0, len(self.filter_map)):
            filter_vol = self.filter_map[
                filter_j]  # fix a single filter,reference to a class object. Make sure that changes are inplace and not new copy object
            for height_indx in range(start=0, stop=self.Output_Height):
                for width_indx in range(start=0,
                                        stop=self.Output_Width):  # fixes a single pixel , pixel i,j in layer L the output vol
                    upstream_grad = np.full(shape=(filter_vol.depth, self.filter_size, self.filter_size),
                                            fill_value=self.output_Tensor.delta_data_mtx[
                                                filter_j, width_indx, height_indx])  # get dE/d(x_{i,j}) . derivative of error w.r.t fixed pixel
                    #                    for filter_depth_indx in range(0,filter_vol.depth):
                    for filter_height_indx in range(0, filter_vol.height):
                        for filter_width_indx in range(0, filter_vol.width):
                            width_stride_dist = self.stride_len * width_indx
                            height_stride_dist = self.stride_len * height_indx
                            filter_vol.delta_data_mtx[:, filter_width_indx, filter_height_indx] += self.input_vol.data_mtx[
                                                                                                   :,
                                                                                                   width_indx + width_stride_dist,
                                                                                                   height_indx + height_stride_dist] * upstream_grad
                            self.input_vol.delta_mtx[:, width_indx + width_stride_dist,
                            height_indx + height_stride_dist] += filter_vol.data_mtx[:, filter_width_indx,
                                                                 filter_height_indx] * upstream_grad


                            # initial try at backprop changed with something else


    def Naive_backwardpass_init(self):
        """
        :param X:
        :return:
        """
        # this fucking backward pass ...sigh
        # need to make two backward pass gradient calculations
        # gradient w.r.t filter weights which will be used for weight updates.
        # gradient w.r.t current image representation/ current layer which will be used as gradient flow to lower layers

        for filter_j in range(0, len(self.filter_map)):
            filter_vol = self.filter_map[filter_j]  # fix a single filter
            for height_indx in range(start=0, stop=self.Output_Height):
                for width_indx in range(start=0,
                                        stop=self.Output_Width):  # fixes a single pixel , pixel i,j in layer L the output vol
                    upstream_grad = self.input_vol.get_gradient(filter_j, width_indx,
                                                                height_indx)  # get dE/d(x_{i,j}) . derivative of error w.r.t fixed pixel
                    for filter_depth_indx in range(0, filter_vol.depth):
                        for filter_height_indx in range(0, filter_vol.height):
                            for filter_width_indx in range(0, filter_vol.width):  # for fixed w^l_{m,n,c}
                                width_stride_dist = self.stride_len * width_indx
                                height_stride_dist = self.stride_len * height_indx
                                filter_vol.delta_data_mtx[filter_depth_indx, filter_width_indx, filter_height_indx] += \
                                self.input_vol.data_mtx[
                                    filter_depth_indx, width_indx + width_stride_dist, height_indx + height_stride_dist] * upstream_grad
                                self.input_vol.delta_mtx[
                                    filter_depth_indx, width_indx + width_stride_dist, height_indx + height_stride_dist] += \
                                filter_vol.data_mtx[
                                    filter_depth_indx, filter_width_indx, filter_height_indx] * upstream_grad
