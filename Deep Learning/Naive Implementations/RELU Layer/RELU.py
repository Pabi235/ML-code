class ReLu(object):
    def __inti__(self,input_vol):
        self.input_vol = input_vol
        self.vectorized_RELU=np.vectorize(self.scalar_RELU,otypes=[np.float])

    def Naive_forwardpass(self):
        self.Output_tensor = self.input_vol.copy
        self.Output_tensor = self.vectorized_RELU(self.Output_tensor.data_mtx)
        return self.Output_tensor

    def Naive_backwardpass(self,upstream_gradient):
        return upstream_gradient * np.array((self.input_vol >=0))

    def scalar_RELU(self,x):
        return np.max((0,x))
