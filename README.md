# AdaResNet: Enhancing ResNet with Adaptive Weighting of Input Data and Processed Data

In very deep networks, gradients can become very small during backpropagation, making it difficult to train the early layers. ResNet (Residual Network) addresses this by allowing gradients to flow directly through the network via skip connections, facilitating the training of much deeper networks. 

However, in the process of skip connections, the input data (ipd) is directly added to the transformed data (tfd), treating ipd and tfd as the same, instead of adapting to different scenarios. 

In this project, we propose AdaResNet (Auto-Adapting Residual Network), which automatically adapts the ratio of ipd and tfd with respect to the training data. We introduce a variable, weight, to represent this ratio. This variable is adjusted automatically during the backpropagation process, allowing it to adapt to the training data rather than remaining fixed. The verification results show that AdaResNet achieves higher accuracy and faster training speeds.

For any advice to improve this project or questions about this porject, please contact suguest@126.com or suhong@cuit.edu.cn.

Verification:
  We use the cifar10 test data to do the verificaion.
  
Method. We compare the ResNet50 and AdaResNet based on Resnet50 to comparison, named cifar10Test_ResNet50.py and cifar10Test_AdaResNet.py separately.
    The main change in cifar10Test_AdaResNet.py is as follows:、
      class AddWithWeight(layers.Layer):
        def __init__(self, **kwargs):
            super(AddWithWeight, self).__init__(**kwargs)
    
        def build(self, input_shape):
            self.Weight = self.add_weight(name='Weight', 
                                        shape=(1,),
                                        initializer='zeros',
                                        trainable=True)
            super(AddWithWeight, self).build(input_shape)
    
        def call(self, inputs):
            x, d = inputs
            return d + self.Weight * x

 
