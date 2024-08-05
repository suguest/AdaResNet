# AdaResNet AdaResNet: Enhancing ResNet with Adaptive Weighting of Residual Learning and Residual Blocks

In very deep networks, gradients can become very small during backpropagation, making it difficult to train the early layers. ResNet (Residual Network) addresses this by allowing gradients to flow directly through the network via skip connections, facilitating the training of much deeper networks. 

However, in the process of skip connections, the skipped input (SI) is directly added to the processed data (PD), treating SI and PD as the same, instead of adapting to different scenarios. 

In this project, we propose AdaResNet (Auto-Adapting Residual Network), which automatically adapts the ratio of SI and PD with respect to the training data. We introduce a variable, β, to represent this ratio. This variable is adjusted automatically during the backpropagation process, allowing it to adapt to the training data rather than remaining fixed. The verification results show that AdaResNet achieves higher accuracy and faster training speeds.

Verification:
  We use the cifar10 test data to do the verificaion.
  
1. Method. We compare the ResNet50 and AdaResNet based on Resnet50 to comparison, named cifar10Test_ResNet50.py and cifar10Test_AdaResNet.py separately.
    The main change in cifar10Test_AdaResNet.py is as follows:
    1) # Customize layers, add beta parameters
    class AddWithBeta(layers.Layer):
        def __init__(self, **kwargs):
            super(AddWithBeta, self).__init__(**kwargs)
    
        def build(self, input_shape):
            # add a trainable parameter beta
            self.beta = self.add_weight(name='beta', 
                                        shape=(1,),
                                        initializer='he_normal', 
                                        trainable=True)
            super(AddWithBeta, self).build(input_shape)
    
        def call(self, inputs):
            x, d = inputs
            return x + self.beta * d
    
        def compute_output_shape(self, input_shape):
            return input_shape[0]

    2) Define new model:
        added_output = AddWithBeta()([input_tensor, d])
        flattened_output = layers.Flatten()(added_output)
        output_tensor = layers.Dense(10, activation='softmax')(flattened_output)      
        model = models.Model(inputs=base_model.input, outputs=output_tensor)

   
2. One verification shows a more than 15% imporment in accuracy. The detailed result is:
   AdaResNet:
      Epoch 1/10
      1563/1563 ━━━━━━━━━━━━━━━━━━━━ 779s 431ms/step - accuracy: 0.2891 - loss: 3.1908 - val_accuracy: 0.3756 - val_loss: 1.7882
      Epoch 2/10
      1563/1563 ━━━━━━━━━━━━━━━━━━━━ 739s 473ms/step - accuracy: 0.4299 - loss: 1.5977 - val_accuracy: 0.3724 - val_loss: 1.7840
      Epoch 3/10
      1563/1563 ━━━━━━━━━━━━━━━━━━━━ 745s 475ms/step - accuracy: 0.4373 - loss: 1.5873 - val_accuracy: 0.4365 - val_loss: 1.8197
      Epoch 4/10
      1563/1563 ━━━━━━━━━━━━━━━━━━━━ 741s 436ms/step - accuracy: 0.4865 - loss: 1.4527 - val_accuracy: 0.4257 - val_loss: 1.7038
      Epoch 5/10
      1563/1563 ━━━━━━━━━━━━━━━━━━━━ 484s 309ms/step - accuracy: 0.5024 - loss: 1.4102 - val_accuracy: 0.4383 - val_loss: 1.8701
      Epoch 6/10
      1563/1563 ━━━━━━━━━━━━━━━━━━━━ 513s 317ms/step - accuracy: 0.5470 - loss: 1.2993 - val_accuracy: 0.5787 - val_loss: 1.2174
      Epoch 7/10
      1563/1563 ━━━━━━━━━━━━━━━━━━━━ 484s 305ms/step - accuracy: 0.6078 - loss: 1.1478 - val_accuracy: 0.5110 - val_loss: 1.5548
      Epoch 8/10
      1563/1563 ━━━━━━━━━━━━━━━━━━━━ 423s 271ms/step - accuracy: 0.6523 - loss: 1.0434 - val_accuracy: 0.3487 - val_loss: 2.9229
      Epoch 9/10
      1563/1563 ━━━━━━━━━━━━━━━━━━━━ 274s 175ms/step - accuracy: 0.6952 - loss: 0.9203 - val_accuracy: 0.6402 - val_loss: 1.1875
      Epoch 10/10
      1563/1563 ━━━━━━━━━━━━━━━━━━━━ 274s 175ms/step - accuracy: 0.7218 - loss: 0.8425 - val_accuracy: 0.6335 - val_loss: 1.3200
      313/313 ━━━━━━━━━━━━━━━━━━━━ 7s 23ms/step - accuracy: 0.6388 - loss: 1.3160 
      Test accuracy: 0.6334999799728394

   ResNet50:
       Epoch 1/10
      1563/1563 ━━━━━━━━━━━━━━━━━━━━ 455s 281ms/step - accuracy: 0.2905 - loss: 2.3935 - val_accuracy: 0.2885 - val_loss: 6.0147
      Epoch 2/10
      1563/1563 ━━━━━━━━━━━━━━━━━━━━ 439s 281ms/step - accuracy: 0.3610 - loss: 2.0191 - val_accuracy: 0.1567 - val_loss: 2.3087
      Epoch 3/10
      1563/1563 ━━━━━━━━━━━━━━━━━━━━ 639s 409ms/step - accuracy: 0.1791 - loss: 2.4902 - val_accuracy: 0.2465 - val_loss: 2.3713
      Epoch 4/10
      1563/1563 ━━━━━━━━━━━━━━━━━━━━ 767s 463ms/step - accuracy: 0.3595 - loss: 1.8868 - val_accuracy: 0.3345 - val_loss: 3.0695
      Epoch 5/10
      1563/1563 ━━━━━━━━━━━━━━━━━━━━ 726s 465ms/step - accuracy: 0.4140 - loss: 1.7874 - val_accuracy: 0.4524 - val_loss: 1.5407
      Epoch 6/10
      1563/1563 ━━━━━━━━━━━━━━━━━━━━ 688s 440ms/step - accuracy: 0.4054 - loss: 1.8089 - val_accuracy: 0.4181 - val_loss: 3.7808
      Epoch 7/10
      1563/1563 ━━━━━━━━━━━━━━━━━━━━ 603s 351ms/step - accuracy: 0.4340 - loss: 1.6688 - val_accuracy: 0.4964 - val_loss: 1.3689
      Epoch 8/10
      1563/1563 ━━━━━━━━━━━━━━━━━━━━ 467s 299ms/step - accuracy: 0.4894 - loss: 1.4900 - val_accuracy: 0.4449 - val_loss: 1.8033
      Epoch 9/10
      1563/1563 ━━━━━━━━━━━━━━━━━━━━ 480s 307ms/step - accuracy: 0.5190 - loss: 1.3942 - val_accuracy: 0.4308 - val_loss: 1.6836
      Epoch 10/10
      1563/1563 ━━━━━━━━━━━━━━━━━━━━ 510s 312ms/step - accuracy: 0.4993 - loss: 1.4523 - val_accuracy: 0.4760 - val_loss: 2.0953
      313/313 ━━━━━━━━━━━━━━━━━━━━ 26s 82ms/step - accuracy: 0.4807 - loss: 2.0483 
      Test accuracy: 0.47600001096725464
