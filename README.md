# AdaResNet AdaResNet: Enhancing ResNet with Adaptive Weighting of Residual Learning and Residual Blocks

In very deep networks, gradients can become very small during backpropagation, making it difficult to train the early layers. ResNet (Residual Network) addresses this by allowing gradients to flow directly through the network via skip connections, facilitating the training of much deeper networks. 

However, in the process of skip connections, the skipped input (SI) is directly added to the processed data (PD), treating SI and PD as the same, instead of adapting to different scenarios. 

In this project, we propose AdaResNet (Auto-Adapting Residual Network), which automatically adapts the ratio of SI and PD with respect to the training data. We introduce a variable, β, to represent this ratio. This variable is adjusted automatically during the backpropagation process, allowing it to adapt to the training data rather than remaining fixed. The verification results show that AdaResNet achieves higher accuracy and faster training speeds.
