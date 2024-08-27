import tensorflow as tf
from tensorflow.keras import layers, models, datasets
from tensorflow.keras.utils import to_categorical

# 自定义层，添加 beta 参数
class AddWithBeta(layers.Layer):
    def __init__(self, **kwargs):
        super(AddWithBeta, self).__init__(**kwargs)

    def build(self, input_shape):
        self.beta = self.add_weight(name='beta', 
                                    shape=(1,),
                                    initializer='zeros',
                                    trainable=True)
        super(AddWithBeta, self).build(input_shape)

    def call(self, inputs):
        x, d = inputs
        return d + self.beta * x

# 定义残差块
def identity_block(x, filters, kernel_size=3):
    x_skip = x
    
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    x = AddWithBeta()([x_skip, x])
    x = layers.ReLU()(x)
    
    return x

def convolutional_block(x, filters, kernel_size=3, strides=2):
    x_skip = x
    
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    x_skip = layers.Conv2D(filters, 1, strides=strides, padding='same')(x_skip)
    x_skip = layers.BatchNormalization()(x_skip)
    
    x = AddWithBeta()([x_skip, x])
    x = layers.ReLU()(x)
    
    return x

# 构建自定义ResNet模型
def build_custom_resnet(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)
    
    x = convolutional_block(x, 64)
    x = identity_block(x, 64)
    
    x = convolutional_block(x, 128)
    x = identity_block(x, 128)
    
    x = convolutional_block(x, 256)
    x = identity_block(x, 256)
    
    x = convolutional_block(x, 512)
    x = identity_block(x, 512)
    
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return models.Model(inputs, outputs)

# 加载CIFAR-10数据集
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# 预处理数据
train_images, test_images = train_images / 255.0, test_images / 255.0
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 构建和编译模型
model = build_custom_resnet(input_shape=(32, 32, 3), num_classes=10)
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# 训练模型
history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))

# 评估模型
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc}")

# 绘制训练和验证的损失及准确率
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
