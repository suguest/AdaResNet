import tensorflow as tf
from tensorflow.keras import layers, models, datasets
from tensorflow.keras.utils import to_categorical

# 
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
        
# 
class AddWithWeight2(layers.Layer):
    def __init__(self, **kwargs):
        super(AddWithWeight2, self).__init__(**kwargs)

    def build(self, input_shape):
        self.Weight = self.add_weight(name='Weight', 
                                    shape=(1,),
                                    initializer='RandomNormal',
                                    trainable=True)
        super(AddWithWeight2, self).build(input_shape)

    def call(self, inputs):
        x, d = inputs
        return d + self.Weight * x

# 
def identity_block(x, filters, kernel_size=3):
    x_skip = x
    
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    x = AddWithWeight()([x_skip, x])
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
    
    x = AddWithWeight2()([x_skip, x])
    x = layers.ReLU()(x)
    
    return x

# 
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

# 
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# 
train_images, test_images = train_images / 255.0, test_images / 255.0
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 
model = build_custom_resnet(input_shape=(32, 32, 3), num_classes=10)
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# 
history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))

# 
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc}")

