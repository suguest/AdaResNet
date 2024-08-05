import tensorflow as tf
from tensorflow.keras import layers, models, datasets
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K

# 自定义层，添加 beta 参数
class AddWithBeta(layers.Layer):
    def __init__(self, **kwargs):
        super(AddWithBeta, self).__init__(**kwargs)

    def build(self, input_shape):
        # 添加可训练参数 beta
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

# 加载CIFAR-10数据集
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# 预处理数据
train_images, test_images = train_images / 255.0, test_images / 255.0
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 定义输入
input_tensor = layers.Input(shape=(32, 32, 3))

# 构建ResNet模型
base_model = ResNet50(weights=None, include_top=False, input_tensor=input_tensor, pooling='avg')

# 获取中间层的输出
x = base_model.output

# 假设我们选择某个层的输出 d
d = layers.Dense(32 * 32 * 3, activation='relu')(x)  # 调整形状
d = layers.Reshape((32, 32, 3))(d)  # 重塑形状以匹配输入

# 添加自定义的 AddWithBeta 层
added_output = AddWithBeta()([input_tensor, d])

# 展平并添加输出层，调整形状以匹配标签
flattened_output = layers.Flatten()(added_output)
output_tensor = layers.Dense(10, activation='softmax')(flattened_output)

# 构建新的模型
model = models.Model(inputs=base_model.input, outputs=output_tensor)

# 编译模型
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
