# 网络架构
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 创建一个序列模型
model = models.Sequential()

# 添加一系列卷积层、池化层、全连接层等组成的神经网络。
# 用了4个卷积层和4个池化层。其中，卷积层的卷积核数量依次为32、64、128、128，大小均为3x3；池化层的窗口大小均为2x2。
model.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
# layers.Flatten()函数用于将输入数据展平成一维向量，常用于卷积神经网络中卷积层之后的全连接层。
# 例如，在这个程序中，4个卷积层和4个池化层之后，我们通过Flatten()将卷积层的输出展平成一维向量，然后再接上一个全连接层来进行分类。
model.add(layers.Flatten())
# layers.Dropout(0.5)函数用于在训练过程中对输入数据进行随机失活(dropout)处理，以降低模型的过拟合风险。
# 0.5表示50%的输入数据会被随机失活，这个值可以根据实际情况进行调整。
model.add(layers.Dropout(0.5))
# layers.Dense(512, activation='relu')函数用于添加一个全连接层，其中512表示输出的维度，activation='relu'表示激活函数类型为ReLU。
# 在这个程序中，我们添加了一个512个神经元的全连接层，用于将展平后的卷积层输出映射到一个512维的特征向量。
model.add(layers.Dense(512, activation='relu'))
# layers.Dense(1, activation='sigmoid')函数用于添加一个二元分类器，其中1表示输出的维度，activation='sigmoid'表示激活函数类型为sigmoid。
# 在这个程序中，我们将全连接层的输出通过sigmoid函数映射到一个0-1之间的概率值，用于对图像进行二元分类（猫和狗）。
model. add(layers.Dense(1, activation='sigmoid'))

# 打印模型的结构和参数数量。
print(model.summary())

# 模板的编译
# 打印模型的结构和参数数量。
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(learning_rate=1e-4),
              metrics=['acc'])


# 利用数据增强器训练网络
# 定义训练集和验证集的路径。
train_dir='./cats_and_dogs_small/train'
validation_dir='./cats_and_dogs_small/validation'

# 定义数据增强器，包括对图像进行旋转、平移、剪切、缩放、水平翻转等操作。
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,)
test_datagen = ImageDataGenerator(rescale=1./255)


# 创建训练集和验证集的数据生成器。
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(150, 150),
                                                    batch_size=32,
                                                    class_mode='binary')


validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                        target_size=(150, 150),
                                                        batch_size=32,
                                                        class_mode='binary')


# 拟合函数
# 使用数据生成器拟合模型，并记录训练过程的历史数据。
# 拟合函数这里改动了下，原来的steps_per_epoch=100，运行时会出错
# 原因是数据集量变小，结合运行错误提示，上限可以到63，因此这里改为steps_per_epoch=63；
# 同理， validation_steps也应该随着改变，改为 validation_steps=32，以下代码已做更正。
history = model.fit(train_generator,
                              # steps_per_epoch=100,
                              steps_per_epoch=63,
                              epochs=120,
                              validation_data=validation_generator,
                              validation_steps=32) # 改为32


# 保存模板
model.save('cats_and_dogs.h5')

# 输出结果
# 绘制训练过程中的准确率和损失函数变化曲线，用于分析模型的训练效果。
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


