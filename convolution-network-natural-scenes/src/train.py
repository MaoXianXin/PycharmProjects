import os
import time
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import save_model

from nets.conv_net import ConvModel
from utils.data_generator import train_val_generator
from utils.image_plot import plot_images
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

train_gen = train_val_generator(
    data_dir='../dataset/imagenette2-160/imagenette2-160/train',
    target_size=(150, 150),    # 把图片的h和w从64变成150，增大图片的分辨率
    batch_size=32,
    class_mode='categorical',
    subset='training')

val_gen = train_val_generator(
    data_dir='../dataset/imagenette2-160/imagenette2-160/train',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation')

# ImageDataGenerator的返回结果是个迭代器，调用一次才会吐一次结果，可以使用.next()函数分批读取图片。
# 取15张训练集图片进行查看
train_batch, train_label_batch = train_gen.next()
plot_images(train_batch, train_label_batch)

# 取15张测试集图片进行查看
val_batch, val_label_batch = val_gen.next()
plot_images(val_batch, val_label_batch)

# 类实例化
model = ConvModel()
callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2, mode='max', restore_best_weights=True)]

'''
模型设置tf.keras.Sequential.compile

用到的参数：
- loss：损失函数，对于分类任务，如果标签没做onehot编码，一般使用"sparse_categorical_crossentropy"，否则使用"categorical_crossentropy"。
- optimizer：优化器，这里选用"sgd"，更多优化器请查看https://tensorflow.google.cn/api_docs/python/tf/keras/optimizers。
- metrics：评价指标，这里选用"accuracy"，更多优化器请查看https://tensorflow.google.cn/api_docs/python/tf/keras/metrics。
'''

# 设置损失函数loss、优化器optimizer、评价标准metrics
model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.SGD(
        learning_rate=0.003, momentum=0.5, decay=1e-3),
    metrics=['accuracy'])

'''
模型训练tf.keras.Sequential.fit

用到的参数：
- x：输入的训练集，可以用ImageDataGenerator读取的数据。
- steps_per_epoch：输入整数，每一轮跑多少步数，这个数可以通过 图片总量/batch_size 得到，如2520/32=78.75。
- epochs：输入整数，数据集跑多少轮模型训练，一轮表示整个数据集训练一次。
- validation_data：输入的验证集，也可以用ImageDataGenerator读取的数据。
- validation_steps：输入整数，验证集跑多少步来计算模型的评价指标，一步会读取batch_size张图片，所以一共验证validation_steps * batch_size张图片。
- shuffle：每轮训练是否打乱数据顺序，默认True。

返回：
History对象，History.history属性会记录每一轮训练集和验证集的损失函数值和评价指标。
'''

history = model.fit(x=train_gen,
                    epochs=50, validation_data=val_gen,
                    shuffle=True, callbacks=callbacks)
eval_history = model.evaluate(val_gen)
print('-----val_loss-----', '\n', eval_history[0])
print('-----val_acc-----', '\n', eval_history[1])
# 画图查看history数据的变化趋势
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.xlabel('epoch')
plt.show()

'''
模型保存tf.keras.models.save_model

用到的参数：
- model：要保存的模型，也就是搭建的keras.Sequential。
- filepath：模型保存路径。
'''
# 模型保存
# 创建保存路径
# model_name = "model-" + time.strftime('%Y-%m-%d-%H-%M-%S')
# model_path = os.path.join('models', model_name)
# if not os.path.exists(model_path):
#     os.makedirs(model_path)
#
# save_model(model=model, filepath=model_path)
model.save_weights('test.h5')