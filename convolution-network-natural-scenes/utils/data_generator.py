from tensorflow.keras.preprocessing.image import ImageDataGenerator

'''
图片生成器tf.keras.preprocessing.image.ImageDataGenerator
用到的参数：
- rescale：输入一个整数，通常为1/255，由于图像像素都是0~255的整数，rescale可以
           让所有像素统一乘上一个数值，如果是1/255，像素会被转化为0~1之间的数。

从目录读取图片tf.keras.preprocessing.image.ImageDataGenerator.flow_from_directory
用到的参数：
- directory：图片存放路径。
- target_size：图片宽高缩放到指定的大小，默认(256, 256)。
- batch_size：每次读取的图片数，默认32。
- class_mode：类别格式，默认'categorical'。
              如果是'sparse'：类别['paper', 'rock', 'scissors'] ——> [0, 1, 2]
              如果是'categorical'：类别['paper', 'rock', 'scissors'] ——> [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
              如果是'input'：类别['paper', 'rock', 'scissors']保持不变
              如果是None：不返回标签。
'''

def train_val_generator(data_dir, target_size, batch_size, class_mode=None, subset='training'):
    train_val_datagen = ImageDataGenerator(rescale=1./255., validation_split=0.2)
    return train_val_datagen.flow_from_directory(
        directory=data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=class_mode,
        subset=subset)


def test_generator(data_dir, target_size, batch_size, class_mode=None):
    test_datagen = ImageDataGenerator(rescale=1./255.)
    return test_datagen.flow_from_directory(
        directory=data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=class_mode)


def pred_generator(data_dir, target_size, batch_size, class_mode=None):
    pred_datagen = ImageDataGenerator(rescale=1./255.)
    return pred_datagen.flow_from_directory(
        directory=data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=class_mode)