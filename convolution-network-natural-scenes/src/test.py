from tensorflow.keras.models import load_model
from utils.data_generator import test_generator, pred_generator
from utils.image_plot import plot_images

test_gen = test_generator(
    data_dir='../dataset/natural-scenes/seg_test',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical')

pred_gen = pred_generator(
    data_dir='../dataset/natural-scenes/seg_pred',
    target_size=(64, 64),
    batch_size=32,
    class_mode=None)

'''
模型载入tf.keras.models.load_model
用到的参数：
- filepath：载入模型存放的路径。

模型验证tf.keras.Sequential.evaluate
用到的参数：
- x：输入的验证集，可以用ImageDataGenerator读取的数据。

模型预测tf.keras.Sequential.predict
用到的参数：
- x：需要做预测的数据集，可以用ImageDataGenerator读取的数据。
'''

model_path = '../models/model-2020-07-22-16-51-19'
loaded_model = load_model(filepath=model_path)
loss, accuracy = loaded_model.evaluate(x=test_gen)

pred_batch = pred_gen.next()
pred_result = loaded_model.predict(x=pred_batch)
plot_images(pred_batch, pred_result)