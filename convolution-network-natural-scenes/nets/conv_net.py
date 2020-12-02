from tensorflow.keras import Model
from tensorflow.keras.layers import Flatten, Conv2D, MaxPool2D, Dense, Dropout, BatchNormalization

'''
卷积操作tf.keras.layers.Conv2D
用到的参数：
- filters：输入整数，卷积核个数（等于卷积后输出的通道数）。
- kernel_size：卷积核的大小，设置为整数就表示卷积核的height = width = 指定整数。
- strides：卷积核的滑动步长，默认为(1, 1)。
- kernel_initializer：权重初始化，默认是'glorot_uniform'（即Xavier均匀初始化）。
    可选项：
    - 'RandomNormal'：正态分布采样，均值为0，标准差0.05
    - 'glorot_normal'：正态分布采样，均值为0，标准差stddev = sqrt(2 / (fan_in + fan_out))
    - 'glorot_uniform'：均匀分布采样，范围[-limit, limit]，标准差limit = sqrt(6 / (fan_in + fan_out))
    - 'lecun_normal'：正态分布采样，均值为0，标准差stddev = sqrt(1 / fan_in)
    - 'lecun_uniform'：均匀分布采样，范围[-limit, limit]，标准差limit = sqrt(3 / fan_in)
    - 'he_normal'：正态分布采样，均值为0，标准差stddev = sqrt(2 / fan_in)
    - 'he_uniform'：均匀分布采样，范围[-limit, limit]，标准差limit = sqrt(6 / fan_in)
    fan_in是输入的神经元个数，fan_out是输出的神经元个数。
- activation：激活函数。
    可选项：
    - 'sigmoid'：sigmoid激活函数
    - 'tanh'：tanh激活函数
    - 'relu'：relu激活函数
    - 'elu'或tf.keras.activations.elu(alpha=1.0)：elu激活函数
    - 'selu'：selu激活函数
    - 'swish': swish激活函数(tf2.2版本以上才有)
    - 'softmax': softmax函数
    - input_shape：如果是第一层卷积，需要设置输入图片的大小(height, width, channel)，如input_shape=(128, 128, 3)。
    - name：输入字符串，给该层设置一个名称。

池化操作tf.keras.layers.MaxPool2D
用到的参数：
- pool_size：池化的大小，设置为整数就表示池化的height = width = 指定整数。
- strides：池化的滑动步长，通常等于pool_size。
- name：输入字符串，给该层设置一个名称。

全连接操作tf.keras.layers.Dense
用到的参数：
- units：输入整数，全连接层神经元个数。
- activation：激活函数，分类网络的输出层一般用'softmax'激活函数。
- name：输入字符串，给该层设置一个名称。

展平操作tf.keras.layers.Flatten
举例说明：
[[1,2,3],
 [4,5,6],   ——>   [1,2,3,4,5,6,7,8,9]
 [7,8,9]]
'''


def get_norm(norm_type):
    if norm_type == "batch":
        return BatchNormalization()
    else:
        raise ValueError(f"Unrecognized norm_type {norm_type}")


# 定义一个子类来搭建模型
class ConvModel(Model):

    def __init__(self):
        # 父类初始化
        super(ConvModel, self).__init__()

        # 卷积层conv_1_1
        self.conv_1_1 = Conv2D(input_shape=(
            150, 150, 3), filters=32, kernel_size=3, activation='relu', name='conv_1_1')
        self.norm_1_1 = get_norm(norm_type='batch')

        # 卷积层conv_1_2
        self.conv_1_2 = Conv2D(filters=32, kernel_size=3,
                               activation='relu', name='conv_1_2')
        self.norm_1_2 = get_norm(norm_type='batch')

        # 池化层max_pool_1
        self.max_pool_1 = MaxPool2D(pool_size=2, name='max_pool_1')

        # 卷积层conv_2_1
        self.conv_2_1 = Conv2D(filters=64, kernel_size=3,
                               activation='relu', name='conv_2_1')
        self.norm_2_1 = get_norm(norm_type='batch')

        # 卷积层conv_2_2
        self.conv_2_2 = Conv2D(filters=64, kernel_size=3,
                               activation='relu', name='conv_2_2')
        self.norm_2_2 = get_norm(norm_type='batch')

        # 池化层max_pool_2
        self.max_pool_2 = MaxPool2D(pool_size=2, name='max_pool_2')

        # 展平层flatten
        self.flatten = Flatten(name='flatten')
        self.dropout = Dropout(0.5)
        self.flatten_norm = get_norm(norm_type='batch')

        # 全连接层
        self.dense = Dense(units=10, activation="softmax", name='logit')

    def call(self, x):
        x = self.conv_1_1(x)
        x = self.norm_1_1(x)
        x = self.conv_1_2(x)
        x = self.norm_1_2(x)
        x = self.max_pool_1(x)
        x = self.conv_2_1(x)
        x = self.norm_2_1(x)
        x = self.conv_2_2(x)
        x = self.norm_2_2(x)
        x = self.max_pool_2(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.flatten_norm(x)
        x = self.dense(x)
        return x
