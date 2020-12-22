from __future__ import print_function

import os
import random

import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Conv2D, Activation, AveragePooling2D
from tensorflow.keras.layers import Input, Dense, Flatten, add
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

# FLAGS参数设置
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 4, '')
# 训练数据路径
tf.app.flags.DEFINE_string('file_path',
                           'D:\\education\\CASIA_V5\\CASIA-FaceV5',
                           'Filepattern for training data.')
tf.app.flags.DEFINE_string('model_path',
                           'D:\\git\\facenet\\models\\facenet\\20180402-114759\\20180402-114759.pb',
                           '模型保存路径')
tf.app.flags.DEFINE_integer('height', 480, '')
tf.app.flags.DEFINE_integer('width', 640, '')
tf.app.flags.DEFINE_integer('IMAGE_CHANNELS', 3, '')
tf.app.flags.DEFINE_integer('num_classes', 500, '类别数')
tf.app.flags.DEFINE_integer('epochs', 9, '训练轮数')
# 模式：训练、测试
tf.app.flags.DEFINE_string('flag', 'train', 'train or eval.')


def res_block(x, channels, i):
    if i == 1:  # 第二个block
        strides = (1, 1)
        x_add = x
    else:  # 第一个block
        strides = (2, 2)
        # x_add 是对原输入的bottleneck操作
        x_add = Conv2D(channels,
                       kernel_size=(3, 3),
                       activation='relu',
                       padding='same',
                       strides=strides)(x)

    x = Conv2D(channels,
               kernel_size=(3, 3),
               activation='relu',
               padding='same')(x)
    x = Conv2D(channels,
               kernel_size=(3, 3),
               padding='same',
               strides=strides)(x)
    x = add([x, x_add])
    Activation(K.relu)(x)
    return x


def build_model(input_shape):
    inpt = Input(shape=input_shape)

    # conv_1  28x28→28x28x16
    x = Conv2D(16,
               kernel_size=(3, 3),
               activation='relu',
               input_shape=input_shape,
               padding='same'
               )(inpt)

    # conv_2 28x28x16→14x14x16
    # x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    for i in range(2):
        x = res_block(x, 16, i)

    # conv_3 14x14x16→7x7x32
    for i in range(2):
        x = res_block(x, 32, i)

    x = AveragePooling2D(pool_size=(7, 7))(x)
    x = Flatten()(x)
    x = Dense(FLAGS.num_classes, activation='softmax')(x)

    # Construct the model.
    model = Model(inputs=inpt, outputs=x)
    plot_model(model, to_file='resnet_casiafacev5.png')
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model


def read_image(path):  # (x_train, y_train), (x_eval, y_eval)
    imagepaths = []  # len:500*5 = 2500
    labels = []  # len:500*5 = 2500
    classes = sorted(os.walk(path).__next__()[1])  # list,len:500

    for c in classes:  # c:000,001,002...499
        c_dir = os.path.join(path, c)
        walk = os.walk(c_dir).__next__()[2]  # list,['000_0.bmp','000_1.bmp','000_2.bmp','000_3.bmp','000_4.bmp']
        for sample in walk:  # 000_0.bmp
            if sample.endswith('.bmp'):
                imagepaths.append(os.path.join(c_dir, sample))
                labels.append(int(c))

    # 用train_test_split函数划分出训练集，测试集，和验证集(顺序随机）
    # X_test,y_test shape:125 ； val_imagepath,y_val shape:125  ； X_train,y_train shape:2250 ;
    train_imagepath, test_imagepath, y_train, y_test = train_test_split(imagepaths, labels, test_size=0.3,
                                                                        random_state=42)
    test_imagepath, val_imagepath, y_test, y_val = train_test_split(test_imagepath, y_test, test_size=0.3,
                                                                    random_state=24)

    '''使训练集长度为batch size的倍数'''
    # len(train_imagepath) = 2252  len(y_train) = 2252
    lentrain = len(train_imagepath)
    for _ in range(FLAGS.batch_size - lentrain % FLAGS.batch_size):
        c = classes[random.randint(0, 499)]
        c_dir = os.path.join(path, c)
        sample = os.walk(c_dir).__next__()[2][random.randint(0, 4)]
        train_imagepath.append(os.path.join(c_dir, sample))
        y_train.append(int(c))

    # one-hot格式编码
    y_train = keras.utils.to_categorical(y_train, FLAGS.num_classes)
    y_test = keras.utils.to_categorical(y_test, FLAGS.num_classes)
    y_val = keras.utils.to_categorical(y_val, FLAGS.num_classes)

    # len(train_imagepath) = 2252 ；val_imagepath 125 ；test_imagepath 125
    return (train_imagepath, y_train), (test_imagepath, y_test), (val_imagepath, y_val)


# 读取图片函数
def get_im_cv2(paths):
    images = []
    for path in paths:
        img = cv2.imread(path)
        # Reduce size
        resized = cv2.resize(img, (FLAGS.width, FLAGS.height))
        # normalize:
        resized = resized.astype('float32')
        resized /= 127.5
        resized -= 1.
        images.append(resized)
    images = np.array(images).reshape(len(paths), FLAGS.height, FLAGS.width, FLAGS.IMAGE_CHANNELS)
    images = images.astype('float32')
    return images


def get_batch(X_path, y_):
    '''
    参数：
        X_train：所有图片路径列表
        y_train: 所有图片对应的标签列表
    返回:
        一个generator，x: 获取的批次图片 y: 获取的图片对应的标签
    '''
    while 1:  # 如果没有while true的话，在一轮epoch结束后无法继续迭代，第二轮训练之前会遇到StopIteration异常
        for i in range(0, len(X_path), FLAGS.batch_size):
            x = get_im_cv2(X_path[i:i + FLAGS.batch_size])
            y = y_[i:i + FLAGS.batch_size]
            # 最重要的就是这个yield，它代表返回，返回以后循环还是会继续，然后再返回。就比如有一个机器一直在作累加运算，
            # 但是会把每次累加中间结果告诉你一样，直到把所有数加完
            yield (np.array(x), np.array(y))


def train(model, train_imagepath, y_train, val_imagepath, y_val):
    checkpoint = ModelCheckpoint(FLAGS.model_path,
                                 monitor='val_loss',
                                 verbose=1,  # 详细信息模式，0 或者 1 。
                                 save_best_only=True,
                                 save_weights_only=False,
                                 mode='auto',
                                 period=1  # 每个检查点之间的间隔（训练轮数）
                                 )

    model.fit_generator(generator=get_batch(train_imagepath, y_train),
                        steps_per_epoch=len(train_imagepath) // FLAGS.batch_size,
                        epochs=FLAGS.epochs,
                        verbose=2,
                        callbacks=[checkpoint],
                        validation_data=get_batch(val_imagepath, y_val),
                        validation_steps=1,  # 设置验证多少次数据后取平均值作为此epoch训练后的效果，val_loss,val_acc的值受这个参数直接影响
                        shuffle=True,
                        max_queue_size=3,
                        workers=1)  # 最多需要启动的进程数量


def test(model, test_imagepath, y_test):
    model.load_weights(FLAGS.model_path)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    score = model.evaluate_generator(generator=get_batch(test_imagepath, y_test),
                                     verbose=1,
                                     steps=len(test_imagepath) // FLAGS.batch_size)
    #
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


def main(_):
    input_shape = (FLAGS.height, FLAGS.width, FLAGS.IMAGE_CHANNELS)
    _model = build_model(input_shape)

    # the data, split between train and test sets
    (train_imagepath, y_train), (test_imagepath, y_test), (val_imagepath, y_val) = read_image(FLAGS.file_path)

    if FLAGS.flag == "train":
        train(_model, train_imagepath, y_train, val_imagepath, y_val)
    elif FLAGS.flag == "eval":
        test(_model, test_imagepath, y_test)


if __name__ == '__main__':
    tf.app.run()