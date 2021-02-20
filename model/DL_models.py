#imports
import os, sys
import time, math

#import 3rd party
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras import models, layers, optimizers
from keras import applications as APP
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from inspect import getmembers, isfunction


def keras_one_model(func_name, inputshape =(150,150) , param_denselayer=[2048,512,1], weights = None):
    '''
    train one model from the keras.application
    only thing to adjust is to decide which denselayer parameter should be applied

    :param func_name: model name in keras.application
    :param input_shape: for input shape, if you put [150, 150], input_shape will be (150,150,3) for RGB colored input
    :param param_denselayer: list, denselayer parameter
    :param weights: weights, None or imagenet
    :return: deep learning architecture
    '''
    #existing DL architecture in keras.applications
    inputshape=inputshape+(3,)

    model_DL = getattr(APP, func_name)
    model_APP = model_DL(weights=weights, include_top=False, input_shape=inputshape, pooling=None)

    #adjusting dense layer
    additional_model = models.Sequential()
    additional_model.add(model_APP)
    additional_model.add(layers.Flatten())
    for param in param_denselayer:
        additional_model.add(layers.Dense(param, activation='relu'))
    return additional_model

def keras_whole_models(subdirs, weights='imagenet', batch_size=100, epochs=30, param_denselayer=[2048,512,1]):
    '''
    training dataset through whole keras.application deep learning architecture
    and save the trained model and save the plot

    :param subdirs: subdirectories
        subdirs =['F:\\Dataset\\Schizophrenia_CNN\\image_tif_copy\\Train\\h',
                 'F:\\Dataset\\Schizophrenia_CNN\\image_tif_copy\\Train\\s',
                 'F:\\Dataset\\Schizophrenia_CNN\\image_tif_copy\\Validation\\h',
                 'F:\\Dataset\\Schizophrenia_CNN\\image_tif_copy\\Validation\\s',
                 'F:\\Dataset\\Schizophrenia_CNN\\image_tif_copy\\Test\\h',
                 'F:\\Dataset\\Schizophrenia_CNN\\image_tif_copy\\Test\\s']
    :param weights: None or 'imagenet' which is given in keras.application
    :param batch_size: batch_size
    :param epochs: epochs
    :param param_denselayer: list value

            layer(type)                               output Shape
            vgg16(keras.application model)            (None, 4, 4, 512)
            flatten(Flatten)                          (None, 8192)
            dense_1(Dense)                            (None, 2048 [this can be adjusted by user converting argument using list])
            dense_2(Dense)                            (None, 512  [this can be adjusted by user converting argument using list])
            dense_3(Dense)                            (None, 1    [this can be adjusted by user converting argument using list])

    :return:
    '''


    # weights = 'imagenet'
    # epochs =30
    # batch_size=100
    # subdirs=['F:\\Dataset\\Schizophrenia_CNN\\image_tif_copy\\Train\\h', 'F:\\Dataset\\Schizophrenia_CNN\\image_tif_copy\\Train\\s','F:\\Dataset\\Schizophrenia_CNN\\image_tif_copy\\Validation\\h','F:\\Dataset\\Schizophrenia_CNN\\image_tif_copy\\Validation\\s','F:\\Dataset\\Schizophrenia_CNN\\image_tif_copy\\Test\\h','F:\\Dataset\\Schizophrenia_CNN\\image_tif_copy\\Test\\s']


    dirs_train = [train for train in subdirs if 'Train' in train]
    dir_train = os.path.dirname(dirs_train[0])
    dirs_test = [test for test in subdirs if 'Test' in test]
    dir_test = os.path.dirname(dirs_test[0])
    dirs_val = [val for val in subdirs if 'Validation' in val]
    dir_val = os.path.dirname(dirs_val[0])

    #data generator
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # find whole functions in keras.applications such as Xception, VGG16, VGG19 ...
    # such names Xception, VGG16, VGG19 are not module, it's function name
    func_namesAPP = [func_APP[0] for func_APP in getmembers(APP, isfunction)]

    model_final_arch = {}
    for func_name in func_namesAPP:
        # func_name=func_namesAPP[-4]
        print(func_name)
        try:
            model = keras_one_model(func_name, inputshape =(150,150) , param_denselayer=param_denselayer, weights=weights)
        except:
            print("Exception : {}".format(func_name))
        model.summary()
        model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])
        train_generator = train_datagen.flow_from_directory(dir_train, target_size=(150, 150), batch_size=batch_size,
                                                            class_mode='binary')
        validation_generator = test_datagen.flow_from_directory(dir_val, target_size=(150, 150), batch_size=batch_size,
                                                                class_mode='binary')

        steps_per_epoch=cal_batch_steps_per_epoch(dir_train, batch_size)
        history = model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs,
                                      validation_data=validation_generator, validation_steps=50)
        time_cur = time.strftime('%Y%m%d')
        save_model_name = '{}_{}_{}'.format(time_cur, func_name, 'pretrain' if weights=='imagenet' else _)
        dir_save_file = os.path.join(os.getcwd(), 'save_model', save_model_name+'.h5')
        model.save(dir_save_file)
        save_plot(history, save_model_name, show=False, save=True)
        print()



def save_plot(history, dir_save, show=False, save=True):
    '''
    plot the trained model and save

    :param history: keras history class
    :param show: binary
    :param save: binary
    :return:
    '''
    # dir_save='20210220_VGG16_pretrain'
    dir_save[9:]


    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy on {}'.format(dir_save[9:]))
    plt.legend()
    dir_save_file = os.path.join(os.getcwd(), 'save_model', dir_save+"_Acc")
    plt.savefig(dir_save_file)

    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss{}'.format(dir_save[9:]))
    plt.legend()
    plt.show()
    dir_save_file = os.path.join(os.getcwd(), 'save_model', dir_save+"_loss")
    plt.savefig(dir_save_file)

    plt.close('all')

def cal_batch_steps_per_epoch(dir_train, batch):
    '''
    calculate steps_per_epoch in terms of batch, and number of training dataset

    :param dir_train: train dataset directory
    :param batch: batch size
    :return: number of steps_per_epoch
    '''
    # dir_train = 'F:\\Dataset\\Schizophrenia_CNN\\image_tif_copy\\Train'
    # batch =100
    num_files = 0
    for cls in os.listdir(dir_train):
        print(cls)
        dir_cls = os.path.join(dir_train, cls)
        num_files +=len(os.listdir(dir_cls))
    batch_size =math.ceil(num_files/batch)

    return batch_size


# 이거 확인해볼것
# # plot해서 저장하는거
# # .h로 save해야됨
#
# from PIL import Image
# dir_tif = 'F:\\Dataset\\Schizophrenia_CNN\\image_tif'
# for i in os.listdir(dir_tif):
#     print(i)
#     pwd_tif = os.path.join(dir_tif, i)
# im = Image.open(pwd_tif)
# import numpy
# imarray= numpy.array(im)
# imarray[:, :, 0].max()
# imarray[:, :, 0].min()
# imarray[:, :, 1].max()
# imarray[:, :, 1].min()
# imarray[:, :, 2].max()
# imarray[:, :, 2].min()
#
#
# train_datagen = ImageDataGenerator(rescale=1./255)
# test_datagen = ImageDataGenerator(rescale=1./255)
# train_generator = train_datagen.flow_from_directory(dir_train, target_size=(150, 150), batch_size=100, class_mode ='binary')
# validation_generator = test_datagen.flow_from_directory(dir_val, target_size=(150,150), batch_size=100, class_mode='binary')
#
# for data_batch, labels_batch in train_generator:
#     print('data_batch.shape : ', data_batch.shape)
#     print('labels_batch.shape : ', labels_batch.shape)
#     break
#
# history=model.fit_generator(train_generator, steps_per_epoch=10, epochs=2, validation_data=validation_generator, validation_steps=50)
# dir_save_file = 'F:\\Dataset\\Schizophrenia_CNN\\save_model\\model.h5'
# model.save('dir_save_file')