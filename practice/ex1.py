from keras import applicationszzz
from keras.applications import xception
from keras.applications import re


import importlib.util
spec = importlib.util.spec_from_file_location("applications")
foo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo)
foo.MyClass()

models_DL = ['Xception','VGG16','VGG19','ResNet50','ResNet101','ResNet152','ResNet50V2','ResNet101V2','ResNet152V2',
          'InceptionV3','InceptionResNetV2','MobileNet','MobileNetV2','DenseNet121','DenseNet169','DenseNet201',
          'NASNetMobile','NASNetLarge','EfficientNetB0','EfficientNetB1','EfficientNetB2','EfficientNetB3',
          'EfficientNetB4','EfficientNetB5','EfficientNetB6','EfficientNetB7']


# 되는거 같은거 21.02.19. 오후 9시
def keras_whole_models(subdirs):
    '''
    training dataset through whole keras.application deep learning architecture

    :param subdirs: subdirectories
    :return:
    '''
    # subdirs=['F:\\Dataset\\Schizophrenia_CNN\\image_tif_copy\\Train\\h', 'F:\\Dataset\\Schizophrenia_CNN\\image_tif_copy\\Train\\s','F:\\Dataset\\Schizophrenia_CNN\\image_tif_copy\\Validation\\h','F:\\Dataset\\Schizophrenia_CNN\\image_tif_copy\\Validation\\s','F:\\Dataset\\Schizophrenia_CNN\\image_tif_copy\\Test\\h','F:\\Dataset\\Schizophrenia_CNN\\image_tif_copy\\Test\\s']
    dirs_train = [train for train in subdirs if 'Train' in train]
    dir_train = os.path.dirname(dirs_train[0])
    dirs_test = [test for test in subdirs if 'Test' in test]
    dir_test = os.path.dirname(dirs_test[0])
    dirs_val = [val for val in subdirs if 'Validation' in val]
    dir_val = os.path.dirname(dirs_val[0])

    # find whole functions in keras.applications such as Xception, VGG16, VGG19 ...

    # such names Xception, VGG16, VGG19 are not module, it's function name
    func_namesAPP = [func_APP[0] for func_APP in getmembers(APP, isfunction)]
    model_final_arch = {}
    for func_name in func_namesAPP:
        print(func_name)
        # func_name = func_namesAPP[0]
        try:
            model_DL = getattr(APP, func_name)
            model_APP = model_DL(weights=None, include_top=False, input_shape=(150, 150, 3), pooling=None)
            # model_APP.summary()
            print(model_APP.layers[-1].output_shape)
            model_final_arch[func_name] = model_APP.layers[-1].output_shape
        except:
            print("Exception : {}".format(func_name))

        additional_model = models.Sequential()
        additional_model.add(model_APP)
        additional_model.add(layers.Flatten())
        # additional_model.add(layers.Dense(4096, activation='relu'))
        additional_model.add(layers.Dense(2048, activation='relu'))
        # additional_model.add(layers.Dense(1024, activation='relu'))
        additional_model.add(layers.Dense(512, activation='relu'))
        additional_model.add(layers.Dense(1, activation='relu'))
        additional_model.summary()


# model_final_arch =
# {'DenseNet121': (None, 4, 4, 1024),
#  'DenseNet169': (None, 4, 4, 1664),
#  'DenseNet201': (None, 4, 4, 1920),
#  'InceptionResNetV2': (None, 3, 3, 1536),
#  'InceptionV3': (None, 3, 3, 2048),
#  'MobileNet': (None, 4, 4, 1024),
#  'MobileNetV2': (None, 5, 5, 1280),
#  'NASNetLarge': (None, 5, 5, 4032),
#  'NASNetMobile': (None, 5, 5, 1056),
#  'ResNet101': (None, 5, 5, 2048),
#  'ResNet101V2': (None, 5, 5, 2048),
#  'ResNet152': (None, 5, 5, 2048),
#  'ResNet152V2': (None, 5, 5, 2048),
#  'ResNet50': (None, 5, 5, 2048),
#  'ResNet50V2': (None, 5, 5, 2048),
#  'VGG16': (None, 4, 4, 512),
#  'VGG19': (None, 4, 4, 512),
#  'Xception': (None, 5, 5, 2048)}


def keras_find_py_module():
    '''
    ERROR :: DOESN't WORK
    find all .py file name in keras.application
    ['densenet.py', 'mobilenet.py', 'nasnet.py', 'resnet.py', 'resnet50.py', 'vgg16.py', 'vgg19.py', 'xception.py']

    however this doesn't work
    because application name is DenseNet121 instead of densenet.py name, densenet


    :return: keras.application module
    '''
    modules = [module for module in os.listdir(os.path.dirname(APP.__file__)) if '_' not in module]
    for module in modules:
        module_real = module[:-3]
        print(module_real)
        module_DL = getattr(APP, module_real)
        module_DL(weights=None, include_top=False, input_shape=(150, 150, 3), pooling=None)
        print(DL_VGG16_poolingNone.layers[-1].output_shape)
        module_real


def keras_whole_module():
    # find whole functions in keras.applications such as Xception, VGG16, VGG19 ...
    from inspect import getmembers, isfunction
    # such names Xception, VGG16, VGG19 are not module, it's function name
    func_namesAPP = [func_APP[0] for func_APP in getmembers(APP, isfunction)]
    model_final_arch = {}
    for func_name in func_namesAPP:
        print(func_name)
        # func_name = func_namesAPP[0]
        try:
            model_DL = getattr(APP, func_name)
            model_APP = model_DL(weights=None, include_top=False, input_shape=(150, 150, 3), pooling=None)
            # model_APP.summary()
            print(model_APP.layers[-1].output_shape)
            model_final_arch[func_name] = model_APP.layers[-1].output_shape
        except:
            print("Exception : {}".format(func_name))

        additional_model = models.Sequential()
        additional_model.add(model_APP)
        additional_model.add(layers.Flatten())
        # additional_model.add(layers.Dense(4096, activation='relu'))
        additional_model.add(layers.Dense(2048, activation='relu'))
        # additional_model.add(layers.Dense(1024, activation='relu'))
        additional_model.add(layers.Dense(512, activation='relu'))
        additional_model.add(layers.Dense(1, activation='relu'))
        additional_model.summary()


batch_size = 20
datagen = ImageDataGenerator(rescale=1. / 255)

# input_shape
# from keras, using VGG16
# input_tensor=Input(shape=input_shape, dtype='float32', name ='input')
# DL_VGG16 = VGG16(weights='imagenet', include_top=False, input_shape =(150,150,3) , pooling='max')
DL_VGG16 = VGG16(weights=None, include_top=False, input_shape=(150, 150, 3), pooling='max')

DL_VGG16.summary()

DL_VGG16_poolingNone = VGG16(weights=None, include_top=False, input_shape=(150, 150, 3), pooling=None)
DL_VGG16_poolingNone.layers[-1].output_shape

DL_VGG16_poolingNone.summary()
# conv_base = VGG16(weights='imagenet', include_top = False, input_shape=(150,150,3))
# conv_base.summary()


additional_model = models.Sequential()
additional_model.add(DL_VGG16_poolingNone)
additional_model.add(layers.Flatten())
additional_model.add(layers.Dense(4096, activation='relu'))
additional_model.add(layers.Dense(2048, activation='relu'))
additional_model.add(layers.Dense(1024, activation='relu'))
additional_model.add(layers.Dense(4, activation='relu'))
additional_model.summary()

from keras.applications import ResNet50
from keras.layers import Dense, Input, Activation
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization

input = Input(shape=(224, 224, 3))
DL_ResNet50 = ResNet50(input_tensor=input, include_top=False, weights=None, pooling='max')

x = DL_ResNet50.output
x = Dense(1024, name='fully', init='uniform')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dense(512, init='uniform')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dense(3, activation='softmax', name='softmax')(x)
model = Model(model.input, x)
model.summary()

conv_base.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
conv_base.add(layers.Dropout(0.5))
conv_base.add(layers.Dense(1, activation='sigmoid'))
conv_base.compile(opimizer=optimizers.RMSprop(lr=2e-5), loss='binary_crossentropy', metrics=['acc'])


def extract_features(directory, sample_count):
    '''

    :param directory:
    :param sample_count:
    :return:
    '''
    # directory= dir_train
    # sample_count=2000

    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    # generator = datagen.flow_from_directory(directory, target_size=(150,150), batch_size = batch_size, class_mode='binary')
    generator = datagen.flow_from_directory(directory, batch_size=batch_size,
                                            class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size:(i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels


def DL_architecture(DL, input_shape):
    '''

    :param DL:
    :param input_shape:
    :return:
    '''

    from keras.applications import VGG16

    x = DL_ResNet50.output
    x = Dense(1024, name='fully', init='uniform')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(512, init='uniform')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(3, activation='softmax', name='softmax')(x)
    model = Model(model.input, x)
    model.summary()


# example
i = 0
for inputs_batch, labels_batch in generator:

    print('inputs_batch.shape : ', inputs_batch.shape)
    print('labels_batch.shape : ', labels_batch.shape)
    i += 1
    if i > 3:
        break

# example


features_train, labels_train = extract_features(dir_train, 2000)
features_valid, labels_valid = extract_features(dir_val, 1000)
features_test, labels_test = extract_features(dir_val, 1000)

# reshape
features_train = np.reshape(features_train, 2000)
features_valid = np.reshape(features_train, 1000)
features_test = np.reshape(features_train, 1000)

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(opimizer=optimizers.RMSprop(lr=2e-5), loss='binary_crossentropy', metrics=['acc'])

history = model.fit(features_train, labels_train, epochs=30, batch_size=20,
                    validataion_data=(features_valid, labels_valid))

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


