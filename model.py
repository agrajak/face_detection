# VGG16 원본 소스코드는 강의노트 코드를 참조하였습니다.

from keras.models import Sequential, load_model
from keras.layers import *
from keras import optimizers
from keras import regularizers
from keras import backend as K
class vgg16:
  def __init__(self,train=True):
    self.num_classes = 16
    self.weight_decay = 0.0005
    self.x_shape = [90,90,3]

    self.model = self.build_model()
    self.model.load_weights('14.h5')

  def build_model(self):

    model = Sequential()
    weight_decay = self.weight_decay
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=self.x_shape, activation='relu',
                      kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Flatten())

    model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    
    model.add(Dense(self.num_classes))
    model.add(Lambda(lambda x: K.tf.nn.softmax(x)))

    return model

  def normalize_production(self, x):
    mean = 129.94 #cifar10 training set mean
    std = 65.56 #cifar10 training set std
    return (x-mean)/(std+1e-7)
  def predict(self, x, normalize=True, batch_size=32):
    if normalize:
        x = self.normalize_production(x)
    return self.model.predict(x,batch_size)
      