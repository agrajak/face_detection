import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import *
import matplotlib.pyplot as plt

from keras import optimizers
from keras import regularizers
import numpy as np
import os.path as p
import cv2 as cv
import os

def load_faces(ratio):
    x = list()
    y = list()
    for i in range(16):
        target_dir = p.normpath(os.getcwd()+'/gdrive/My Drive/AI/output_new/'+str(i+1))
        for f in os.listdir(target_dir):
            if p.isfile(p.join(target_dir, f)):
                img = cv.imread(p.join(target_dir, f), cv.IMREAD_COLOR)
                res = cv.resize(img, dsize=(90, 90), interpolation=cv.INTER_CUBIC)
                x.append(res)
                y.append(i)

    shuffle_list = np.arange(len(x))
    np.random.shuffle(shuffle_list)

    x_data = list()
    y_data = list()

    for i in shuffle_list:
        x_data.append(x[i])
        y_data.append(y[i])

    x_data = np.asarray(x_data, dtype=np.float32)
    y_data = np.asarray(y_data, dtype=np.float32)
    
    print('size of x_data: ', len(x_data))
    size = int(len(x_data)*ratio)
    x_train = x_data[:size]
    y_train = y_data[:size]
    x_test = x_data[size:]
    y_test = y_data[size:]
    return (x_train, y_train), (x_test, y_test)

class cifar10vgg:
    def __init__(self,train=True):
        self.num_classes = 16
        self.weight_decay = 0.0005
        self.x_shape = [90,90,3]

        self.model = self.build_model()
        if train:
            self.model = self.train(self.model)
        else:
            self.model.load_weights('cifar10vgg.h5')

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
        model.add(Activation('softmax'))

        return model

    def normalize(self,X_train,X_test):
        mean = np.mean(X_train,axis=(0,1,2,3))
        std = np.std(X_train, axis=(0,1,2,3))
        X_train = (X_train-mean)/(std+1e-7)
        X_test = (X_test-mean)/(std+1e-7)
        print('training set mean is ', mean)
        print('training set std is ', std)
        return X_train, X_test

    def normalize_production(self,x):
        mean = 120.707 #cifar10 training set mean
        std = 64.15 #cifar10 training set std
        return (x-mean)/(std+1e-7)

    def predict(self,x,normalize=True,batch_size=32):
        if normalize:
            x = self.normalize_production(x)
        return self.model.predict(x,batch_size)

    def train(self,model):
        batch_size = 128
        maxepoches = 250
        learning_rate = 0.008
        lr_decay = 1e-6
        lr_drop = 20
        
        (x_train, y_train), (x_test, y_test) = load_faces(1.0)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train, x_test = self.normalize(x_train, x_test)

        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)

        def lr_scheduler(epoch):
            return learning_rate * (0.5 ** (epoch // lr_drop))

        callbacks = [keras.callbacks.LearningRateScheduler(lr_scheduler),
                     keras.callbacks.TensorBoard(
                          log_dir='./',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True
                      ),
                      keras.callbacks.ModelCheckpoint('model.cp', monitor='val_loss', verbose = 1,
                      save_best_only=False, save_weights_only=False, mode='auto', period=1)
                      ]

        train_datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.5,
            height_shift_range=0.5,
            zoom_range = 0.4,
            shear_range=0.2,
            horizontal_flip=True,
            validation_split=0.2,
            vertical_flip=False)
        

        train_generator = train_datagen.flow(x_train, y_train,
                                        batch_size=batch_size,
                                            subset='training')
        
        validation_generator = train_datagen.flow(x_train, y_train,
                                        batch_size=batch_size,
                                                 subset="validation")

        adam = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=lr_decay, amsgrad=False)
        sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

        hist = model.fit_generator(train_datagen.flow(x_train, y_train,
                                        batch_size=batch_size),
                                        steps_per_epoch=x_train.shape[0] // batch_size,
                                        epochs=maxepoches,
                                        validation_data=validation_generator,
                                      validation_steps = (x_train.shape[0]/5) // batch_size, 
                                      callbacks=callbacks,verbose=1)
        
        
        fig, loss_ax = plt.subplots()
        acc_ax = loss_ax.twinx()

        loss_ax.plot(hist.history['loss'], 'y', label='train loss')
        loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
        loss_ax.set_xlabel('epoch')
        loss_ax.set_ylabel('loss')
        loss_ax.legend(loc='upper left')

        acc_ax.plot(hist.history['acc'], 'b', label='train acc')
        acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')
        acc_ax.set_ylabel('accuracy')
        acc_ax.legend(loc='upper left')

        plt.show()
        
        model.save_weights('cifar10vgg.h5')
        return model

if __name__ == '__main__':

    (x_train, y_train), (x_test, y_test) = load_faces(0.8)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    y_train = keras.utils.to_categorical(y_train, 16)
    y_test = keras.utils.to_categorical(y_test, 16)

    model = cifar10vgg()

    predicted_x = model.predict(x_test)
    residuals = np.argmax(predicted_x,1)!=np.argmax(y_test,1)

    loss = sum(residuals)/len(residuals)
    print("the validation 0/1 loss is: ",loss)
