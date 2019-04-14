from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout,Activation
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from IPython import display
from PIL import Image
import os
import numpy as np
from keras.preprocessing import image
from keras.callbacks import EarlyStopping, ModelCheckpoint
import json
def modelMaker(num_classes):
    from keras.applications.inception_v3 import InceptionV3
    conv_base = InceptionV3(weights='imagenet', include_top=False, input_shape=(150,150,3))
    model = Sequential()
    model.add(conv_base)
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])
    return model

def makeModel(num_classes):
    classifier = Sequential()
    # First section
    classifier.add(Convolution2D(filters=32, kernel_size=(3,3), padding="same", input_shape=(96,96,3), activation="relu" ))
    classifier.add(BatchNormalization(axis=1))
    classifier.add(MaxPooling2D(pool_size=(3,3)))
    classifier.add(Dropout(0.25))
    
    #Second section
    classifier.add(Convolution2D(filters=64, kernel_size=(3,3),padding="same", activation="relu" ))
    classifier.add(BatchNormalization(axis=1))
    classifier.add(Convolution2D(filters=64, kernel_size=(3,3),padding="same", activation="relu" ))
    classifier.add(BatchNormalization(axis=1))
    classifier.add(MaxPooling2D(pool_size=(2,2)))
    classifier.add(Dropout(0.25))
    
    #Third section
    classifier.add(Convolution2D(filters=128, kernel_size=(3,3),padding="same", activation="relu" ))
    classifier.add(BatchNormalization(axis=1))
    classifier.add(Convolution2D(filters=128, kernel_size=(3,3),padding="same", activation="relu" ))
    classifier.add(BatchNormalization(axis=1))
    classifier.add(MaxPooling2D(pool_size=(2,2)))
    classifier.add(Dropout(0.25))
    
    #Final Fully connected --> Relu layer
    classifier.add(Flatten())
    classifier.add(Dense(units=1024, activation="relu"))
    classifier.add(BatchNormalization())
    classifier.add(Dropout(0.5))
    
    classifier.add(Dense(units=num_classes, activation="softmax"))
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])
    return classifier
    

def getDataset(classes,train_source,test_source):
    trainGen = ImageDataGenerator(
    rescale = 1.0/255,
    shear_range = 0.1,
    zoom_range = 0.2,
    horizontal_flip = True,
    vertical_flip=True
    )
    
    testGen = ImageDataGenerator(rescale = 1.0/255)
    
    trainSet = trainGen.flow_from_directory(train_source, target_size=(96,96), batch_size=32, class_mode='categorical')
    testSet = testGen.flow_from_directory(test_source, target_size=(96,96), batch_size=32, class_mode='categorical')
    return trainSet, trainSet

train_dir = 'images/train/'
test_dir = 'images/test/'
num_models = len(os.listdir('models/'))
num_classes = len(os.listdir('images/train'))
#classifier = modelMaker(num_classes)
classifier = makeModel(9)
trainSet, testSet = getDataset(classes=num_classes, train_source=train_dir, test_source=test_dir)
with open("classes.txt","w+") as file:
        file.write(json.dumps(trainSet.class_indices))
es = EarlyStopping(monitor='val_acc', mode='max', min_delta=0.5)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
classifier.fit_generator(trainSet,steps_per_epoch=8000,epochs=10,
        validation_data=testSet, validation_steps=800, callbacks=[mc])
classifier.save('models/my_model{}.h5'.format(num_models))
