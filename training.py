from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from keras.preprocessing.image import ImageDataGenerator
from IPython import display
from PIL import Image
import os
def createCNN(classes,cnnOutput=(32,3,3), poolSize=(2,2)):
    """
    :classes: (int) number of categories
    """
    classifier = Sequential()
    classifier.add(Convolution2D(filters = 32, kernel_size = (3,3), input_shape=(64,64,3), activation='relu'))
    classifier.add(BatchNormalization(axis=1))
    classifier.add(MaxPooling2D(pool_size=poolSize))
    classifier.add(Dropout(0.25))
    classifier.add(Convolution2D(filters=64, kernel_size=(3,3),padding="same", activation="relu" ))
    classifier.add(BatchNormalization(axis=1))
    classifier.add(MaxPooling2D(pool_size=(2,2)))
    classifier.add(Dropout(0.5))
    classifier.add(Flatten())
    classifier.add(Dense(output_dim=classes, activation='relu'))
    classifier.add(Dense(output_dim=num_classes, activation='sigmoid'))
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
    
    trainSet = trainGen.flow_from_directory(train_source, target_size=(64,64), batch_size=32, class_mode='categorical')
    testSet = testGen.flow_from_directory(test_source, target_size=(64,64), batch_size=32, class_mode='categorical')
    return trainSet, trainSet

train_dir = 'images/train/'
test_dir = 'images/test/'
num_classes = len(os.listdir('images/train'))
classifier = createCNN(num_classes)
trainSet, testSet = getDataset(classes=num_classes, train_source=train_dir, test_source=test_dir)

classifier.fit_generator(trainSet,steps_per_epoch=8000,epochs=10, validation_data=testSet, validation_steps=800)
classifier.save('models/my_model.h5')
