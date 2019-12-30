import os,cv2
from keras.models import load_model
from keras.utils import np_utils
from keras.layers import Activation, Dropout,Conv2D, Convolution2D, GlobalAveragePooling2D,Flatten,Dense,MaxPooling2D
from keras.models import Sequential
from keras_squeezenet import SqueezeNet
from keras.optimizers import Adam,SGD
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import pickle
import numpy as np

IMAGES_PATH = "images"
CLASS_NAME = {}
LEN_CLASSES = 0

# Take each directory and convert it into a dict
def getclassmap():
    global CLASS_NAME
    global LEN_CLASSES
    count = 0
    for directory in os.listdir(IMAGES_PATH):
        CLASS_NAME[directory] = count
        count += 1
    LEN_CLASSES = len(CLASS_NAME)

#make data ready for the model
def pre_process_data():
    dataset = []
    for dir in CLASS_NAME.keys():
        path = os.path.join(IMAGES_PATH,dir)
        if os.path.isdir(path):
            for file in os.listdir(path):
                if not file.startswith("."):
                    file_path = os.path.join(path,file)
                    img = cv2.imread(file_path,cv2.IMREAD_UNCHANGED)
                    img = cv2.resize(img,(64,64))
                    img = img[:, :, np.newaxis]
                    #print(img.shape)
                    dataset.append([img,dir])
    return dataset

def map_labels(label):
    return CLASS_NAME[label]

# CNN Model
def generate_model():

    model = Sequential()
    model.add(Conv2D(16, (2, 2), input_shape=(64, 64, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='same'))
    model.add(Conv2D(64, (5,5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(LEN_CLASSES, activation='softmax'))
    sgd = SGD(lr=1e-3)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model





getclassmap()
dataset = pre_process_data()


data , labels = zip(*dataset)

data_train, data_test, labels_train, labels_test = train_test_split(data,labels,test_size=0.33)

data_test, data_validate, labels_test, labels_validate = train_test_split(data_test,labels_test,test_size = 0.5)




labels_train = list(map(map_labels, labels_train))
labels_test = list(map(map_labels, labels_test))
labels_validate = list(map(map_labels, labels_validate))
labels_train = np_utils.to_categorical(labels_train)
labels_test = np_utils.to_categorical(labels_test)
labels_validate = np_utils.to_categorical(labels_validate)

model = generate_model()

data_train = np.array(data_train) /255.0
data_validate = np.array(data_validate)/255.0
data_test = np.array(data_test)/255.0
model.fit(np.array(data_train), np.array(labels_train),validation_data=(np.array(data_validate),np.array(labels_validate)) ,epochs=150,batch_size=25)

scores = model.evaluate(np.array(data_test), np.array(labels_test),verbose = 1)
print (scores)

model.save("my_cnn_model.h5")
