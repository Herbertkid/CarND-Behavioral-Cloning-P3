import os
import csv
from sklearn.utils import shuffle

samples = []
correction = 0.25
S_angle=[]
#load image path and angle from driving_log.csv and correct the left and right image angle
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    #load center left and right images
    first_row = next(reader)
    for line in reader:
        samples.append('./IMG/'+line[0].split('/')[-1])
        S_angle.append(float(line[3]))
        samples.append('./IMG/'+line[1].split('/')[-1])
        S_angle.append(float(line[3])+correction)
        samples.append('./IMG/'+line[2].split('/')[-1])
        S_angle.append(float(line[3])-correction)

from sklearn.model_selection import train_test_split
#split the dataset to 20%(test) and 80%(train)
train_samples, validation_samples, train_angle, validation_angle = train_test_split(samples,S_angle, test_size=0.2)

import cv2
import numpy as np
import sklearn

def generator(X, y,batch_size=32):
    num_samples = len(X)
    while 1: # Loop forever so the generator never terminates
        X,y = shuffle(X,y)
        for offset in range(0, num_samples, batch_size):
            batch_X = X[offset:offset+batch_size]
            batch_y = y[offset:offset+batch_size]
            images = []
            angles = []
            for i in range(len(batch_X)):
#                 print(batch_X[i])
                image = cv2.imread(batch_X[i])
#                 print(image.shape())
                images.append(image)
                angles.append(batch_y[i])
                #flip the image 
                n_image = np.fliplr(image)
                images.append(n_image)
                angles.append(-batch_y[i])

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
batch_size=128

# compile and train the model using the generator function
train_generator = generator(train_samples, train_angle,batch_size=batch_size)
validation_generator = generator(validation_samples, validation_angle,batch_size=batch_size)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Cropping2D,Conv2D

#CNN Architecture
model = Sequential()

model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Conv2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Conv2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Conv2D(64,3,3,activation="relu"))
model.add(Conv2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
#set fit generator and epochs=20
model.fit_generator(train_generator, steps_per_epoch=np.ceil(len(train_samples)/batch_size), validation_data=validation_generator, validation_steps=np.ceil(len(validation_samples)/batch_size), epochs=20, verbose=1)
            
            
model.save('model.h5')
print('model done')
exit()