import os
import csv
import os.path

#%%
samples = []
UseOtherCamras=True
correction = [0,0.2,-0.2] # this is a parameter to tune
#%%
#with open(r'CarND-Behavioral-Cloning-P3-master/data/driving_log.csv') as csvfile:
#    reader = csv.reader(csvfile)
#    for line in reader:
#        if 'IMG' in line[0]:
#            steeringAng=float(line[3])
#            for ii in range(UseOtherCamras*2+1):
#                if os.path.isfile(r'CarND-Behavioral-Cloning-P3-master/data/'+line[ii].strip()):
#                    tmpImgName=os.path.abspath(r'CarND-Behavioral-Cloning-P3-master/data/'+line[ii].strip())
#                    tmpSteringAng=str(steeringAng+correction[ii])
#                    samples.append([tmpImgName, tmpSteringAng])

#%%
i=0
with open(r'CarND-Behavioral-Cloning-P3-master/dataRoee/driving_log.csv') as csvfile2:
    reader = csv.reader(csvfile2)
    for line in reader:
        if 'IMG' in line[0]:
            steeringAng=float(line[3])
            for ii in range(UseOtherCamras*2+1):
                if os.path.isfile(line[ii].strip()):
                    i=i+1
                    if True:#i>=5000:
                        tmpImgName=os.path.abspath(line[ii].strip())
                        tmpSteringAng=str(steeringAng+correction[ii])
                        samples.append([tmpImgName, tmpSteringAng])

            
                    
#    reader2 = csv.reader(csvfile2)
#    for line2 in reader2:
#        if 'IMG' in line2[0]:       
#            if os.path.isfile(line2[0]):            
#                samples.append(line2)
#
#%%
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        mid_batch=batch_size//2
        for offset in range(0, (num_samples-mid_batch-1), mid_batch):
            batch_samples_a = samples[offset:offset+mid_batch]
            batch_samples_b = samples[offset+mid_batch:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples_a:
                #name = r'CarND-Behavioral-Cloning-P3-master/data/IMG/'+batch_sample[0].split('/')[-1]
                name = batch_sample[0]
                center_image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
                center_angle = float(batch_sample[1])
                images.append(center_image)
                angles.append(center_angle)
            
            for batch_sample in batch_samples_b:
                #name = r'CarND-Behavioral-Cloning-P3-master/data/IMG/'+batch_sample[0].split('/')[-1]
                name = batch_sample[0]
                center_image = cv2.flip(cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB),1)
                center_angle = -1*float(batch_sample[1])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)
#%% Test generator and offset angles:
if False:
    from matplotlib import pyplot as plt
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,30)
    fontScale              = 1
    fontColor              = (255,255,255)
    lineType               = 2
    
    aaa=next(train_generator)
    img = aaa[0][0]
    ang = aaa[1][0]
    pt1, pt2 = (160, 80), (160-int(20*np.sin(ang)), 80+int(20*np.cos(ang))) 
    cv2.arrowedLine(img, pt1, pt2, (0,0,255), 1)
    cv2.putText(img,str(ang), 
    bottomLeftCornerOfText, 
    font, 
    fontScale,
    fontColor,
    lineType)
    plt.imshow(img, cmap = 'gray')
    #cv2.imshow('asd',img)
    #cv2.waitKey()
#%%
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Dropout, Activation
#from keras.layers.convolutional import ZeroPadding2D, Convolution2D
#from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D, Conv2D
from keras.layers import Lambda
ch, row, col = 3, 80, 320  # Trimmed image format

#def converter(x):
#    #x has shape (batch, width, height, channels)
#    return (0.21 * x[:,:,:,:1]) + (0.72 * x[:,:,:,1:2]) + (0.07 * x[:,:,:,-1:])
    
model = Sequential()
#%%
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Cropping2D(cropping=((55,25), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: K.tf.image.rgb_to_grayscale(x)))
#model.add(Lambda(lambda x: converter(x)))
#%% 
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(row, col,1), output_shape=(row, col,1)))
# preprocess data
#%%
model.add(Conv2D(24, kernel_size=(5, 5), activation='relu', strides=(2,2)))
#%%
model.add(Conv2D(36, kernel_size=(5, 5), activation='relu', strides=(2,2)))
model.add(Conv2D(48, kernel_size=(5, 5), activation='relu', strides=(2,2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
#%%
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(1))

for layer in model.layers:
    print(layer.get_output_at(0).get_shape().as_list())

#%%
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, 
            nb_val_samples=len(validation_samples), nb_epoch=3)

model.save('modelGuy_5kimg_withFlip_with3Cams_gra3.h5')
import pickle
with open('modelGuy_5kimg_withFlip_with3Cams_gra3.pickle', 'wb') as handle:
    pickle.dump(history_object.history['loss'], handle, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(history_object.history['val_loss'], handle, protocol=pickle.HIGHEST_PROTOCOL)

numpy_loss_history = np.array(history_object.history['loss'])
np.savetxt("loss_history.txt", numpy_loss_history, delimiter=",")
numpy_val_loss_history = np.array(history_object.history['val_loss'])
np.savetxt("val_loss_history.txt", numpy_val_loss_history, delimiter=",")

 #%%

with open('modelHistory_myVgg_3ImgAngle0p2Correction2.pickle', 'rb') as handle:
    historyLoss = pickle.load(handle)
    historyValLoss = pickle.load(handle)
#%%    
from matplotlib import pyplot as plt
### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()