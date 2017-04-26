##import important libs and packages
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from math import floor
from ket_utility import *

#debugging switch
debug_prt=0
debug_hist=0
num_bins = 25

## log file is located '/folder/driving_log.csv'
## IMG file is located '/folder/IMG'
##file1_folder='./Data_good_combined/'
#we may include any number of files
file2_folder='./Data_new/'
#
file_folders=[file2_folder]
#MAC, Linux split_char='/', WINDOWS '\\'
split_char='/'
#get all the lines, with local paths
lines=get_lines(file_folders, split_char)
if debug_prt:
    print('number of lines before zero speed', len(lines))
#histogram of steering angles from raw data
if debug_hist:
    angles=[]
    for line in lines:
        angles.append(float(line[3]))

    avg_samples_per_bin = len(angles)/num_bins
    hist, bins = np.histogram(angles, num_bins)
    myx=1.0/hist
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.plot((np.min(angles), np.max(angles)), (avg_samples_per_bin, avg_samples_per_bin), 'k-')
    plt.show()
#end of hist

# purge_zero_speed from ket_utility.
# this is to remove zero speed since no driving at that speed
lines=purge_zero_speed(lines)

if debug_prt:
    print('first 3 lines of the log file, after zero speed purge', len(lines))
    print(lines[0:3])

# reduce_zero_steering from ket_utility.
#this is to reduce zero steering agnle otherwise dominates, this  
# dominates the driving behavoir, keep only 7%.
lines=reduce_zero_steering(lines, 0.07)
if debug_prt:
    print('number of lines after zero steering', len(lines))

# get_left_rigth_aug reduce_zero_steering from ket_utility.
# this is to include left, right, cameras, as well as to augment
# steering angles greater than +-0.3 (if -, less than). 
# also, this changes the lines format
# lines[0]=path to image (left, right or center individually)
# lines[1]=steering angles
# lines[2]=True or False, True if the  angle and image match
# False, if the image needs flipping in the generator to match the angle
lines=get_left_right_aug(lines)

if debug_prt:
    print('after get left rigth')
    print('number of lines after get_left_right', len(lines), len(angles))
    
#histogram of steering angles after reducing zero degree steering
#and including left, rigth cameras 
#as well as augmenting steering angles which are greater than +-0.3 
if debug_hist:
    angles=[]
    for line in lines:
        angles.append(line[1])
    
    avg_samples_per_bin = len(angles)/num_bins
    hist, bins = np.histogram(angles, num_bins)
    myx=1.0/hist
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.plot((np.min(angles), np.max(angles)), (avg_samples_per_bin, avg_samples_per_bin), 'k-')
    plt.show()

#split the  samples, set aside 20% for validation
train_lines, validation_lines = train_test_split(lines, test_size=0.2)

if debug_prt:
    print('training lines:', len(train_lines), 'valiations lines:', len(validation_lines))



#reads from the lines and returns a batch of file
def generator(input_lines, batch_size):
    num_samples = len(input_lines)
    while 1:
        shuffle(input_lines)
        for offset in range(0, num_samples, batch_size):
            batch_lines = input_lines[offset:offset+batch_size]
            
            images = []
            measurements = []
            for each_batch_line in batch_lines:
                current_path=each_batch_line[0]
                if debug_prt:
                    print('current_path', current_path)
                if each_batch_line[2]:
                    #image and measurement correct in the line
                    image=cv2.imread(current_path)
                    images.append(image)
                    measurement=each_batch_line[1]
                    measurements.append(measurement)
                else:
                    image=cv2.imread(current_path)
                    image=cv2.flip(image,1)
                    images.append(image)
                    #measurement was corrected already
                    measurement=each_batch_line[1]
                    measurements.append(measurement) 
                
                if debug_prt:
                    print('image size', np.size(image, 0), np.size(image,1))
            
            new_images=[]
            for image in images:
                #process_image from ket_utility
                #blurs the image and converts from BGR to RGB
                #cv2 reading is BGR, and RGB provides better processing
                image=process_image(image)
                new_images.append(image)
               
            #yield shuffle

            yield  np.array(new_images), np.array(measurements)

# compile and train the model using the generator function
batch_size=32
train_generator = generator(train_lines, batch_size)
validation_generator = generator(validation_lines, batch_size)

from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.core import Activation
from keras.layers.convolutional import Conv2D
from keras.callbacks import ModelCheckpoint
#now model slight modification to NVIDIA paper

model = Sequential()
model.add(Lambda(lambda x: x/255.0 -0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((40,20),(0,0))))
model.add(Conv2D(3, (1, 1)))
model.add(Conv2D(24, (5, 5), strides=(2,2), activation='elu'))
model.add(Conv2D(36, (5, 5), strides=(2,2), activation='elu'))
model.add(Conv2D(48, (5, 5), strides=(2,2), activation='elu'))
model.add(Conv2D(64, (3, 3), strides=(2,2), activation='elu'))
model.add(Conv2D(64, (3, 3), activation='elu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Activation('elu'))
model.add(Dense(50))
model.add(Activation('elu'))
model.add(Dense(10))
model.add(Activation('elu'))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
steps_per_epoch_=len(train_lines)

checkpointer = ModelCheckpoint(filepath="weights.hdf5", verbose=1, save_best_only=True)
history=model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch_, validation_data=validation_generator, validation_steps=len(validation_lines), verbose=1, epochs=10)
                      
model.save('model_nvidia.h5')
plot_model(model, to_file='model_nvidia.png')

if debug_prt:
    print(model.summary())
