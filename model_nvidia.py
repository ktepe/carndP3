##import important libs and packages
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from ket_utility import *

#debugging switch
debug_prt=0

## log file is located './folder/driving_log.csv'
## IMG file is located './folder/IMG'
file1_folder='./Data_good_combined/'
file2_folder='./Data_new/'
#
file_folders=[file2_folder]
#MAC, Linux split_char='/', WINDOWS '\\'
split_char='/'
#get all the lines, with local paths
lines=get_lines(file_folders, split_char)
if debug_prt:
    print('number of lines before zero speed', len(lines))

lines=purge_zero_speed(lines)

if debug_prt:
    print('first 3 lines of the log file, after zero speed purge', len(lines))
    print(lines[0:3])

lines=reduce_zero_steering(lines, 0.07)
if debug_prt:
    print('number of lines after zero steering', len(lines))

lines=get_left_right_aug(lines)

print('after get left rigth')

if debug_prt:
    print('number of lines after get_left_right', len(lines), len(angles))
angles=[]
for line in lines:
    angles.append(line[1])
    
num_bins = 25
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

#upto here is fixed


from sklearn.utils import shuffle
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
                image=process_image(image)
                new_images.append(image)
               
            #yield shuffle
            yield  np.array(new_images), np.array(measurements)

# compile and train the model using the generator function
train_generator = generator(train_lines, batch_size=32)
validation_generator = generator(validation_lines, batch_size=32)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.core import Activation
from keras.layers.convolutional import Conv2D


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
samples_per_epoch_=len(train_lines)
history=model.fit_generator(train_generator, samples_per_epoch=samples_per_epoch_, validation_data=validation_generator, validation_steps=len(validation_lines), verbose=1, epochs=5)
                      
model.save('model_nvidia.h5')

print(model.summary())