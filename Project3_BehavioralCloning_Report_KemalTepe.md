# **Behavioral Cloning** 

## Kemal Tepe, ketepe@gmail.common

### Objective: To build automous driving model from a simulated driving scenarios using Convolutional Neural Networks (CNN).

### Summary:

The idea is to teach a vehicle controller how to drive from actual driving conditions. In this project, training data, images from three different cameras from a simulated vehicle are collected along with steering angles. With this training data, a CNN network is trained and a model is generated using Tensorflow backend with Keras frontend. NVIDIA's published CNN network is sligthly modified to generate the model. The generated model by [model_nvidia.py](./model_nvidia.py) is [model_nvidia.h5](./model_nvidia.h5). The model is used in the simulator and a video clip of the run can be viewed by using movie file [run1.mp4](./run1.mp4), which displays nearly 1.5 lap on the circular course.  


### The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

The entire code can be obtained using [here](./model_nvidia.py). The video file is downloaded from [My public dropbox folder](dropbox). Now we will go to important parts of the steps of the project


#### 1. Use the simulator to collect data of good driving behavior
It is key to get used to the simulator. I have used MAC Bookair to collect data since my laptop (Lenovo Helix) did not have dedicated graphic cars to run the simulation efficiently. I followed instructions provided by the Udacity team: 3-laps in counterclockwise direction, 2-laps in counterclockwise direction, 6-8 sharp turn sections, 10-15 recovering from side to center.

After that, the data needs to be processes, and ugmented which I will talk about in the following sections. However, collecting useful data by mimicing an effective driving is key to clone the behavior.

### 2. Build, a convolution neural network in Keras that predicts steering angles from images

Before moving to the final model, which is derived from [NVIDIA paper](./nvidia_model.pdf) cited in the project site.  The model used is given in Table 1.


|Table 1: Architecture | | |
|---------|--------|--------|
|Layer | Description | Parameters |
|Layer 1| Lambda| normalization, input=160x320x3 | 
|Layer 2| Cropping2D| crop rows, top=40, bottom=20, new size=100x320x3 | 
|Layer 3| CNN 24x5x5 | ELU activiation |
|Layer 4| CNN 36x5x5 | ELU activation|
|Layer 5| CNN 48x5x5 | ELU activation|
|Layer 6| CNN 64x3x3 | ELU activation|
|Layer 7| CNN 64x3x3 | ELU activation|
|Layer 8| Flatten ||
|Layer 9| Dense |100, ELU activation|
|Layer 9| Dense |50, ELU activation|
|Layer 10| Dense |10, ELU activation|
|Layer 11| Dense |1|

The model of the code is given below too.

```python 
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
steps_per_epoch_=floor(len(train_lines)/batch_size)
history=model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch_, validation_data=validation_generator, validation_steps=len(validation_lines), verbose=1, epochs=5)
model.save('model_nvidia.h5')
```



[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

#### 3. Train and validate the model with a training and validation set

#### 4. Test that the model successfully drives around track one without leaving the road


#### 5. Summarize the results with a written report



###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
