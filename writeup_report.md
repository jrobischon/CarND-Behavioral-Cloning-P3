# Behavioral Cloning

## Description

The purpose of this project is to train a deep learning model to drive a car around a track in a simulated environment.  The driving simulator, provided by Udacity, allows a user to manually steer a car around a track.  The simulator records intermittent images from 3 virtual cameras (center, left, right) as well as corresponding steering angles.  

After using the simulator to collect a sufficient amount of data, a model is trained to predict the appropriate steering angle for an input image.  This performance of this model is then tested by using the output of the model to automatically control the car as it drives around the same track that was used for training.


## Data Collection

The car was steered manually around the track for several laps in both the clockwise and counterclockwise directions. In addition, several instances were recorded in which the car was steered back to center from the lane edges.  


## Data Augmentation

The center camera data was further expanded with the images captured using the left and right cameras.  An angle offset was added to the steering angle associated with the left camera (+0.1 degree) and right camera (-0.1 degree) camera images to account for the offsets from the car center line.  

Also, each all images were flipped across the vertical axis.  Corresponding steering angles were adjusted by multiplying by -1.  The final set of images consists of 6x more images than those taken from the center camera.

Center Image

![img0](https://github.com/jrobischon/CarND-Behavioral-Cloning-P3/blob/master/img_center1.jpg)

Center Image (Flipped)

![img1](https://github.com/jrobischon/CarND-Behavioral-Cloning-P3/blob/master/img_center1_flip.jpg)

Left Image

![img2](https://github.com/jrobischon/CarND-Behavioral-Cloning-P3/blob/master/img_left1.jpg)


Right Image

![img3](https://github.com/jrobischon/CarND-Behavioral-Cloning-P3/blob/master/img_right1.jpg)


## Model Architecture
The NVIDIA behavioral cloning model archtitecture was replicated in Keras with normalization and cropping layers.
Details are provided below:

Input Shape = (160, 320, 3)

Layer 1: Normalization

Layer 2: Cropping

Layer 3: Convolution (output depth = 24, stride = 2, kernel = (5,5)) w/ relu activation

Layer 4: Convolution (output depth = 36, stride = 2, kernel = (5,5)) w/ relu activation

Layer 5: Convolution (output depth = 48, stride = 2, kernel = (5,5)) w/ relu activation

Layer 6: Convolution (output depth = 64, stride = 1, kernel = (3,3)) w/ relu activation

Layer 7: Convolution (output depth = 64, stride = 1, kernel = (3,3)) w/ relu activation

Layer 8: Fully Connected (output depth = 100)

Layer 9: Fully Connected (output depth = 50)

Layer 10: Fully Connected (output depth = 10)

Layer 11: Fully Connected (output depth = 1)


## Model Training
The model was trained using the ADAM optimizer to minimize mean squared error (MSE).  A batch size of 32 was used for 5 epochs.

A variety of offset angles were tested to adjust for the orientations of the left and right cameras.  The final offset of 0.1 degrees was chosen based on visually assessing the response of the car as it reaches the edges of the track.  Very large angle offsets result in extreme overcorrections as the car nears the lane edge.  Conversely, if the offset is too small then the car will be unlikely to return to the center of the track.


## Final Results
The model was used to drive the car around the same track that was used to collect training data. 
The results can be seen in the video in the link below: 

https://www.youtube.com/watch?v=FK6Asmhwtbw&feature=youtu.be
