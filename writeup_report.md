
## Data Collection



## Data Augmentation

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



## Final Results
The model was used to drive the car around the same track that was used to collect training data. 
The results can be seen in the video in the link below: 

https://www.youtube.com/watch?v=FK6Asmhwtbw&feature=youtu.be
