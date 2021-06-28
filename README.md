# TAU (Tampere University) Vehicle Type Recognition

## Task Description
Categorize images, as it is a classical machine learning task, which has been a key driver in machine learning research since the birth of deep neural networks. The goal is to classify images of different vehicle types (total 17 categories); For instance, cars, bicycles, boats, vans, etc. 

The data that I used has been collected from the [Open Images Dataset](https://storage.googleapis.com/openimages/web/index.html); an annotated collection of over 9 million images. I used a subset of open images, selected to contain only vehicle categories among the total of 600 object classes.

The task will consist on 2 mandatory steps:
1. Training a sklearn model with CNN feature extractor.
2. Training a CNN

## Code Structure and Explaination
In this assignment, a pretrained convolutional neural network (CNN) is being used to extract the features of the images. The CNN used were Inception_V3, Xception and MobileNet.

I used some trick to increase the performance of the CNN since the dataset is skewed. As a result, I gave more weight to small classes. Those classes were Tank, Barge, Segway, Ambulance, Snowmobile, Limousine and Cart. I also added augmentation by flipping the images.

Each input image is resize and normalize before rotating and add into training data. In addition, the labels are converted to one-hot vector. By using the train_test_split, I divided the dataset to 80 for training and 20 for testing.

## Result
| Model Name   | Public Score | Private Score |
| ----------   | ------------ | ------------- |
| MobileNet    | 0.82547      | 0.81338       |
| MobileNet_V2 | 0.80591      | 0.79091       |
| Inception_V3 | 0.87963      | 0.87642       |
| Xception     | 0.89518      | 0.88430       |