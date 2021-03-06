# N days of Kera

I'm starting my 'N days of Kera', just trying to challenge myself to learn a bit more everyday
about this framework. At first, I think about doing this for 30 days, but I may continue longer.
I don't know much of Kera (and Python), so forgive me if I do something stupid or not too smart,
or even if it looks like I want to kill an ant with a cannon. I'm just learning, testing and studying :)

So let's start.

## Day 1
### Description
(2017-04-02)
MLP (Multi-Layer perceptron) for Multi-class classification.
I used the Iris dataset and tried to classify it. I configured 80% of the data for training
and 20% for testing. Here is the file: [day1.py](day1.py)

### Results
The MLP got 100% of accuracy on tests when using in a configuration with 4x64x64x8x3 neurons
and 10000 epochs.

## Day 2
### Description
(2017-04-04)
MLP for Regression using Boston Housing dataset.
I used 13x8x8x1 neurons. Here is the file: [day2.py](day2.py) 

## Results
Using RELU activation function, it has gotten a good MSE (3.5 x 10^(-3)), but I think it had some kind of 
overfitting. Later I'll take a deeper look on the dataset and on whether the model is overfit.

## Day 3
### Description
(2017-04-05) MLP for classification using MNIST dataset. I took it from https://github.com/wxs/keras-mnist-tutorial/blob/master/MNIST%20in%20Keras.ipynb .
I changed it and made my own tests to learn this kind of task. Here is the file: [day3.ipynb](day3.ipynb)

## Day 4
### Description
(2017-04-06)
MLP for classification using a dataset of students collected by myself.
I used 16x64x64x8x1 neurons. In a near future I'll use some technique of feature selection
to reduce dimensionality of the input. Here is the file: [day4.py](day4.py)

## Results
I have gotten 93.4% of accuracy on tests.

## Day 5
### Description
(2017-04-09)
CNN (using Conv2D) for classification using the MNIST dataset.
Here is the file: [day5.ipynb](day5.ipynb)
## Results
I have gotten 99.35% of accuracy on tests with the configuration that can be found in the file.
