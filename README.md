# N days of Kera

I'm starting my 'N days of Kera', just trying to challenge myself to learn a bit more everyday
about this framework. At first, I think about doing this for 30 days, but I may continue longer.
I don't know much of Kera (and Python), so forgive me if I do something stupid or not too smart,
or even if it looks like I want to kill an ant with a cannon.

So let's start.

## Day 1
### Description
(2017-04-02)
MLP (Multi-Layer perceptron) for Multi-class classification.
I used the Iris dataset and tried to classify it. I configured 80% of the data for training
and 20% for testing.

### Results
The MLP got 100% of accuracy on tests when using in a configuration with 4x64x64x8x3 neurons
and 10000 epochs.

## Day 2
### Description
(2017-04-04)
MLP for Regression using Boston Housing dataset.
I used 13x8x8x1. 

## Results
Using RELU activation function, it has gotten a good MSE (3.5 x 10^(-3)), but I think it had some kind of 
overfitting. Later I'll take a deeper look on the dataset and on whether the model is overfit.