# SAMME - Multiclass AdaBoost (Under Development)

@author: Jia Geng

@email: jxg570@miami.edu

@website: https://www.linkedin.com/in/jia-geng/

## Introduction

SAMME [1] is a multi-class adaboost algorithm. 

## To use:

```
# set up parameters, e.g 500 weak learners and 5 classes
num_learner = 500
num_cats = 5

# prepare your data. need to be in the following format
# if your data are image and the image arrays are too larger to be put into list
# you can modify the src code so that it takes img file id and make it read the image file online
train_data = [(X, label), (X, label), (X, label), ...] 

# put your weak learner into a list. your weak learner need to have a `.predict(X) -> label` method
# if not, you need to modify the src code to make it compatible
weak_learners = [wl1, wl2, ...]  

# construct the booster instance
booster = SAMME(num_learner, num_cats)

# train the booster
booster.train(train_data, weak_learners)

# make prediction
prediction = booster.predict(X)
```

## Dev Log

11/02/19: Implement the main methods. Still need to test.

11/03/19: Added a small number to prevent possible divide by zero case

11/05/19: Tested.


## Reference

[1] Zhu, Ji & Rosset, Saharon & Zou, Hui & Hastie, Trevor. (2006). Multi-class AdaBoost. _Statistics and its interface._ 2. 10.4310/SII.2009.v2.n3.a8.
