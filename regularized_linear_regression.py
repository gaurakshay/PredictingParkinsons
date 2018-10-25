""" CS 5033 Homework 2
    Code for Problem 1

"""
import os
import numpy as np
from numpy import genfromtxt
from sklearn.linear_model import ElasticNet
import math
from matplotlib import pyplot as plt

# Specify the filepath.
pwd = os.path.dirname(__file__)
filepath = os.path.join(pwd, 'ParkinsonsData/parkinsons_updrs.data')

# Alpha values.
alphaValues = np.array([0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0])

# Lambda / l1_ratio values.
lambdaValues = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

# generate numpy array from the data file.
data = genfromtxt(filepath, delimiter=',', skip_header=1)
######DATA CLEANUP######
# remove subject attribute
data = np.delete(data, 0, axis=1)
# remove gender attribute
data = np.delete(data, 1, axis=1)
# remove motor_UPDRS attribute
data = np.delete(data, 2, axis=1)

######1.a######
# create testing dataset (20% of rows)
# choose 20% of the rows from total rows without replacement.
# then slice the dataset using chosen rows.
# create separate entries for the predictor variables and predictand.
randomSample = np.random.choice(data.shape[0], size=math.floor(0.2*data.shape[0]), replace=False)
testingData = data[randomSample, :]
testingData_y = testingData[:, 2]
testingData_x = np.delete(testingData, 2, axis=1)

# remove rows allocated as testingData
data = np.delete(data, randomSample, axis=0)

######1.e(iterage b-d 100 times)######
iter_err = []
for i in range(0, 100):
    ######1.b######
    # create training data (3/4 of remaining data)
    # using same methodology as used for creating testing data.
    randomSample = np.random.choice(data.shape[0], size=math.floor(0.75*data.shape[0]), replace=False)
    trainingData = data[randomSample, :]
    trainingData_y = trainingData[:, 2]
    trainingData_x = np.delete(trainingData, 2, axis=1)

    # allocate remaining data as validation data(1/4 of remaining data)
    validationData = np.delete(data, randomSample, axis=0)
    validationData_y = validationData[:, 2]
    validationData_x = np.delete(validationData, 2, axis=1)

    ######1.c######
    # create models for every combo of alpha and lambda values.
    # train the models on the predictor and predictand from training data.
    models = []
    for alphaa in alphaValues:
        for lambdaa in lambdaValues:
            model = ElasticNet(alpha=alphaa, l1_ratio=lambdaa)
            model.fit(trainingData_x, trainingData_y)
            models.append(model)

    ######1.d######
    # predict predictant using the models trained on the training data.
    # use validation data predictors to make the predictions.
    # calculate error (predicted - expected)^2
    # then average
    model_err = []
    for model in models:
        prediction = model.predict(validationData_x)
        error = prediction - validationData_y
        sq_error = np.square(error)
        sum_sq_error = np.sum(sq_error)
        mean_sq_err = sum_sq_error / validationData_y.shape[0]
        model_err.append(mean_sq_err)
    iter_err.append(model_err)

######1.e(average mean-sq-err for each combination)######
# add up error for each combo in each of 100 iterations.
# then average
errs = []
for j in range (0, 110):
    sum_er = 0
    for i in range(0, 100):
        sum_er = sum_er + iter_err[i][j]
    errs.append(sum_er/100)

# Rearrange the 1-D array to 2-D array to help plotting.
arr = []
for i in range(0, len(alphaValues)):
    arr.append([])
    for j in range(0, len(lambdaValues)):
        arr[i].append(errs[i*len(alphaValues) + j])


######1.f######
fig, ax = plt.subplots()
heatmap = ax.pcolor(arr)
cbar = plt.colorbar(heatmap)
cbar.set_label('Mean Square Error', rotation=90)

# Set the ticks to alpha and lambda values.
ax.set_xticks(np.arange(len(lambdaValues)) + 0.5, minor=False)
ax.set_yticks(np.arange(len(alphaValues)) + 0.5, minor=False)

# Labels
ax.set_xticklabels(lambdaValues, minor=False)
ax.set_yticklabels(alphaValues, minor=False)
ax.set_xlabel('Lambda values')
ax.set_ylabel('Alpha values')
plt.title('Mean square error for alpha and lambda')
plt.show()

######1.g######
# find the min error
min_err = 9999999
alphaInd = 0
lambdaInd = 0
for i in range(0, len(alphaValues)):
    for j in range(0, len(lambdaValues)):
        if arr[i][j] < min_err:
            min_err = arr[i][j]
            alphaInd = i
            lambdaInd = j

print(min_err)
print(alphaValues[alphaInd], lambdaValues[lambdaInd])

# train the model on (training+validation) using the best combo
model = ElasticNet(alpha=alphaValues[alphaInd], l1_ratio=lambdaValues[lambdaInd])
model.fit(np.delete(data, 2, axis=1), data[:, 2])

######1.h######
# predict using the model trained on (training+validation) data
prediction = model.predict(testingData_x)
error = prediction - testingData_y
sq_error = np.square(error)
sum_sq_error = np.sum(sq_error)
mean_sq_err = sum_sq_error / testingData_y.shape[0]

print(mean_sq_err)
