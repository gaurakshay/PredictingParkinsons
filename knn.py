""" CS 5033 Homework 2
    Code for Problem 2

"""
import os
import numpy as np
from numpy import genfromtxt
from sklearn.neighbors import KNeighborsRegressor
import math
from matplotlib import pyplot as plt

# Specify the filepath.
pwd = os.path.dirname(__file__)
filepath = os.path.join(pwd, 'ParkinsonsData/parkinsons_updrs.data')

# K values.
kValues = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50])

# generate numpy array from the data file.
data = genfromtxt(filepath, delimiter=',', skip_header=1)
######DATA CLEANUP######
# keep only Age, total_UPDRS, Jitter(%), Shimmer, NHR and HNR
data = data[:, [1, 5, 6, 11, 17, 18]]

######1.a######
# create testing dataset (20% of rows)
# choose 20% of the rows from total rows without replacement.
# then slice the dataset using chosen rows.
# create separate entries for the predictor variables and predictand.
randomSample = np.random.choice(data.shape[0], size=math.floor(0.2*data.shape[0]), replace=False)
testingData = data[randomSample, :]
testingData_y = testingData[:, 1]
testingData_x = np.delete(testingData, 1, axis=1)

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
    trainingData_y = trainingData[:, 1]
    trainingData_x = np.delete(trainingData, 1, axis=1)

    # allocate remaining data as validation data(1/4 of remaining data)
    validationData = np.delete(data, randomSample, axis=0)
    validationData_y = validationData[:, 1]
    validationData_x = np.delete(validationData, 1, axis=1)

    ######1.c######
    # create models for every K-value.
    # train the models on the predictor and predictand from training data.
    models = []
    for k in kValues:
        model = KNeighborsRegressor(n_neighbors=k)
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
# add up error for each k-value in each of 100 iterations.
# then average
errs = []
for j in range (0, len(kValues)):
    sum_err = 0
    for i in range(0, 100):
        sum_err = sum_err + iter_err[i][j]
    errs.append(sum_err/100)

print(errs)

######1.f######
fig, ax = plt.subplots()
ax.plot(errs)

# Set the ticks to k-values.
ax.set_xticks(np.arange(len(kValues)), minor=False)

# Labels
ax.set_xticklabels(kValues, minor=False)
ax.set_xlabel('k-Values')
ax.set_ylabel('Mean square error')
plt.title('Mean square error for various k-values')
plt.show()

######1.g######
# find the min error
min_err = 9999999
kInd = 0
for i in range(0, len(errs)):
    if errs[i] < min_err:
        min_err = errs[i]
        kInd = i

print(min_err)
print(kValues[kInd])

# train the model on (training+validation) using the best k-value
model = KNeighborsRegressor(n_neighbors=kValues[kInd])
model.fit(np.delete(data, 1, axis=1), data[:, 1])

######1.h######
# predict using the model trained on (training+validation) data
prediction = model.predict(testingData_x)
error = prediction - testingData_y
sq_error = np.square(error)
sum_sq_error = np.sum(sq_error)
mean_sq_err = sum_sq_error / testingData_y.shape[0]

print(mean_sq_err)
