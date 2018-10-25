""" CS 5033 Homework 2
    Code for Problem 4

"""
import os
import numpy as np
from numpy import genfromtxt
from sklearn.cluster import KMeans
import math
from matplotlib import pyplot as plt


# Specify the filepath.
pwd = os.path.dirname(__file__)
filepath = os.path.join(pwd, 'ParkinsonsData/parkinsons_updrs.data')

# K values.
# Running the file with all k-Values will take about 15 minutes to run (on my
# 4 year old pc with dual core i5).
# Consider running with only a few k-values to validate correctness.
kValues = np.array([ 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 50])
# kValues = np.array([ 2, 3, 4, 5, 6])

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

# remove rows allocated as testingData
data = np.delete(data, randomSample, axis=0)

######1.e(iterage b-d 100 times)######
errs = []
for i in range(0,100):
    print(i)
    ######1.b######
    # create training data (3/4 of remaining data)
    # using same methodology as used for creating testing data.
    randomSample = np.random.choice(data.shape[0], size=math.floor(0.75*data.shape[0]), replace=False)
    trainingData = data[randomSample, :]

    # allocate remaining data as validation data(1/4 of remaining data)
    validationData = np.delete(data, randomSample, axis=0)

    err = []
    ######1.c######
    # create models for every K-value.
    # train the models on the predictor and predictand from training data.
    for k in kValues:
        model = KMeans(n_clusters=k)
        model.fit(trainingData)
        mean_UPDRS = [0] * k
        counts = [0] * k
        for i in range(0, len(model.labels_)):
            counts[model.labels_[i]] = counts[model.labels_[i]] + 1
            mean_UPDRS[model.labels_[i]] = mean_UPDRS[model.labels_[i]] + trainingData[i][1]
        mean_UPDRS = [(mean_UPDRS[i]/counts[i]) for i in range(0, len(mean_UPDRS))]
        prediction = model.predict(validationData)

        ######1.d######
        # predict predictant using the models trained on the training data.
        # use validation data predictors to make the predictions.
        # calculate error (predicted - expected)^2
        # then average
        error = [0] * len(prediction)
        for i in range(0, len(prediction)):
            error[i] = (validationData[i][1] - mean_UPDRS[prediction[i]])**2
        err.append(sum(error)/len(error))
    errs.append(err)

######1.e(average mean-sq-err for each combination)######
# add up error for each combo in each of 100 iterations.
# then average
mean_errs = []
for j in range (0, len(kValues)):
    sum_err = 0
    for i in range(0, 100):
        sum_err = sum_err + errs[i][j]
    mean_errs.append(sum_err/100)


######1.f######
fig, ax = plt.subplots()
ax.plot(mean_errs)


# Set the ticks to K-values.
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
for i in range(0, len(mean_errs)):
    if mean_errs[i] < min_err:
        min_err = mean_errs[i]
        kInd = i

print(min_err)
print(kValues[kInd])

# train the model on (training+validation) using the best k-value
model = KMeans(n_clusters=kValues[kInd])
model.fit(data)
mean_UPDRS = [0] * kValues[kInd]
counts = [0] * kValues[kInd]
for i in range(0, len(model.labels_)):
    counts[model.labels_[i]] = counts[model.labels_[i]] + 1
    mean_UPDRS[model.labels_[i]] = mean_UPDRS[model.labels_[i]] + data[i][1]
mean_UPDRS = [(mean_UPDRS[i]/counts[i]) for i in range(0, len(mean_UPDRS))]

######1.h######
# predict using the model trained on (training+validation) data
prediction = model.predict(testingData)
error = [0] * len(prediction)
for i in range(0, len(prediction)):
    error[i] = abs((testingData[i][1] - mean_UPDRS[prediction[i]])**2)
mean_sq_err =  sum(error)/len(error)

print(mean_sq_err)
