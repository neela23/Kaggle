from numpy import loadtxt, zeros, ones, array, linspace, logspace, mean, std, arange
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from pylab import plot, show, xlabel, ylabel

import pandas as pd
import numpy as np

#Evaluate the linear regression

def feature_normalize(X):
    '''
    Returns a normalized version of X where
    the mean value of each feature is 0 and the standard deviation
    is 1. This is often a good preprocessing step to do when
    working with learning algorithms.
    '''
    mean_r = []
    std_r = []

    X_norm = X

    n_c = X.shape[1]
    for i in range(n_c):
        m = mean(X[:, i])
        s = std(X[:, i])
        mean_r.append(m)
        std_r.append(s)
        X_norm[:, i] = (X_norm[:, i] - m) / s

    return X_norm, mean_r, std_r


def compute_cost(X, y, theta):
    #Number of training samples
    m = y.size

    predictions = X.dot(theta)

    sqErrors = (predictions - y)

    J = (1.0 / (2 * m)) * sqErrors.T.dot(sqErrors)

    return J


def gradient_descent(X, y, theta, alpha, num_iters):
    '''
    Performs gradient descent to learn theta
    by taking num_items gradient steps with learning
    rate alpha
    '''
    m = y.size
    J_history = zeros(shape=(num_iters, 1))

    for i in range(num_iters):

        predictions = X.dot(theta)

        theta_size = theta.size

        for it in range(theta_size):

            temp = X[:, it]
            temp.shape = (m, 1)

            errors_x1 = (predictions - y) * temp

            theta[it][0] = theta[it][0] - alpha * (1.0 / m) * errors_x1.sum()

        J_history[i, 0] = compute_cost(X, y, theta)

    return theta, J_history

#Load the dataset


def feature_building(data): # Not used yet. Will be used later
	data.fillna(0, inplace= True)
	data.loc[data.isOpen.isnull(), 'Open'] =1
	mappings = {'0':0, 'a':1, 'b':2, 'c':3, 'd':4}
	data.StoreType.replace(mappings, inplace=True)
	data.Assortment.replace(mappings, inplace=True)
	data.StateHoliday.replace(mappings, inplace=True)

rossman = pd.read_csv("train.csv", low_memory= False) # The training data
store = pd.read_csv("comp.csv", low_memory = False)
rossman_test = pd.read_csv("test.csv") # Test data. Used only for testing in the end. All feature manipulations done on training data must be done on this also

rossman = rossman[rossman.Open!=0]
rossman = pd.merge(rossman, store, on='Store')
rossman["CompetitionDistance"] = rossman["CompetitionDistance"].fillna(0) # fillna(i) fills empty rows with i
rossman["CompetitionOpenSinceMonth"] = rossman["CompetitionOpenSinceMonth"].fillna(rossman["CompetitionOpenSinceMonth"].mean())

rossman['Year'] = rossman.Date.apply(lambda x:x.split('-')[0])
rossman['Year'] = rossman['Year'].astype(float) # Adding new attributes
rossman['Month'] = rossman.Date.apply(lambda x:x.split('-')[1])
rossman['Month'] = rossman['Month'].astype(float)


rossman_test.fillna(1, inplace= True) # Store open if not provided
rossman_test = pd.merge(rossman_test, store,on= 'Store')
rossman_test["CompetitionDistance"]= rossman_test["CompetitionDistance"].fillna(0)
rossman_test["CompetitionOpenSinceMonth"] = rossman["CompetitionOpenSinceMonth"].fillna(rossman["CompetitionOpenSinceMonth"].mean())

rossman_test['Year'] = rossman_test.Date.apply(lambda x:x.split('-')[0])
rossman_test['Year'] = rossman_test['Year'].astype(float)
rossman_test['Month'] = rossman_test.Date.apply(lambda x:x.split('-')[1])
rossman_test['Month'] = rossman_test['Month'].astype(float)


#sale_means = rossman.groupby('Store').mean().Sales
#sale_means.name = 'Sales_Means'

#feature_building(rossman)
#feature_buliding(rossman_test)
#rossman = rossman.join(sale_means, on = 'Store')

#print('Rossman', rossman["Sales_Means"])
print(rossman.shape)
#fig, axs = plt.subplots(1, 2, sharey=True)
#rossman.plot(kind = 'scatter', x = 'SchoolHoliday' , y = 'Sales' , ax = axs[0], figsize = (16,8))
#rossman.plot(kind = 'scatter', x = 'Customers' , y ='Sales', ax= axs[1], figsize = (16,8) )
#rossman.plot(kind = 'scatter', x = 'Store' , y = 'Sales' , ax = axs[0], figsize = (16,8))

#plt.show()

features= ['Store','SchoolHoliday' ,'CompetitionDistance','Year','Month','Promo','cmp msr'] # Features used for prediction
X = rossman[features]
print 'X = ', X
y = rossman.Sales # The value we are going to predict




# number of training samples
#m = y.size

#y.shape = (m, 1)

#Scale features and set them to zero mean
x, mean_r, std_r = feature_normalize(X)

#Add a column of ones to X (interception data)
it = ones(shape=(m, 3))
it[:, 1:3] = x

#Some gradient descent settings
iterations = 100
alpha = 0.01

#Init Theta and Run Gradient Descent
theta = zeros(shape=(3, 1))

theta, J_history = gradient_descent(it, y, theta, alpha, iterations)
print theta, J_history
plot(arange(iterations), J_history)
xlabel('Iterations')
ylabel('Cost Function')
show()

#Predict price of a 1650 sq-ft 3 br house
#price = array([1.0,   ((1650.0 - mean_r[0]) / std_r[0]), ((3 - mean_r[1]) / std_r[1])]).dot(theta)
#print 'Predicted price of a 1650 sq-ft, 3 br house: %f' % (price)
