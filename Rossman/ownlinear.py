from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def ToWeight(y):
	w = np.zeros(y.shape, dtype = float)
	ind = y !=0
	w[ind] = 1./(y[ind]**2)
	return w

def error_Calc(ythat, y):
	w = ToWeight(y)
	rmspe = np.sqrt(np.mean(w *(y/ythat-1) ** 2))
	#rmspe = np.sqrt(np.mean( w * (y-ythat)**2))
	return rmspe

def rmspe(ythat, y):  # The new error function
	return np.sqrt(np.mean((y/ythat-1) ** 2))	

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


def gradient_descent(X, y, theta, alpha, num_iters):
    '''
    Performs gradient descent to learn theta
    by taking num_items gradient steps with learning
    rate alpha
    '''
    m = y.size
    J_history = np.zeros(shape=(num_iters, 1))

    for i in range(num_iters):

        predictions = X.dot(theta)

        theta_size = theta.size

        for it in range(theta_size):

            temp = X[:, it]
            temp.shape = (m, 1)

            errors_x1 = (predictions - y) * temp

            theta[it][0] = theta[it][0] - alpha * (1.0 / m) * errors_x1.sum()

        J_history[i, 0] = rmpse(X, y)

    return theta, J_history

iterations=100
alpha = 0.01
theta = np.zeros(shape=(7,844392))
theta, J_history = gradient_descent(X, y, theta, alpha, iterations)
print theta, J_history











'''
alg = LinearRegression()
alg.fit(X, y) # Algorithm used all features to predict y
print alg.intercept_
print alg.coef_

#print(alg.predict([5,0]))
#print(alg.score(X,y))
kf = KFold(rossman.shape[0], n_folds=3, random_state=1) #Dividing data into three sets - training, test and crossvalidation
#skf = StratifiedKFold(rossman["Sales"], 3)
predictions = []
#sales_mean = train.groupby('Store').mean().Sales


for train, test in kf:
	print 'Train', train
	print 'Test', test
	train_predictors = (rossman[features].iloc[train,:]) # The entire row will be used for prediction
	train_target = rossman["Sales"].iloc[train] # The target used to train algorithm
	alg.fit(train_predictors, train_target) # Train algo to fit
	test_predictions = alg.predict(rossman[features].iloc[test,:])
	predictions.append(test_predictions)

predictions = np.fix(predictions)
predictions = np.concatenate(predictions, axis=0)
#print(np.count_nonzero(predictions))

error = rmspe(predictions, rossman.Sales)
print('**********E*********rror', error)


testPredictions = alg.predict(rossman_test[features])

#plt.scatter(rossman.Sales, predictions, color='black')
#plt.plot(rossman.Sales,predictions,color='blue', linewidth=3)
#plt.xticks(())
#plt.yticks(())

#plt.show()
'''
submission = pd.DataFrame({"Id": rossman_test["Id"] ,"Sales": testPredictions})
submission.to_csv("secondSub.csv", index = False)
	
