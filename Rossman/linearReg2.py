from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from operator import attrgetter

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

def performTimeSeriesCV(X_train, y_train, number_folds, algorithm, parameters):
    """
    Given X_train and y_train (the test set is excluded from the Cross Validation),
    number of folds, the ML algorithm to implement and the parameters to test,
    the function acts based on the following logic: it splits X_train and y_train in a
    number of folds equal to number_folds. Then train on one fold and tests accuracy
    on the consecutive as follows:
    - Train on fold 1, test on 2
    - Train on fold 1-2, test on 3
    - Train on fold 1-2-3, test on 4
    ....
    Returns mean of test accuracies.
    """
 
    print 'Parameters --------------------------------> ', parameters
    print 'Size train set: ', X_train.shape
    
    # k is the size of each fold. It is computed dividing the number of 
    # rows in X_train by number_folds. This number is floored and coerced to int
    k = int(np.floor(float(X_train.shape[0]) / number_folds))
    print 'Size of each fold: ', k
    
    # initialize to zero the accuracies array. It is important to stress that
    # in the CV of Time Series if I have n folds I test n-1 folds as the first
    # one is always needed to train
    accuracies = np.zeros(folds-1)
 
    # loop from the first 2 folds to the total number of folds    
    for i in range(2, number_folds + 1):
        print ''
        
        # the split is the percentage at which to split the folds into train
        # and test. For example when i = 2 we are taking the first 2 folds out 
        # of the total available. In this specific case we have to split the
        # two of them in half (train on the first, test on the second), 
        # so split = 1/2 = 0.5 = 50%. When i = 3 we are taking the first 3 folds 
        # out of the total available, meaning that we have to split the three of them
        # in two at split = 2/3 = 0.66 = 66% (train on the first 2 and test on the
        # following)
        split = float(i-1)/i
        
        # example with i = 4 (first 4 folds):
        #      Splitting the first       4        chunks at          3      /        4
        print 'Splitting the first ' + str(i) + ' chunks at ' + str(i-1) + '/' + str(i) 
        
        # as we loop over the folds X and y are updated and increase in size.
        # This is the data that is going to be split and it increases in size 
        # in the loop as we account for more folds. If k = 300, with i starting from 2
        # the result is the following in the loop
        # i = 2
        # X = X_train[:(600)]
        # y = y_train[:(600)]
        #
        # i = 3
        # X = X_train[:(900)]
        # y = y_train[:(900)]
        # .... 
        X = X_train[:(k*i)]
        y = y_train[:(k*i)]
        print 'Size of train + test: ', X.shape # the size of the dataframe is going to be k*i
 
        # X and y contain both the folds to train and the fold to test.
        # index is the integer telling us where to split, according to the
        # split percentage we have set above
        index = int(np.floor(X.shape[0] * split))
        
        # folds used to train the model        
        X_trainFolds = X[:index]        
        y_trainFolds = y[:index]
        
        # fold used to test the model
        X_testFold = X[(index + 1):]
        y_testFold = y[(index + 1):]
        
        # i starts from 2 so the zeroth element in accuracies array is i-2. performClassification() is a function which takes care of a classification problem. This is only an example and you can replace this function with whatever ML approach you need.
        accuracies[i-2] = performClassification(X_trainFolds, y_trainFolds, X_testFolds, y_testFolds, algorithm, parameters)
        
        # example with i = 4:
        #      Accuracy on fold         4     :    0.85423
        print 'Accuracy on fold ' + str(i) + ': ', acc[i-2]
    
    # the function returns the mean of the accuracy on the n-1 folds    
    return accuracies.mean()

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
fig, axs = plt.subplots(1, 2, sharey=True)
rossman.plot(kind = 'scatter', x = 'SchoolHoliday' , y = 'Sales' , ax = axs[0], figsize = (16,8))
rossman.plot(kind = 'scatter', x = 'Customers' , y ='Sales', ax= axs[1], figsize = (16,8) )
#rossman.plot(kind = 'scatter', x = 'Store' , y = 'Sales' , ax = axs[0], figsize = (16,8))

#plt.show()

features= ['Store','SchoolHoliday' ,'CompetitionDistance','Year','Month','Promo','cmp msr'] # Features used for prediction
X = rossman[features]
print 'X = ', X
y = rossman.Sales # The value we are going to predict

#alg = LinearRegression()
#alg.fit(X, y) # Algorithm used all features to predict y
print alg.intercept_
print alg.coef_

alg = PolynomialFeatures(degree=2)
alg.fir_transform(X,y)
#print(alg.predict([5,0]))
#print(alg.score(X,y))
kf = KFold(rossman.shape[0], n_folds=3, random_state=1) #Dividing data into three sets - training, test and crossvalidation
#skf = StratifiedKFold(rossman["Sales"], 3)
predictions = []
#sales_mean = train.groupby('Store').mean().Sales

#rossman = sorted(rossman, key= lambda year:attrgetter(rossman['Year']))
print('Ordered?', rossman)


'''
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
'''

error = rmspe(predictions, rossman.Sales)
print('**********E*********rror', error)


testPredictions = alg.predict(rossman_test[features])

submission = pd.DataFrame({"Id": rossman_test["Id"] ,"Sales": testPredictions})
submission.to_csv("secondSub.csv", index = False)
	
