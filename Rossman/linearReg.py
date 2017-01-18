from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def error_weight(yval):
	weight = np.zeros(yval.shape, dtype = float)
	index = yval !=0
	weight[index] = 1./(yval[index]**2)
	return weight

def error_Calc(ynew, yval):
	w = error_weight(yval)
	mean = np.mean(w *(yval/ynew-1) ** 2)
	error_value = np.sqrt(mean)
	#error_value = np.sqrt(np.mean( w * (y-ynew)**2))
	return error_value

def error_value(ynew, y):  # The new error function
	mean = np.mean((y/ynew-1) ** 2)
	return np.sqrt(mean)	

def feature_engineering(data): 
	data.loc[data.isOpen.isnull(), 'Open'] =1
	mapper = {'0':0, 'a':1, 'b':2, 'c':3, 'd':4} # Changing character values to integers. This is much needed.
	data.StoreType.replace(mapper, inplace=True)
	data.Assortment.replace(mapper, inplace=True)
	data.StateHoliday.replace(mapper, inplace=True)

rossman = pd.read_csv("train.csv", low_memory= False) # The training data
store = pd.read_csv("comp.csv", low_memory = False) #The store data that will be used
rossman_test = pd.read_csv("test.csv") # Test data. Used only for testing in the end. All feature manipulations done on training data must be done on this also

rossman = rossman[rossman.Open!=0]
rossman = pd.merge(rossman, store, on='Store')
rossman["CompetitionDistance"] = rossman["CompetitionDistance"].fillna(0) # fillna(i) fills empty rows with i
rossman["CompetitionOpenSinceMonth"] = rossman["CompetitionOpenSinceMonth"].fillna(rossman["CompetitionOpenSinceMonth"].mean())

rossman['Year'] = rossman.Date.apply(lambda x:x.split('-')[0])#Splitting Year from date
rossman['Year'] = rossman['Year'].astype(float)#Adding new attribute as float
rossman['Month'] = rossman.Date.apply(lambda x:x.split('-')[1])#Splitting month from date
rossman['Month'] = rossman['Month'].astype(float)


rossman_test.fillna(1, inplace= True) # Store is assumeed to be open if not provided
rossman_test = pd.merge(rossman_test, store,on= 'Store')#Store is another set of data that is required to be used while training
rossman_test["CompetitionDistance"] = rossman_test["CompetitionDistance"].fillna(0)
rossman_test["CompetitionOpenSinceMonth"] = rossman["CompetitionOpenSinceMonth"].fillna(rossman["CompetitionOpenSinceMonth"].mean()) #Filling competetionOpen data with mean of other datas

rossman_test['Year'] = rossman_test.Date.apply(lambda x:x.split('-')[0])
rossman_test['Year'] = rossman_test['Year'].astype(float)
rossman_test['Month'] = rossman_test.Date.apply(lambda x:x.split('-')[1])
rossman_test['Month'] = rossman_test['Month'].astype(float)


#sale_means = rossman.groupby('Store').mean().Sales
#sale_means.name = 'Sales_Means'

#feature_engineering(rossman)
#feature_buliding(rossman_test)
#rossman = rossman.join(sale_means, on = 'Store')

#print('Rossman', rossman["Sales_Means"])

#fig, axs = plt.subplots(1, 2, sharey=True)
#rossman.plot(kind = 'scatter', x = 'SchoolHoliday' , y = 'Sales' , ax = axs[0], figsize = (16,8))
#rossman.plot(kind = 'scatter', x = 'Customers' , y ='Sales', ax= axs[1], figsize = (16,8) )
#rossman.plot(kind = 'scatter', x = 'Store' , y = 'Sales' , ax = axs[0], figsize = (16,8))

#plt.show()

features=  ['Store','SchoolHoliday' ,'CompetitionDistance','Year','Month','Promo','cmp msr'] # Features used for prediction
X = rossman[features]
print 'X = ', X
y = rossman.Sales # The value we are going to predict

alg = LinearRegression()
alg.fit(X, y) # Algorithm used all features to predict y
print alg.intercept_
print alg.coef_

#print(alg.predict([5,0]))
#print(alg.score(X,y))
kf = KFold(rossman.shape[0], n_folds=3, random_state=1) #Dividing data into three sets - training, test and crossvalidation
#skf = StratifiedKFold(rossman["Sales"], 3)
predicted_values = []
#sales_mean = train.groupby('Store').mean().Sales


for train, test in kf:
	train_predictors = (rossman[features].iloc[train,:]) # The entire row will be used for prediction
	train_target = rossman["Sales"].iloc[train] # The target used to train algorithm
	alg.fit(train_predictors, train_target) # Train algo to fit
	test_predicted_values = alg.predict(rossman[features].iloc[test,:])
	predicted_values.append(test_predicted_values)

predicted_values = np.fix(predicted_values)
predicted_values = np.concatenate(predicted_values, axis=0)
#print(np.count_nonzero(predicted_values))

error = error_value(predicted_values, rossman.Sales)
print('Error', error)


testPredictions = alg.predict(rossman_test[features])

#plt.scatter(rossman.Sales, predicted_values, color='black')
#plt.plot(rossman.Sales,predicted_values,color='blue', linewidth=3)
#plt.xticks(())
#plt.yticks(())

#plt.show()
submission = pd.DataFrame({"Id": rossman_test["Id"] ,"Sales": testPredictions})
submission.to_csv("secondSub.csv", index = False)
	
