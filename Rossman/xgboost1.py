from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import train_test_split

import xgboost as xg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import operator

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


def feature_engineering(features): 
	features.fillna(0, inplace= True)
	#features.loc[features.isOpen.isnull(), 'Open'] =1
	mapper ={'d':4 , 'c':3, 'b':2, 'a':1, '0':'0'}
	features.StoreType.replace(mapper, inplace=True)
	features.Assortment.replace(mapper, inplace=True)
	features.StateHoliday.replace(mapper, inplace=True) 
    	

	features['Year'] = features.Date.dt.year
        features['Month'] = features.Date.dt.month
        features['Day'] = features.Date.dt.day
        features['DayOfTheWeek'] = features.Date.dt.dayofweek
        features['WeekOfTheYear'] = features.Date.dt.weekofyear
	
        val1 = features.Year - features.Promo2SinceYear
	val2 = features.WeekOfTheYear - features.Promo2SinceWeek	
	features['PromoOpen'] = 12 * val1 +  val2 / 4.0
   	features['PromoOpen'] = features.PromoOpen.apply(lambda x: x if x > 0 else 0)
    	features.loc[features.Promo2SinceYear == 0, 'PromoOpen'] = 0
	month_stringval = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', \
             7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}
    	features['monthStr'] = features.Month.map(month_stringval)
    	features.loc[features.PromoInterval == 0, 'PromoInterval'] = ''
    	features['IsPromotionMonth'] = 0#DefaultValue

    	for val in features.PromoInterval.unique():
        	if (val != ''):
            		for mon in val.split(','):
                		features.loc[(features.monthStr == mon) & (features.PromoInterval == val ), 'IsPromotionMonth'] = 1	
	
	val1 = features.Year - features.CompetitionOpenSinceYear
	val2 = features.Month - features.CompetitionOpenSinceMonth
    	features['CompetitionOpen'] = 12 * val1 + val2	
	return features
 
def errorForXgboost(ynew, y):
    y = np.expm1(y.get_label())
    ynew = np.expm1(ynew)
    return "rmspe error", error_value(ynew,y)

def mappingOfFeatures(features):
    outputfinal = open('xg.fmap', 'w')
    for i, data in enumerate(features):
        outputfinal.write('{0}\t{1}\tq\n'.format(i, data))
    outputfinal.close()

rossman = pd.read_csv("train.csv", low_memory= False, parse_dates = [2]) # The training features
store = pd.read_csv("comp.csv", low_memory = False)
rossman_test = pd.read_csv("test.csv", parse_dates = [3]) # Test features. Used only for testing in the end. All feature manipulations done on training features must be done on this also

rossman = rossman[rossman.Open!=0]
rossman = pd.merge(rossman, store, on='Store')
rossman["CompetitionDistance"] = rossman["CompetitionDistance"].fillna(0) # fillna(i) fills empty rows with i
rossman["CompetitionOpenSinceMonth"] = rossman["CompetitionOpenSinceMonth"].fillna(rossman["CompetitionOpenSinceMonth"].mean())
'''
rossman['Year'] = rossman.Date.apply(lambda x:x.split('-')[0])
rossman['Year'] = rossman['Year'].astype(float) # Adding new attributes
rossman['Month'] = rossman.Date.apply(lambda x:x.split('-')[1])
rossman['Month'] = rossman['Month'].astype(float)
'''

rossman_test.fillna(1, inplace= True) # Store open if not provided
rossman_test = pd.merge(rossman_test, store,on= 'Store')
rossman_test["CompetitionDistance"]= rossman_test["CompetitionDistance"].fillna(0)#Fill 0 where empty
rossman_test["CompetitionOpenSinceMonth"] = rossman["CompetitionOpenSinceMonth"].fillna(rossman["CompetitionOpenSinceMonth"].mean())#Fill mean in empty rows

'''
rossman_test['Year'] = rossman_test.Date.apply(lambda x:x.split('-')[0])
rossman_test['Year'] = rossman_test['Year'].astype(float)
rossman_test['Month'] = rossman_test.Date.apply(lambda x:x.split('-')[1])
rossman_test['Month'] = rossman_test['Month'].astype(float)
'''

#sale_means = rossman.groupby('Store').mean().Sales
#sale_means.name = 'Sales_Means'
#rossman = rossman.join(sale_means, on = 'Store')

#print('Rossman', rossman["Sales_Means"])
print(rossman.shape)
fig, axs = plt.subplots(1, 2, sharey=True)
rossman.plot(kind = 'scatter', x = 'SchoolHoliday' , y = 'Sales' , ax = axs[0], figsize = (16,8))
rossman.plot(kind = 'scatter', x = 'Customers' , y ='Sales', ax= axs[1], figsize = (16,8) )
#rossman.plot(kind = 'scatter', x = 'Store' , y = 'Sales' , ax = axs[0], figsize = (16,8))

#plt.show()

features= ['Store','SchoolHoliday','Promo','cmp msr','IsPromotionMonth','Year','Month','Day','DayOfTheWeek','WeekOfTheYear','StoreType','CompetitionOpenSinceMonth','CompetitionDistance','PromoOpen'] # Features used for prediction

feature_engineering(rossman)
feature_engineering(rossman_test)

X = rossman[features]
y = rossman.Sales # The value we are going to predict

boosterVariables = {"objective": "reg:linear", "booster" : "gbtree", "eta": 0.3, "max_depth": 10, "subsample": 0.9, "colsample_bytree": 0.7, "silent": 1, "seed": 1200 } 
booting_rounds = 300


training_features, validation_features = train_test_split(rossman, test_size=0.015, random_state=12)#spliting samples
training_predictions = np.log1p(training_features.Sales)
validation_predictions = np.log1p(validation_features.Sales)
trained_data = xg.DMatrix(training_features[features], training_predictions)
validated_data = xg.DMatrix(validation_features[features], validation_predictions)

listedData = [(trained_data, 'train'), (validated_data, 'eval')]
boosted_method = xg.train(boosterVariables, trained_data, booting_rounds, evals=listedData, early_stopping_rounds=110, feval=errorForXgboost, verbose_eval=True)#To print on each iteration

ynew = boosted_method.predict(xg.DMatrix(validation_features[features]))
error = error_value(np.expm1(ynew),validation_features.Sales.values)

testing_data = xg.DMatrix(rossman_test[features])#retreive testing data
testing_result = boosted_method.predict(testing_data)#compute result
write_output = pd.DataFrame({"Id": rossman_test["Id"] ,"Sales": np.expm1(testing_result)})
write_output.to_csv("sixthSubmission.csv", index = False)#Generating final output

