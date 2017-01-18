from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge

import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import operator

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
	#data.loc[data.isOpen.isnull(), 'Open'] =1
	mappings = {'0':0, 'a':1, 'b':2, 'c':3, 'd':4}
	data.StoreType.replace(mappings, inplace=True)
	data.Assortment.replace(mappings, inplace=True)
	data.StateHoliday.replace(mappings, inplace=True)
	
	# Indicate that sales on that day are in promo interval
    	
	data['Year'] = data.Date.dt.year
        data['Month'] = data.Date.dt.month
        data['Day'] = data.Date.dt.day
        data['DayOfWeek'] = data.Date.dt.dayofweek
        data['WeekOfYear'] = data.Date.dt.weekofyear
	
    	month2str = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', \
             7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}
    	data['monthStr'] = data.Month.map(month2str)
    	data.loc[data.PromoInterval == 0, 'PromoInterval'] = ''
    	data['IsPromoMonth'] = 0
    	for interval in data.PromoInterval.unique():
        	if interval != '':
            		for month in interval.split(','):
                		data.loc[(data.monthStr == month) & (data.PromoInterval == interval), 'IsPromoMonth'] = 1	
	
    	# CompetionOpen en PromoOpen from https://www.kaggle.com/ananya77041/rossmann-store-sales/randomforestpython/code
    	# Calculate time competition open time in months
    	#features.append('CompetitionOpen')
    	data['CompetitionOpen'] = 12 * (data.Year - data.CompetitionOpenSinceYear) + \
        (data.Month - data.CompetitionOpenSinceMonth)	
	return data
 
def rmspe_xg(yhat, y):
    y = np.expm1(y.get_label())
    yhat = np.expm1(yhat)
    return "rmspe", rmspe(yhat,y)

def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()

rossman = pd.read_csv("train.csv", low_memory= False, parse_dates = [2]) # The training data
store = pd.read_csv("comp.csv", low_memory = False)
rossman_test = pd.read_csv("test.csv", parse_dates = [3]) # Test data. Used only for testing in the end. All feature manipulations done on training data must be done on this also

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
rossman_test["CompetitionDistance"]= rossman_test["CompetitionDistance"].fillna(0)
rossman_test["CompetitionOpenSinceMonth"] = rossman["CompetitionOpenSinceMonth"].fillna(rossman["CompetitionOpenSinceMonth"].mean())

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

features= ['Store','SchoolHoliday','Promo','cmp msr','IsPromoMonth','Year','Month','Day','DayOfWeek','WeekOfYear','StoreType','Assortment'] # Features used for prediction

feature_building(rossman)
feature_building(rossman_test)

X = rossman[features]
print 'X = ', X
y = rossman.Sales # The value we are going to predict

alg = make_pipeline(PolynomialFeatures(degree=2), Ridge())
alg.fit_transform(X,y)

kf = KFold(rossman.shape[0], n_folds=3, random_state=1)

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

error = rmspe(predictions, rossman.Sales)
print('**********E*********rror', error)


testPredictions = alg.predict(rossman_test[features])

submission = pd.DataFrame({"Id": rossman_test["Id"] ,"Sales": testPredictions})
submission.to_csv("polynomialResult.csv", index = False)
'''
params = {"objective": "reg:linear",
          "booster" : "gbtree",
          "eta": 0.3,
          "max_depth": 10,
          "subsample": 0.9,
          "colsample_bytree": 0.7,
          "silent": 1,
          "seed": 1301
          }
num_boost_round = 300


X_train, X_valid = train_test_split(rossman, test_size=0.012, random_state=10)
y_train = np.log1p(X_train.Sales)
y_valid = np.log1p(X_valid.Sales)
dtrain = xgb.DMatrix(X_train[features], y_train)
dvalid = xgb.DMatrix(X_valid[features], y_valid)

watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, \
  early_stopping_rounds=100, feval=rmspe_xg, verbose_eval=True)

print("Validating")
yhat = gbm.predict(xgb.DMatrix(X_valid[features]))
error = rmspe(np.expm1(yhat),X_valid.Sales.values)
print('RMSPE: {:.6f}'.format(error))

dtest = xgb.DMatrix(rossman_test[features])
test_probs = gbm.predict(dtest)
xgb.plot_importance(gbm)
submission = pd.DataFrame({"Id": rossman_test["Id"] ,"Sales": np.expm1(test_probs)})
submission.to_csv("sixthSubmission.csv", index = False)

create_feature_map(features)
importance = gbm.get_fscore(fmap='xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1))

df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()

featp = df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')
fig_featp = featp.get_figure()
fig_featp.savefig('feature_importance_xgb.png', bbox_inches='tight', pad_inches=1)	'''
