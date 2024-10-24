#Uploaded to git hub
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score
# The above code just imports the libraries as the chosen names
csv_data = pd.read_csv("FuelConsumptionCo2.csv")
 #.read_csv() reads the csv file
print(csv_data.head()) 
#.head() prints the first five rows from the csv file 
#print (csv_data.describe())
selected_data = csv_data[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
# we will call the variables in this program data frame from now on 
#the above executible line of code create a new data frame from the csv_data set containing only the specified columns 
print(selected_data.head(9))
#passing a number n in the head(n) function will display n rows
plotable = selected_data[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
plotable.hist()
plt.savefig('regression.png')
plt.clf()
#creates a histogram which plots range at x axis and frequency of values on y axis refer to ai engineer notebook for detail
plt.scatter(plotable.FUELCONSUMPTION_COMB,plotable.CO2EMISSIONS,color ='blue')
#plots the values against provided arguements
plt.xlabel("FUELCONSUMPTION_COMB")
#xlable
plt.ylabel("Emission")
#ylable
plt.savefig('fuel_vs_co2.png')
#saves the plot as a file
plt.clf()
#clears the current figure
plt.scatter(plotable.ENGINESIZE,plotable.CO2EMISSIONS,color = 'yellow')
plt.xlabel("Engine Size")
plt.ylabel("CO2 Emissions")
plt.savefig('engine_vs_emissions.png')
plt.clf()
plt.scatter(plotable.CYLINDERS,plotable.CO2EMISSIONS,color = 'red')
plt.xlabel("CYLINDERS")
plt.ylabel("CO2 Emissions")
plt.savefig('cylinders_vs_emissions.png')
split = np.random.rand(len(csv_data)) < 0.8
# the ran function takes in an arguement as number of array or dimensions of array and populates it with values between 0 to 1 
# <0.8 condition is applied to all the elements of the array it checks if value is less than 0.8 returns it true and 
# other values return false this makes array 20 percent false and 80 percent true splitting it up
train = csv_data[split]
# matches all the indeces of csv_data with  split those that align with true values are stored in train
test = csv_data[~split]
# matches all the indeces of csv_data with  split those that align with false  values are stored in test
plt.clf()
plt.scatter(train.ENGINESIZE,train.CO2EMISSIONS,color ='blue')
plt.xlabel("EngineSize")
plt.ylabel("CO2Emissions")
plt.savefig('train-distribution.png')
model = linear_model.LinearRegression()
# initializes model as a linear regression object 
train_x = np.asanyarray(train[['ENGINESIZE']])
#train[['ENGINSIZE']] selects the enginesize column from train asanyarray converts that column from train dataframe to an array
train_y = np.asanyarray(train[['CO2EMISSIONS']])
#does the same as explained above 
model.fit(train_x,train_y)
#fit method is used to train the machine learning model .fit() method feeds the model data 
print('Coefficient :',model.coef_)
print('Intercept :',model.intercept_)
plt.clf()
plt.scatter(train.ENGINESIZE,train.CO2EMISSIONS,color='blue')
plt.plot(train_x,model.coef_[0][0]*train_x+model.intercept_[0],'-r')
# -means dashed line r means red further explaination in notebook notes the above formula is always going to remain the same
plt.xlabel('EngineSize')
plt.ylabel('CO2Emissions')
plt.savefig('regression-result.png')
test_x = np.asanyarray(test[['ENGINESIZE']])
#creating test sets
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_= model.predict(test_x)
#predict method is used to make predictions will predcit the values of test_x against test_y and store them in test_y_
# here it will compare Engine Size and CO2emissions
plt.clf()
plt.scatter(test.ENGINESIZE,test.CO2EMISSIONS,color='red')
plt.scatter(train.ENGINESIZE,train.CO2EMISSIONS,color='blue')
plt.plot(train_x,model.coef_[0][0]*train_x+model.intercept_[0],'-r')
plt.xlabel('EngineSize')
plt.ylabel('CO2Emissions')
plt.savefig('predicted.png')
print("Mean absolute Error: %2.f" % np.mean(np.absolute(test_y_-test_y)))
print("Residual Sum Of Squares : %2.f" % np.mean((test_y_-test_y)**2))
print("R2-score : %2.f" % r2_score(test_y,test_y_))
# Using another variable for trainging and testing
model2 = linear_model.LinearRegression()
train_2_x = np.asanyarray(train[['FUELCONSUMPTION_COMB']])
model2.fit(train_2_x,train_y)
test_2_x = np.asanyarray(test[['FUELCONSUMPTION_COMB']])
print("Coeffecient : ",model2.coef_)
print("Intercept : ",model2.intercept_)
test_2y = model2.predict(test_2_x)
plt.clf()
plt.scatter(test.ENGINESIZE,test.CO2EMISSIONS,color='red')
plt.scatter(train.ENGINESIZE,train.CO2EMISSIONS,color='blue')
plt.plot(train_2_x,model2.coef_[0][0]*train_2_x+model2.intercept_[0],'-r')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("CO2EMISSIONS")
plt.savefig('model2.png')
print("Mean absolute Error: %2.f" % np.mean(np.absolute(test_2y-test_y)))
print("Residual Sum Of Squares : %2.f" % np.mean((test_2y-test_y)**2))
print("R2-score : %2.f" % r2_score(test_y,test_2y))
