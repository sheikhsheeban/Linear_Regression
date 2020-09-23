#import dependencies
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#load boston housing data set from sklearn.dataset

from sklearn.datasets import load_boston
boston=load_boston()

#Transfer the dataset into data frame
#data=data we want or the independent variable or also known as the x value
#feature_names=the column name of the data
#target=the target variable or the dapendent variable or also known as the y value

df_x=pd.DataFrame(boston.data,columns=boston.feature_names)
df_y=pd.DataFrame(boston.target)

#get some statistic from the data set
df_x.describe()

#initialize the linear regression model
reg=linear_model.LinearRegression()

#spliting the data into 67% training and 33% testing data
x_train,x_test,y_train,y_test=train_test_split(df_x,df_y,test_size=0.33,random_state=42)

#train the model with our training data
reg.fit(x_train,y_train)

#print coefficient/weights and feature/column of our model
print(reg.coef_)

#print prediction on our test data
y_pred=reg.predict(x_test)
print(y_pred)

#print the actual value
print(y_test)

#check the model preformance or accuracy using mean square error from sklearn
print(mean_squared_error(y_test,y_pred))

#check the model preformance or accuracy using mean square error from numpy
print(np.mean((y_pred-y_test)**2))


