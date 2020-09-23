from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv('data1.txt',delimiter=',')
X=data['Head Size(cm^3)'].values
Y=data['Brain Weight(grams)'].values
m=len(X)
X=X.reshape((m,1))
reg=LinearRegression()
reg=reg.fit(X,Y)
y_pred=reg.predict(X)
plt.scatter(X,Y)
plt.plot(X,y_pred)
plt.show()
r2=reg.score(X,Y)
print(r2)