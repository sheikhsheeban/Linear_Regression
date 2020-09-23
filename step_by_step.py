import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv('headbrain.csv',delimiter=',')
print(data.shape)
print(data.head())
X=data['Head Size(cm^3)'].values
Y=data['Brain Weight(grams)'].values
#mean of X and  Y
x_mean=np.mean(X)
y_mean=np.mean(Y)
m=len(X)
numer=0
denom=0
for i in range(m):
    numer+=(X[i]-x_mean)*(Y[i]-y_mean)
    denom+=(X[i]-x_mean)**2
b1=numer/denom
b0=y_mean-(b1*x_mean)
print(b1,b0)
max_x=np.max(X)+100
min_x=np.min(X)-100
x=np.linspace(min_x,max_x,1000)
y=b0+b1*x
plt.plot(x,y,color='red',label="Regression Line")
plt.scatter(X,Y,color='orange',label="Scatter Plot")
plt.xlabel("Head Size(cm^3)")
plt.ylabel("Brain Weight(grams)")
plt.legend()
plt.show()
ss_t=0
ss_r=0
for i in range(m):
    y_pred=b0+b1*X[i]
    ss_t+=(Y[i]-y_mean)**2
    ss_r+=(Y[i]-y_pred)**2
r2=1-(ss_r/ss_t)
print(r2)