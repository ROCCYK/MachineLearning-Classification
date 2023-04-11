# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 21:41:44 2023

@author: noopa
"""

"""

Artificial dataset created in Excel (anomaly.csv).
First column is Date
Second column is supposed to be Power reading: Good between 90 and 100
Third column: Detector reading: Good between 8-12
Last coumn is QUality which is either Good or Bad. 
"""

import pandas as pd
df = pd.read_csv('C:/Durham College Teaching/10. Winter 2023/Machine Learning Classification/Week-10/Code/anomaly.csv')

#To see how the data is spread betwen Good and Bad
print(df.groupby('Quality')['Quality'].count())

df.drop(['Date'], axis=1, inplace=True)
print(df.head())

#If there are missing entries, drop them.
df.dropna(inplace=True,axis=1)


#Convert non-numeric to numeric
df.Quality[df.Quality == 'Good'] = 1
df.Quality[df.Quality == 'Bad'] = 2


good_mask = df['Quality']== 1 #All good to be True for good data points
bad_mask = df['Quality']== 2 #All values False for good data points
print(good_mask.head())

df.drop('Quality',axis=1,inplace=True)

df_good = df[good_mask]
df_bad = df[bad_mask]

print(f"Good count: {len(df_good)}")
print(f"Bad count: {len(df_bad)}")

# This is the feature vector that goes to the neural net
x_good = df_good.values
x_bad = df_bad.values

from sklearn.model_selection import train_test_split

x_good_train, x_good_test = train_test_split(
        x_good, test_size=0.25, random_state=42)

print(f"Good train count: {len(x_good_train)}")
print(f"Good test count: {len(x_good_test)}")

######### 
#Define the autoencoder model
#Since we're dealing with numeric values we can use only Dense layers.

from sklearn import metrics
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_dim=x_good.shape[1], activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(x_good.shape[1])) 
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()

model.fit(x_good_train,x_good_train,verbose=1,epochs=100)

pred = model.predict(x_good_test)
score1 = np.sqrt(metrics.mean_squared_error(pred,x_good_test))

pred = model.predict(x_good)
score2 = np.sqrt(metrics.mean_squared_error(pred,x_good))

pred = model.predict(x_bad)
score3 = np.sqrt(metrics.mean_squared_error(pred,x_bad))

print(f"Insample Good Score (RMSE): {score1}".format(score1))
print(f"Total Good Score (RMSE): {score2}")
print(f"Bad sample Score (RMSE): {score3}")

