#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# libraries

import pandas as pd
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


# Read the JSON file (After you completed Step 1 of the instructions)

json_file_name = 'houses.json'
house = pd.read_json(json_file_name)


# In[ ]:


# Build Linear Regression Model (Step 2 of instructions)

regr = linear_model.LinearRegression()
regr.fit(house[['sqft_living']], house[['price']])
intercept = regr.intercept_[0]
coef = regr.coef_[0][0]
print("intercept = {}".format(intercept))
print("coef = {}".format(coef))


# In[ ]:


# Make Preditions (Step 3 of instructions)
# todo: uncomment the following lines, and complete the test_array with all the required values

# test_array = [1000, 1200, 1400, 1600, 1800, ..., 3600, 3800, 4000]
results = regr.predict(np.array(test_array).reshape(-1,1)).reshape(1,-1)[0]
for sqft_living, price in zip(test_array, results):
    print("sqft_living: {}, price: {:0.3f}".format(sqft_living, price))


# In[ ]:


# Visualization (Step 4 of instructions)

prediction = pd.DataFrame(test_array, results).reset_index()
prediction.columns = ['price', 'sqft_living']
xs = np.linspace(0,5000,100)
ys = xs * coef + intercept
plt.scatter(house[['sqft_living']], house[['price']])
plt.scatter(prediction[['sqft_living']], prediction[['price']], marker='^', color='red')
plt.plot(xs, ys)
plt.xlim(0,5000)
plt.savefig('result.png')


# In[ ]:




