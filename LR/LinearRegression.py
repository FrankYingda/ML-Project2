#Reference source: https://towardsdatascience.com/linear-regression-using-gradient-descent-97a6c8700931
import pandas as pd  # data handeling
import numpy as np   # numeriacal computing
import matplotlib.pyplot as plt  # plotting core
import time
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns

start = time.clock()

data = pd.read_csv('D:\JetBrains\PyCharm 2018.2\Workspace\MLProject2\LinearRegression\kc_house_data.csv')
print("whether missing values? ", data.isnull().values.any())          #check for missing values

# pairplot to find the most relatively feature to the price
g = sns.pairplot(data, vars=["price", "bedrooms","bathrooms","sqft_living",
                            "sqft_lot","grade","sqft_above","sqft_basement"],kind="reg")
plt.show()

X = data.iloc[:, data.columns == 'sqft_living']
Y = data.iloc[:, data.columns == 'price']

max_data = np.max(X, axis=0)        #standardizing the dataset to avoid overflow
min_data = np.min(X, axis=0)
max_set = np.zeros_like(X);
max_set[:] = max_data
min_set = np.zeros_like(X);
min_set[:] = min_data
X_stand = (X - min_set) / (max_set - min_set)

max_data = np.max(Y, axis=0)
min_data = np.min(Y, axis=0)
max_set = np.zeros_like(Y);
max_set[:] = max_data
min_set = np.zeros_like(Y);
min_set[:] = min_data
Y_stand = (Y - min_set) / (max_set - min_set)

X1 = np.array(X_stand)
Y1 = np.array(Y_stand)

m = 0       #slope
c = 0       #y-intercept
L = 0.8  # The learning Rate
itera = 1000  # The number of iterations to perform gradient descent

n = float(len(X1))  # Number of elements in X

# Performing Gradient Descent
for i in range(itera):
    Y_pred = m * X1 + c  # The current predicted value of Y
    D_k = (-2 / n) * sum(X1 * (Y1 - Y_pred))  # Derivative wrt k
    D_b = (-2 / n) * sum(Y1 - Y_pred)  # Derivative wrt b
    m = m - L * D_k  # Update m
    c = c - L * D_b  # Update c

print("final m and c are", m, c)
Y_pred = m*X1 + c

plt.scatter(X1, Y1)
plt.plot([min(X1), max(X1)], [min(Y_pred), max(Y_pred)], color='red')  # regression line
plt.xlabel("sqrt_living")
plt.ylabel("price")
plt.title ("Learning Rate = 0.8")
plt.show()

end = time.clock()
t=end-start
print("Runtime is: ",t)

