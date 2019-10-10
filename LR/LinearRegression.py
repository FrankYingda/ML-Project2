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


#Y_train = data.iloc[:,data.columns == 'price']
#X_train = data.iloc[:,3:14]

#X_train_features = [x for i,x in enumerate(data.columns) if i > 2 and  i < 14]

#Y_train = np.array(Y_train)

#print(X_train.shape)
#print(Y_train.shape)

#rf = RandomForestClassifier(n_estimators=100, random_state=0)
#rf.fit(X_train, Y_train)
#print("Accuracy on training set: {:.3f}".format(rf.score(X_train, y_train)))
#print("Accuracy on test set: {:.3f}".format(rf.score(X_test, y_test)))

"""def plot_feature_importances(model):
    plt.figure(figsize=(8,6))
    n_features = 8
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), X_train_features)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)"""

#plot_feature_importances(rf)
#plt.show()


#plt.rcParams['figure.figsize'] = (15.0, 15.0)

#sns.pairplot(data)
#plt.show()"""


"""pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.max_columns', 21)
pd.set_option('display.max_rows', 70)

df = pd.read_csv("D:\JetBrains\PyCharm 2018.2\Workspace\MLProject2\LinearRegression\kc_house_data.csv")

#print(df.isnull().values.any())     # check for missing values

#df_des = df[["price","bedrooms","bathrooms","sqft_living","sqft_lot","sqft_above","yr_built","sqft_living15","sqft_lot15"]].describe()

#print(df_des)


sns.pairplot(data=df, x_vars=['sqft_living','sqft_lot','sqft_above','sqft_living15','sqft_lot15'], y_vars=["price"])"""
