#reference: https://github.com/ryanschaub/Breast-Cancer-Classification-using-Support-Vector-Machine-Models/blob/master/Homework%203.ipynb
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

import warnings
warnings.filterwarnings('ignore')

start = time.clock()
column_names = ["id",  "Clump Thickness", "Uniformity of Cell Size","Uniformity of Cell Shape",
                                "Marginal Adhesion", "Single Epithelial Cell Size","Bare Nuclei",
                                "Bland Chromatin", "Normal Nucleoli", "Mitoses", "Class"]

data = pd.read_csv("breast-cancer-wisconsin.data", names = column_names)
data = data.replace(to_replace='?', value = np.nan)
data = data .dropna(how = 'any')
print("whether missing values? ", data.isnull().values.any())          #check for missing values
#print(data)

# heatmap
featurs_mean = list(data.columns[1:11])
corr = data[featurs_mean].corr()
plt.figure(figsize=(14,14))
sns.heatmap(corr, annot=True)
plt.ylim(corr.shape[1],0)
plt.show()

#clssify the data
data2 = data.drop(['id'], axis=1)
data2['Class'] = data2['Class'].map({4:1, 2:0})
features = data2.drop('Class', axis=1)
labels = data2['Class']

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=66)

#Misclassification Penalty Values
Cs = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
#Degree of Polynomial Kernel
degrees = [1, 2, 3, 4, 5]

train_acc = []
test_acc = []

#search for best parameters using input nfold cross validation
def svc_param_selection(X, y, nfolds):
    param_grid = {'C': Cs, 'degree' : degrees}
    search = GridSearchCV(svm.SVC(kernel='linear'), param_grid, cv=nfolds)
    search.fit(X, y)        #fit the search object to input training data
    search.best_params_     #return the best parameters
    return search.best_params_

print("best parameters:", svc_param_selection(X_train, y_train, 10))

svc_linear = svm.SVC(C=0.01, degree=2, kernel='linear')
svc_linear.fit(X_train, y_train)
print("the optimal accuracy is: ", svc_linear.score(X_test, y_test))

#confusion matrix
confusion_mat = confusion_matrix(y_test, svc_linear.predict(X_test))
print("confusion matrix is: \n", confusion_mat)
plt.subplots(figsize=(9,6))
xlabel = ["true","false"]
ylabel = ["positive","negative"]

sns.heatmap(confusion_mat, annot=True, annot_kws={'size':20,'weight':'bold', 'color':'yellow'}, fmt='d',
            xticklabels=xlabel, yticklabels=ylabel)
plt.ylim(confusion_mat.shape[1],0)
plt.show()



########SVM virtualization########
features = data2.iloc[:, 2:5]
labels = data2['Class']

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=66)

cls = svm.SVC(C=0.01, degree=2, kernel='linear')
cls.fit(X_train, y_train)

n_Support_vector = cls.n_support_   # sv number
sv_idx = cls.support_               # sv index
w = cls.coef_                       # direction vector W
b = cls.intercept_

print(n_Support_vector,sv_idx,w,b)

ax = plt.subplot(111, projection='3d')
x = np.arange(0,10,0.1)
y = np.arange(0,10,0.01)
x, y = np.meshgrid(x, y)
z = (w[0,0]*x + w[0,1]*y + b) / (-w[0,2])
surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))

# scatter diagram
x_array = np.array(X_train, dtype=int)
y_array = np.array(y_train, dtype=int)
pos = x_array[np.where(y_array==1)]
neg = x_array[np.where(y_array==0)]
ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c='r', label='Malignant')
ax.scatter(neg[:, 0], neg[:, 1], neg[:, 2], c='b', label='Benign')


ax.set_zlabel('Z')    # 坐标轴
ax.set_ylabel('Y')
ax.set_xlabel('X')
#ax.set_zlim([0, 1])
plt.legend(loc='upper left')
plt.show()


end = time.clock()
t=end-start
print("Runtime is: ",t)


