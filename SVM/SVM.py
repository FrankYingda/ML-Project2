import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
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

data2 = data.drop(['id'], axis=1)
data2['Class'] = data2['Class'].map({4:1, 2:0})

#print(data2)

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
    search = GridSearchCV(svm.SVC(kernel='poly'), param_grid, cv=nfolds)
    search.fit(X, y)        #fit the search object to input training data
    search.best_params_     #return the best parameters
    return search.best_params_

print("best parameters:", svc_param_selection(X_train, y_train, 10))

final_svc_poly = svm.SVC(C=0.01, degree=2, kernel='poly')
final_svc_poly.fit(X_train, y_train)
print("the optimal accuracy is: ", final_svc_poly.score(X_test, y_test))


sns.set()
#confusion matrix
confusion_mat = confusion_matrix(y_test, final_svc_poly.predict(X_test))
print("confusion matrix is: \n", confusion_mat)
plt.subplots(figsize=(9,6))
xlabel = ["true","false"]
ylabel = ["positive","negative"]

sns.heatmap(confusion_mat, annot=True, annot_kws={'size':20,'weight':'bold', 'color':'yellow'}, fmt='d',
            xticklabels=xlabel, yticklabels=ylabel)
plt.ylim(0, confusion_mat.shape[1])

plt.show()

end = time.clock()
t=end-start
print("Runtime is: ",t)


