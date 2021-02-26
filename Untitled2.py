
# coding: utf-8

# In[1]:

get_ipython().magic('matplotlib notebook')
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

np.set_printoptions(precision=2)


fruits = pd.read_table('readonly/fruit_data_with_colors.txt')

feature_names_fruits = ['height', 'width', 'mass', 'color_score']
X_fruits = fruits[feature_names_fruits]
y_fruits = fruits['fruit_label']
target_names_fruits = ['apple', 'mandarin', 'orange', 'lemon']

X_fruits_2d = fruits[['height', 'width']]
y_fruits_2d = fruits['fruit_label']

X_train, X_test, y_train, y_test = train_test_split(X_fruits, y_fruits, random_state=0)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
# we must apply the scaling to the test set that we computed for the training set
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train_scaled, y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(X_train_scaled, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn.score(X_test_scaled, y_test)))

example_fruit = [[5.5, 2.2, 10, 0.70]]
example_fruit_scaled = scaler.transform(example_fruit)
print('Predicted fruit type for ', example_fruit, ' is ', 
          target_names_fruits[knn.predict(example_fruit_scaled)[0]-1])


# In[2]:

from sklearn.datasets import make_classification,make_blobs
from sklearn.datasets import load_breast_cancer
from adspy_shared_utilities import load_crime_dataset
from matplotlib.colors import ListedColormap

cmap_bold = ListedColormap(['#FFFF00', '#00FF00', '#0000FF','#000000'])

from sklearn.datasets import make_regression

plt.figure()
plt.title('Sample regression problem with one input variable')
X_R1, y_R1=make_regression(n_samples=100, n_features=1, n_informative=1, bias=150.0, noise=30, random_state=0)
plt.scatter(X_R1, y_R1, marker='o',s=50)
plt.show()

from sklearn.datasets import make_friedman1
plt.figure()
plt.title('Complex regression problem with one input variable')
X_F1, y_F1=make_friedman1(n_samples=100, n_features=7, random_state=0)
plt.scatter(X_F1[:, 2], y_F1, marker='o',s=50)
plt.show()

plt.figure()
plt.title('Sample binary classification problem with two informative features')
X_C2,y_C2=make_classification(n_samples=100, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, flip_y=0, class_sep=0.5, random_state=0)
plt.scatter(X_C2[:,0], X_C2[:,1], c=y_C2, marker='o', s=50, cmap=cmap_bold)
plt.show()

X_D2, y_D2= make_blobs(n_samples=100, n_features=2, centers=8, cluster_std=1.3, random_state=4)
y_D2=y_D2 % 2
plt.figure()
plt.title('Sample binary classification problem with non-linearly separable classes')
plt.scatter(X_D2[:,0], X_D2[:,1], c=y_D2, marker= 'o', s=50, cmap=cmap_bold)
plt.show()

cancer=load_breast_cancer()
(X_cancer, y_cancer)=load_breast_cancer(return_X_y=True)

(X_crime,y_crime)=load_crime_dataset()


# In[3]:

#Linear Models for Regression
plt.figure(figsize=(5,4))
plt.scatter(X_R1, y_R1, marker='o', s=50, alpha=0.8)
plt.plot(X_R1, linreg.coef)


# In[4]:

from sklearn.linear_model import LinearRegression


X_train, X_test, y_train, y_test= train_test_split(X_R1, y_R1, random_state=0)
linreg=LinearRegression().fit(X_train, y_train)

print('Linear model coeff (w): {}'.format(linreg.coef_))
print('linear model intercept (w): {:.3f}'.format(linreg.intercept_))
print('R-squared score(training): {:.3f}'.format(linreg.score(X_train, y_train)))
print('R-squared score(test): {:.3f}'.format(linreg.score(X_test, y_test)))


# In[5]:

#Linear Models for Regression
plt.figure(figsize=(5,4))
plt.scatter(X_R1, y_R1, marker='o', s=50, alpha=0.8)
plt.plot(X_R1, linreg.coef_*X_R1 + linreg.intercept_,'r')
plt.title('Least-squares linear regression')
plt.xlabel('Feature value(x)')
plt.ylabel('Target value(y)')
plt.show()


# In[6]:

X_train, X_test, y_train, y_test= train_test_split(X_crime, y_crime, random_state=0)
linreg=LinearRegression().fit(X_train, y_train)
print('Crime dataset')
print('linear model coefficient (w): {}'.format(linreg.coef_))
print('linear model intercept (w): {:.3f}'.format(linreg.intercept_))
print('R-squared score(training): {:.3f}'.format(linreg.score(X_train, y_train)))
print('R-squared score(test): {:.3f}'.format(linreg.score(X_test, y_test)))
      


# In[7]:

#Ridge regression
from sklearn.linear_model import Ridge
X_train, X_test, y_train, y_test=train_test_split(X_crime, y_crime)

linridge=Ridge(alpha=20.0).fit(X_train, y_train)
print('Crime dataset')
print('Ridge model coefficient (w): {}'.format(linridge.coef_))
print('Ridge model intercept (w): {:.3f}'.format(linridge.intercept_))
print('R-squared score(training): {:.3f}'.format(linridge.score(X_train, y_train)))
print('R-squared score(test): {:.3f}'.format(linridge.score(X_test, y_test)))
print('Number of non-zero features: {}'.format(np.sum(linridge.coef_!=0)))


# In[8]:

#Ridge regression with feature normalisation
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
from sklearn.linear_model import Ridge

X_train, X_test, y_train, y_test=train_test_split(X_crime, y_crime, random_state=0)
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
linridge=Ridge(alpha=20.0).fit(X_train_scaled,y_train)
print('Crime dataset')
print('ridge regression linear model intercept: {}'
     .format(linridge.intercept_))
print('ridge regression linear model coeff:\n{}'
     .format(linridge.coef_))
print('R-squared score (training): {:.3f}'
     .format(linridge.score(X_train_scaled, y_train)))
print('R-squared score (test): {:.3f}'
     .format(linridge.score(X_test_scaled, y_test)))
print('Number of non-zero features: {}'
     .format(np.sum(linridge.coef_ != 0)))


# In[10]:

#Ridge regression with regularisation parameter
print('Ridge regression: effect of alpha regularisation')
for this_alpha in [0,1,10,20,50,100,1000]:
    linridge=Ridge(alpha=this_alpha).fit(X_train_scaled, y_train)
    r2_train=linridge.score(X_train_scaled, y_train)
    r2_test=linridge.score(X_test_scaled, y_test)
    num_coeff_bigger=np.sum(abs(linridge.coef_)>1.0)
    print('Alpha={:.2f}\nnum abs(coeff)>1.0:{}, \ r-squared training {:.2f}, r_squared_test:{:.2f}\n'.format(this_alpha, num_coeff_bigger, r2_train, r2_test))


# In[11]:

#Lasso regression
from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler
 
scaler=MinMaxScaler()

X_train, X_test, y_train, y_test=train_test_split(X_crime, y_crime, random_state=0)
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

linlasso=Lasso(alpha=2.0, max_iter=10000).fit(X_train_scaled, y_train)
print('Crime dataset')
print('lasso regression linear model intercept: {}'
     .format(linlasso.intercept_))
print('lasso regression linear model coeff:\n{}'
     .format(linlasso.coef_))
print('R-squared score (training): {:.3f}'
     .format(linlasso.score(X_train_scaled, y_train)))
print('R-squared score (test): {:.3f}'
     .format(linlasso.score(X_test_scaled, y_test)))
print('Features with non-zero weight (sorted by absolute magnitude):')
     
for e in sorted (list(zip(list(X_crime), linlasso.coef_)), key=lambda e: -abs(e[1])):
            if e[1]!=0:
               print('\t{}, {:.3f}'.format(e[0],e[1]))


# In[12]:

#Lasso regression with regularisation parameter : alpha
print('Lasso regression: effect of alpha regularization\nparameter on number of features kept in final model\n')

for alpha in [0.5, 1, 2, 3, 5, 10, 20, 50]:
    linlasso = Lasso(alpha, max_iter = 10000).fit(X_train_scaled, y_train)
    r2_train = linlasso.score(X_train_scaled, y_train)
    r2_test = linlasso.score(X_test_scaled, y_test)
    
    print('Alpha = {:.2f}\nFeatures kept: {}, r-squared training: {:.2f}, r-squared test: {:.2f}\n'
         .format(alpha, np.sum(linlasso.coef_ != 0), r2_train, r2_test))


# In[13]:

#Polynomial regression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
X_train, X_test, y_train, y_test= train_test_split(X_crime, y_crime, random_state=0)
linreg=LinearRegression().fit(X_train, y_train)
print('Crime dataset')
print('linear model coefficient (w): {}'.format(linreg.coef_))
print('linear model intercept (w): {:.3f}'.format(linreg.intercept_))
print('R-squared score(training): {:.3f}'.format(linreg.score(X_train, y_train)))
print('R-squared score(test): {:.3f}'.format(linreg.score(X_test, y_test)))

print('\nNow we transform the original input data to add \n\ polynomial features up to degree 2 (quadratic)\n')
poly=PolynomialFeatures(degree=2)
X_F1_poly=poly.fit_transform(X_F1)

X_train, X_test, y_train, y_test=train_test_split(X_F1_poly, y_F1,random_state=0)

linreg=LinearRegression().fit(X_train, y_train)

print('(poly deg 2)linear model coefficient (w): {}'.format(linreg.coef_))
print('(poly deg 2)linear model intercept (w): {:.3f}'.format(linreg.intercept_))
print('(poly deg 2)R-squared score(training): {:.3f}'.format(linreg.score(X_train, y_train)))
print('(poly deg 2)R-squared score(test): {:.3f}'.format(linreg.score(X_test, y_test)))

print('n\Addition of many polynomial features often leads to \n\overfitting, so we often use polynomial features in combnation\n\ with regression that has a regularisation penalty, like ridge\n\ regression.\n ')
X_train, X_test, y_train, y_test=train_test_split(X_F1_poly, y_F1, random_state=0)

linreg=Ridge().fit(X_train, y_train)
print('(poly deg 2 + ridge)linear model coefficient (w): {}'.format(linreg.coef_))
print('(poly deg 2 + ridge)linear model intercept (w): {:.3f}'.format(linreg.intercept_))
print('(poly deg 2 + ridge)R-squared score(training): {:.3f}'.format(linreg.score(X_train, y_train)))
print('(poly deg 2 + ridge)R-squared score(test): {:.3f}'.format(linreg.score(X_test, y_test)))


# In[14]:

#Logistic regression
from sklearn.linear_model import LogisticRegression
from adspy_shared_utilities import (plot_class_regions_for_classifier_subplot)
fig, subaxes=plt.subplots(1,1, figsize=(7,5))
y_fruits_apple=y_fruits_2d==1 #make into a binary problem: apple vs everything else
X_train, X_test, y_train, y_test=train_test_split(X_fruits_2d.as_matrix(),y_fruits_apple.as_matrix(),random_state=0)
clf=LogisticRegression(C=100).fit(X_train, y_train)
plot_class_regions_for_classifier_subplot(clf, X_train, y_train, None, None, 'Logistic regression for binary classification\nFruit dataset: Apple vs others', subaxes)
h=6
w=8
print('A fruit with height{} and width{} is predicted to be: {}'.format(h,w,['not an apple', 'an apple'][clf.predict([[h,w]])[0]]))
h=10
w=7
print('A fruit with height{} and width{} is predicted to be: {}'.format(h,w,['not an apple', 'an apple'][clf.predict([[h,w]])[0]]))
subaxes.set_xlabel('height')
subaxes.set_ylabel('width')

print('Accuracy of logisic regression classifier on training set is: {}'.format(clf.score(X_train, y_train)))
print('Accuracy of logisic regression classifier on test set is: {}'.format(clf.score(X_test, y_test)))
                                                                


# In[15]:

#Logistic regression on simple synthetic dataset
from sklearn.linear_model import LogisticRegression
from adspy_shared_utilities import (plot_class_regions_for_classifier_subplot)

X_train, X_test, y_train, y_test=train_test_split(X_C2, y_C2, random_state=0)

fig, subaxes=plt.subplots(1,1, figsize=(7,5))
clf=LogisticRegression().fit(X_train, y_train)
title='Logistic Regression, simple synthetic dataset C={:.3f}'.format(1.0)
plot_class_regions_for_classifier_subplot(clf, X_train, y_train, None, None, title, subaxes)

print('Accuracy of Logistic Regression classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
print('Accuracy of Logistic Regression classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))


# In[16]:

#Logistic regression regularisation: C parameter
X_train, X_test, y_train, y_test = (
train_test_split(X_fruits_2d.as_matrix(),
                y_fruits_apple.as_matrix(),
                random_state=0))

fig, subaxes = plt.subplots(3, 1, figsize=(4, 10))

for this_C, subplot in zip([0.1, 1, 100], subaxes):
    clf = LogisticRegression(C=this_C).fit(X_train, y_train)
    title ='Logistic regression (apple vs rest), C = {:.3f}'.format(this_C)
    
    plot_class_regions_for_classifier_subplot(clf, X_train, y_train,
                                             X_test, y_test, title,
                                             subplot)
plt.tight_layout()


# In[17]:

#Application to real dataset
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test= train_test_split(X_cancer, y_cancer)
clf=LogisticRegression().fit(X_train, y_train)
print('Breast Cancer dataset')
print('Accuracy of logistic regression on training set: {:.2f}'.format(clf.score(X_train, y_train)))
print('Accuracy of logistic regression on test set: {:.2f}'.format(clf.score(X_test, y_test)))


# In[18]:

#Support Vector Machines
from sklearn.svm import SVC
from adspy_shared_utilities import plot_class_regions_for_classifier_subplot

X_train, X_test, y_train, y_test=train_test_split(X_C2, y_C2, random_state=0)
fig, subaxes=plt.subplots(1,1, figsize=(7,5))
this_c=1.0
clf=SVC(kernel='linear', C=this_c).fit(X_train, y_train)
title='Linear SVC, C={:.3f}'.format(this_c)
plot_class_regions_for_classifier_subplot(clf, X_train, y_train, None, None, title, subaxes)


# In[19]:

#Linear Support Vector Machine:C parameter
from sklearn.svm import LinearSVC
from adspy_shared_utilities import plot_class_regions_for_classifier

X_train, X_test, y_train, y_test = train_test_split(X_C2, y_C2, random_state = 0)
fig, subaxes = plt.subplots(1, 2, figsize=(8, 4))

for this_C, subplot in zip([0.00001, 100], subaxes):
    clf = LinearSVC(C=this_C).fit(X_train, y_train)
    title = 'Linear SVC, C = {:.5f}'.format(this_C)
    plot_class_regions_for_classifier_subplot(clf, X_train, y_train,
                                             None, None, title, subplot)
plt.tight_layout()


# In[20]:

#Application to real dataset
from sklearn.svm import LinearSVC
X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state = 0)

clf = LinearSVC().fit(X_train, y_train)
print('Breast cancer dataset')
print('Accuracy of Linear SVC classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of Linear SVC classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))


# In[21]:

#Multi class classification with linear models
#LinearSVC with M classes generates M one vs rest classifiers
from sklearn.svm import LinearSVC

X_train, X_test, y_train, y_test=train_test_split(X_fruits_2d, y_fruits_2d, random_state=0)

clf=LinearSVC(C=5, random_state=67).fit(X_train, y_train)
print('Coefficient:\n',clf.coef_)
print('Intercepts:\n',clf.intercept_)


# In[22]:

plt.figure(figsize=(6,6))
colors = ['r', 'g', 'b', 'y']
cmap_fruits = ListedColormap(['#FF0000', '#00FF00', '#0000FF','#FFFF00'])

plt.scatter(X_fruits_2d[['height']], X_fruits_2d[['width']],
           c=y_fruits_2d, cmap=cmap_fruits, edgecolor = 'black', alpha=.7)

x_0_range = np.linspace(-10, 15)

for w, b, color in zip(clf.coef_, clf.intercept_, ['r', 'g', 'b', 'y']):
    # Since class prediction with a linear model uses the formula y = w_0 x_0 + w_1 x_1 + b, 
    # and the decision boundary is defined as being all points with y = 0, to plot x_1 as a 
    # function of x_0 we just solve w_0 x_0 + w_1 x_1 + b = 0 for x_1:
    plt.plot(x_0_range, -(x_0_range * w[0] + b) / w[1], c=color, alpha=.8)
    
plt.legend(target_names_fruits)
plt.xlabel('height')
plt.ylabel('width')
plt.xlim(-2, 12)
plt.ylim(-2, 15)
plt.show()


# In[23]:

#Kernelised Support Vector Machines
#Classification
from sklearn.svm import SVC
from adspy_shared_utilities import plot_class_regions_for_classifier

X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state = 0)

# The default SVC kernel is radial basis function (RBF)
plot_class_regions_for_classifier(SVC().fit(X_train, y_train),
                                 X_train, y_train, None, None,
                                 'Support Vector Classifier: RBF kernel')

# Compare decision boundries with polynomial kernel, degree = 3
plot_class_regions_for_classifier(SVC(kernel = 'poly', degree = 3)
                                 .fit(X_train, y_train), X_train,
                                 y_train, None, None,
                                 'Support Vector Classifier: Polynomial kernel, degree = 3')


# In[24]:

#SVM withh RBF kernel using gamma
from adspy_shared_utilities import plot_class_regions_for_classifier

X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state = 0)
fig, subaxes = plt.subplots(3, 1, figsize=(4, 11))

for this_gamma, subplot in zip([0.01, 1.0, 10.0], subaxes):
    clf = SVC(kernel = 'rbf', gamma=this_gamma).fit(X_train, y_train)
    title = 'Support Vector Classifier: \nRBF kernel, gamma = {:.2f}'.format(this_gamma)
    plot_class_regions_for_classifier_subplot(clf, X_train, y_train,
                                             None, None, title, subplot)
    plt.tight_layout()


# In[25]:

#SVM with RBF kernel using gamma and c
from sklearn.svm import SVC
from adspy_shared_utilities import plot_class_regions_for_classifier_subplot

from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state = 0)
fig, subaxes = plt.subplots(3, 4, figsize=(15, 10), dpi=50)

for this_gamma, this_axis in zip([0.01, 1, 5], subaxes):
    
    for this_C, subplot in zip([0.1, 1, 15, 250], this_axis):
        title = 'gamma = {:.2f}, C = {:.2f}'.format(this_gamma, this_C)
        clf = SVC(kernel = 'rbf', gamma = this_gamma,
                 C = this_C).fit(X_train, y_train)
        plot_class_regions_for_classifier_subplot(clf, X_train, y_train,
                                                 X_test, y_test, title,
                                                 subplot)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)


# In[26]:

#Application of SVMs to a real dataset: unnormalized data
from sklearn.svm import SVC
X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer,
                                                   random_state = 0)

clf = SVC(C=10).fit(X_train, y_train)
print('Breast cancer dataset (unnormalized features)')
print('Accuracy of RBF-kernel SVC on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of RBF-kernel SVC on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))


# In[27]:

#Application of SVMs to a real dataset: normalized data with feature preprocessing using minmax scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = SVC(C=10).fit(X_train_scaled, y_train)
print('Breast cancer dataset (normalized with MinMax scaling)')
print('RBF-kernel SVC (with MinMax scaling) training set accuracy: {:.2f}'
     .format(clf.score(X_train_scaled, y_train)))
print('RBF-kernel SVC (with MinMax scaling) test set accuracy: {:.2f}'
     .format(clf.score(X_test_scaled, y_test)))


# In[30]:

#Cross-Validation
#Example on KNN classifier with fruit datasets(2features)
from sklearn.model_selection import cross_val_score
clf=KNeighborsClassifier(n_neighbors=5)
X=X_fruits_2d.as_matrix()
y=y_fruits_2d.as_matrix()
cv_scores=cross_val_score(clf, X, y)

print('Cross Validation scores(3 fold)',cv_scores)
print('Mean cross validation score(3 fold): {:.3f}'.format(np.mean(cv_scores)))


# In[33]:

#Validation curve example
from sklearn.svm import SVC
from sklearn.model_selection import validation_curve

param_range=np.logspace(-3,3,4)
train_scores, test_scores=validation_curve(SVC(), X, y, param_name='gamma', param_range=param_range, cv=3)


# In[34]:

print(test_scores)


# In[35]:

print(train_scores)


# In[37]:

plt.figure()

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title('Validation Curve with SVM')
plt.xlabel('$\gamma$ (gamma)')
plt.ylabel('Score')
plt.ylim(0.0, 1.1)
lw = 2

plt.semilogx(param_range, train_scores_mean, label='Training score',
            color='darkorange', lw=lw)

plt.fill_between(param_range, train_scores_mean - train_scores_std,
                train_scores_mean + train_scores_std, alpha=0.2,
                color='darkorange', lw=lw)

plt.semilogx(param_range, test_scores_mean, label='Cross-validation score',
            color='navy', lw=lw)

plt.fill_between(param_range, test_scores_mean - test_scores_std,
                test_scores_mean + test_scores_std, alpha=0.2,
                color='navy', lw=lw)

plt.legend(loc='best')
plt.show()


# In[ ]:

#Decision Trees
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from adspy_shared_utilities import plot_decision_tree
from sklearn.model_selection import train_test_split

iris=load_iris()

X_train, X_test, y_train, y_test=train_test_split(iris.data, iris.target, random_state=0)

clf=DecisionTreeClassifier().fit(X_train, y_train)

print('Accuracy of Decision Tree Classifier on training set :{:.2f}'.format(clf.score(X_train, y_train)))
print('Accuracy of Decision Tree Classifier on training set :{:.2f}'.format(clf.score(X_test, y_test)))

