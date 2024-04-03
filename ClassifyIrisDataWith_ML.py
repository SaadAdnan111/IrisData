import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics

### Apply Machine Learning Classification Model to the Iris Dataset ###

## Load and Access the Data ..

#df = pd.read_csv("Iris.csv")
iris = load_iris()
#print(type(iris))
#print(iris.keys())  
#print(iris.data[:5])    # Display Iris Data ..
#print(iris.DESCR)  # Display Description of the data ..
#print(iris.feature_names)  # Display Features of the data ..
#print(iris.target_names)  # Display thr target names of the data ..

## Pairplot for the dataset ..

#df = sns.load_dataset('iris')
#sns.pairplot(df, hue = "species")
#plt.show()

## Loading and Splitting Data for the model ..

iris = load_iris()
X = pd.DataFrame(iris.data)
y = pd.DataFrame(iris.target)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify = y, random_state = 100)
#print("Shape of X_train dataset: ", X_train.shape)
#print("Shape of X_test dataset: ", X_test.shape)
#print("Shape of y_train dataset: ", y_train.shape)
#print("Shape of y_test dataset: ", y_test.shape)

## Creating model and Decision Tree Classifier ..

model = DecisionTreeClassifier(max_depth = 3, random_state = 100)
#print(type(model))
model.fit(X_train, y_train)  ## Fitting Data to the model ..

## Visualising Decision Tree ..

#plot_tree(model, feature_names = iris.feature_names, class_names = iris.target_names, filled = True)
#plt.title("Decision Tree Classifier for the model ..", size = 18, color = "grey")
#plt.show()

## what are Important Features (Species) for the model ..

#print(model.feature_importances_)

## Making Predictionn with the model ..

prediction = model.predict(X_test)
print("Prediction: ", str(prediction) + " \n0 : Setosa, 1: Versicolor, 2: Virginica ")
accuracy = (metrics.accuracy_score(y_true = y_test, y_pred = prediction)) * 100
print("Accuracy of the Model: ", str(accuracy) + " %" )
