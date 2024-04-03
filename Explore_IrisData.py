import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics

df = pd.read_csv("Iris.csv")

#print(df.shape)

#print(df.isnull().sum())

#print(df.drop_duplicates(subset = "Species", ))
#print(df.value_counts("Species"))
#sns.set_color_codes("dark")
#sns.scatterplot(x='PetalLengthCm', y='PetalWidthCm', hue='Species', data=df, ) 
#sns.countplot(x = "Species", data = df, color = "b")
#numeric_df = df.drop(['Id', 'Species'], axis=1)

#print(numeric_df.corr(method = "pearson"))
#plt.show()

#def DrawBoxPlot(y):
#    sns.boxplot(x = "Species", y = y , data = df)

#plt.figure(figsize = (10, 10))
#plt.subplot(221)
#DrawBoxPlot('PetalLengthCm')
#plt.subplot(180)
#DrawBoxPlot('SepalWidthCm')

#plt.show()

'''


## Finding Outliers and removing them ..

## For SepalWitdthCm Column ..

Q1 = np.percentile(df["SepalWidthCm"], 25, interpolation = "midpoint")
Q3 = np.percentile(df["SepalWidthCm"], 75, interpolation = "midpoint")
IQR = Q3 - Q1 
print("IQR: ", IQR)
print("Before removing outliers: ", df.shape)

## get upper and lower range ..

upper = np.where(df["SepalWidthCm"] >= (Q3 + 1.5 * IQR))
lower = np.where(df["SepalWidthCm"] <= (Q1 - 1.5 * IQR))

## Remove the outliers ..

df.drop(upper[0], inplace = True)
df.drop(lower[0], inplace = True)
print("AFter Removing Outliers: ", df.shape)

## Visualise the data after removing outliers ..

sns.boxplot(x = "SepalWidthCm", data = df)
plt.show()



## Check for the remaining cols .. if they have outliers ...

plt.figure(figsize = (10, 10))
plt.subplot(1, 3, 1)
sns.boxplot(x = "SepalLengthCm", data = df)
plt.subplot(1, 3, 2)
sns.boxplot(x = "PetalWidthCm", data = df)
plt.subplot(1, 3, 3)
sns.boxplot(x = "PetalLengthCm", data = df)
plt.show()

## There no outliers for the rem. cols ..



## Summary Statistics for the Iris Dataset ..
num_df = df.select_dtypes(include=[np.number])
summary = {}
summary["Minimum"] = np.min(num_df)
summary["First Quartile"] = np.percentile(num_df, 25)
summary["Median (Q2)"] = np.median(num_df)
summary["Third Quartile"] = np.percentile(num_df, 75)
summary["Maximum"] = np.max(num_df)
summary["Std.Dev."] = np.std(num_df)
print(pd.Series(summary))

'''

