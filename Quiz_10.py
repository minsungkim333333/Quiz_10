import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, KFold
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

column_names=['sepal-length','sepal-width','petal-length','petal-width','class']
data=pd.read_csv("./data/09_irisdata.csv", names=column_names)
print(data.shape)
print(data.describe())
print(data.groupby('class').size())
scatter_matrix(data, figsize=(10, 10))
plt.savefig("scatter_matrix.png")
X = data.iloc[:, 0:4].values
y = data.iloc[:, 4].values
model = DecisionTreeClassifier()
model.fit(X, y)
y_pred = model.predict(X)
kfold = KFold(n_splits=10, random_state=10, shuffle=True)
results = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
print(results.mean())
