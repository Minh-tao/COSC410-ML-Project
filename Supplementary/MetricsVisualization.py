import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.preprocessing
import sklearn.metrics
import sklearn.svm
import sklearn.tree
import sklearn.neighbors

X = sns.load_dataset("iris")
y = X["species"]
X.drop("species", axis=1, inplace=True)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.20)

# standardize data
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

sklearn.metrics.ConfusionMatrixDisplay.from_estimator(knn, X_test, y_test)

plt.savefig("CM.pdf")