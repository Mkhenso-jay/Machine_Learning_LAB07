# part3_adaboost.py

##% Import Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from utils import save_plot

##% Define Plotting Function
def plot_decision_boundaries(X, y, clfs, titles, filename_prefix):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    for clf, title in zip(clfs, titles):
        clf.fit(X, y)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=0.3)
        plt.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', marker='^', label='Class 0')
        plt.scatter(X[y == 1, 0], X[y == 1, 1], c='green', marker='o', label='Class 1')
        plt.xlabel('Alcohol')
        plt.ylabel('OD280/OD315 of diluted wines')
        plt.title(title)
        plt.legend()
        save_plot(f"{filename_prefix}_{title.lower().replace(' ', '_')}.png")

##% Main Function
def main():
    ##% Load and Preprocess Wine Dataset
    df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
    df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash',
                       'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols',
                       'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
    df_wine = df_wine[df_wine['Class label'] != 1]
    y = df_wine['Class label'].values
    X = df_wine[['Alcohol', 'OD280/OD315 of diluted wines']].values

    # Encode labels and split data
    le = LabelEncoder()
    y = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    ##% Train and Evaluate Decision Tree Stump and AdaBoost
    tree = DecisionTreeClassifier(criterion='entropy', random_state=1, max_depth=1)
    tree.fit(X_train, y_train)
    y_train_pred = tree.predict(X_train)
    y_test_pred = tree.predict(X_test)
    tree_train = accuracy_score(y_train, y_train_pred)
    tree_test = accuracy_score(y_test, y_test_pred)
    print(f'Decision tree stump train/test accuracies: {tree_train:.3f}/{tree_test:.3f}')

    ada = AdaBoostClassifier(estimator=tree, n_estimators=500, learning_rate=0.1, random_state=1)
    ada.fit(X_train, y_train)
    y_train_pred = ada.predict(X_train)
    y_test_pred = ada.predict(X_test)
    ada_train = accuracy_score(y_train, y_train_pred)
    ada_test = accuracy_score(y_test, y_test_pred)
    print(f'AdaBoost train/test accuracies: {ada_train:.3f}/{ada_test:.3f}')

    ##% Visualize Decision Boundaries
    clfs = [tree, ada]
    titles = ['Decision Tree Stump', 'AdaBoost']
    plot_decision_boundaries(X_train, y_train, clfs, titles, "wine_decision_boundaries")

if __name__ == "__main__":
    main()