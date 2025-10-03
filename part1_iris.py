# part1_iris.py

##% Import Libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import warnings
from majority_voting import MajorityVoteClassifier
from utils import save_plot

warnings.filterwarnings('ignore')

##% Main Function
def plot_decision_boundaries(X, y, clf, title, filename):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    clf.fit(X, y)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
    plt.xlabel('Sepal Width')
    plt.ylabel('Petal Length')
    plt.title(title)
    save_plot(filename)

def main():
    ##% Load and Preprocess Iris Dataset
    iris = datasets.load_iris()
    X, y = iris.data[50:, [1, 2]], iris.target[50:]  # Classes 1 and 2, sepal width and petal length

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    ##% Train Individual Classifiers
    clf1 = LogisticRegression(penalty='l2', C=0.001, random_state=1)
    clf2 = DecisionTreeClassifier(max_depth=1, criterion='entropy', random_state=0)
    clf3 = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')

    # Create pipelines for standardization
    pipe1 = Pipeline([['sc', StandardScaler()], ['clf', clf1]])
    pipe3 = Pipeline([['sc', StandardScaler()], ['clf', clf3]])

    # Evaluate individual classifiers
    clf_labels = ['Logistic Regression', 'Decision Tree', 'KNN']
    print('10-fold cross validation:\n')
    for clf, label in zip([pipe1, clf2, pipe3], clf_labels):
        scores = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10, scoring='roc_auc')
        print(f'ROC AUC: {scores.mean():.2f} (+/- {scores.std():.2f}) [{label}]')

    ##% Create and Evaluate Majority Voting Classifier
    mv_clf = MajorityVoteClassifier(classifiers=[pipe1, clf2, pipe3])
    clf_labels += ['Majority Voting']
    all_clf = [pipe1, clf2, pipe3, mv_clf]

    for clf, label in zip(all_clf, clf_labels):
        scores = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10, scoring='roc_auc')
        print(f'ROC AUC: {scores.mean():.2f} (+/- {scores.std():.2f}) [{label}]')

    ##% Plot Decision Boundaries
    for clf, label in zip(all_clf, clf_labels):
        plot_decision_boundaries(X_train, y_train, clf, f'Decision Boundary - {label}', f"{label.lower().replace(' ', '_')}_decision_boundary.png")

if __name__ == "__main__":
    main()