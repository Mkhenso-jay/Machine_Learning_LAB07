# part4_comparison.py

##% Import Libraries
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import LabelEncoder
from six import iteritems
from majority_voting import MajorityVoteClassifier
from utils import save_plot, save_results
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

##% Custom _name_estimators Function
def _name_estimators(estimators):
    """Generate names for estimators."""
    return [(estimator.__class__.__name__.lower(), estimator) for estimator in estimators]

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
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(title)
        plt.legend()
        save_plot(f"{filename_prefix}_{title.lower().replace(' ', '_')}.png")

##% Implement Majority Voting Classifier
class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):
    """A majority vote ensemble classifier"""
    def __init__(self, classifiers, vote='classlabel', weights=None):
        self.classifiers = classifiers
        self.named_classifiers = {key: value for key, value in _name_estimators(classifiers)}
        self.vote = vote
        self.weights = weights

    def fit(self, X, y):
        self.lablenc_ = LabelEncoder()
        self.lablenc_.fit(y)
        self.classes_ = self.lablenc_.classes_
        self.classifiers_ = []
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X, self.lablenc_.transform(y))
            self.classifiers_.append(fitted_clf)
        return self

    def predict(self, X):
        if self.vote == 'probability':
            maj_vote = np.argmax(self.predict_proba(X), axis=1)
        else:
            predictions = np.asarray([clf.predict(X) for clf in self.classifiers_]).T
            maj_vote = np.apply_along_axis(
                lambda x: np.argmax(np.bincount(x, weights=self.weights)),
                axis=1, arr=predictions)
        maj_vote = self.lablenc_.inverse_transform(maj_vote)
        return maj_vote

    def predict_proba(self, X):
        probas = np.asarray([clf.predict_proba(X) for clf in self.classifiers_])
        avg_proba = np.average(probas, axis=0, weights=self.weights)
        return avg_proba

    def get_params(self, deep=True):
        if not deep:
            return super(MajorityVoteClassifier, self).get_params(deep=False)
        else:
            out = self.named_classifiers.copy()
            for name, step in iteritems(self.named_classifiers):
                for key, value in iteritems(step.get_params(deep=True)):
                    out[f'{name}__{key}'] = value
            return out

##% Main Function
def main():
    ##% Load and Preprocess Iris Dataset
    iris = datasets.load_iris()
    X, y = iris.data[:, [1, 2]], iris.target  # Using sepal width and petal length
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    ##% Create Classifiers
    clf_lr = LogisticRegression(random_state=1)
    clf_knn = KNeighborsClassifier()
    clf_dt = DecisionTreeClassifier(random_state=1)
    clf_rf = RandomForestClassifier(random_state=1, n_estimators=100)
    voting_clf = VotingClassifier(estimators=[('lr', clf_lr), ('knn', clf_knn), ('dt', clf_dt), ('rf', clf_rf)], voting='hard')
    bagging_clf = BaggingClassifier(estimator=clf_dt, n_estimators=100, random_state=1)
    adaboost_clf = AdaBoostClassifier(estimator=clf_dt, n_estimators=100, random_state=1)
    mv_clf = MajorityVoteClassifier(classifiers=[clf_lr, clf_knn, clf_dt, clf_rf])

    classifiers = [clf_lr, clf_knn, clf_dt, clf_rf, voting_clf, bagging_clf, adaboost_clf, mv_clf]
    classifier_names = ['Logistic Regression', 'KNN', 'Decision Tree', 'Random Forest',
                        'Voting Classifier', 'Bagging', 'AdaBoost', 'Majority Voting']

    ##% Evaluate Classifiers
    results = []
    for clf, name in zip(classifiers, classifier_names):
        scores = cross_val_score(clf, X_train, y_train, cv=10, scoring='accuracy')
        results.append({
            'Classifier': name,
            'Mean Accuracy': scores.mean(),
            'Std Accuracy': scores.std()
        })

    results_df = pd.DataFrame(results)
    print(results_df.sort_values('Mean Accuracy', ascending=False))
    save_results(results_df, "classifier_comparison.csv")

    ##% Plot Decision Boundaries
    ensemble_clfs = [clf_dt, clf_rf, bagging_clf, adaboost_clf, mv_clf]  # Focus on ensembles
    ensemble_titles = ['Decision Tree', 'Random Forest', 'Bagging', 'AdaBoost', 'Majority Voting']
    plot_decision_boundaries(X_train, y_train, ensemble_clfs, ensemble_titles, "iris_decision_boundaries")

    ##% Tune AdaBoost Parameters
    param_grid_ada = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.1, 0.5, 1.0],
        'estimator__max_depth': [1, 2, 3]
    }
    ada = AdaBoostClassifier(estimator=DecisionTreeClassifier(random_state=1))
    gs_ada = GridSearchCV(ada, param_grid_ada, cv=5, scoring='accuracy', n_jobs=-1)
    gs_ada.fit(X_train, y_train)
    print(f"AdaBoost Best parameters: {gs_ada.best_params_}")
    print(f"AdaBoost Best score: {gs_ada.best_score_:.3f}")

    ##% Tune Bagging Parameters
    param_grid_bag = {
        'n_estimators': [50, 100, 200],
        'max_samples': [0.5, 0.8, 1.0],
        'max_features': [0.5, 0.8, 1.0]
    }
    bag = BaggingClassifier(estimator=DecisionTreeClassifier(random_state=1), random_state=1)
    gs_bag = GridSearchCV(bag, param_grid_bag, cv=5, scoring='accuracy', n_jobs=-1)
    gs_bag.fit(X_train, y_train)
    print(f"Bagging Best parameters: {gs_bag.best_params_}")
    print(f"Bagging Best score: {gs_bag.best_score_:.3f}")

if __name__ == "__main__":
    main()