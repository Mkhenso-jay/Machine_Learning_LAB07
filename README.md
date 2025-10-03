# Ensemble Learning Lab: Combining Models for Improved Performance

## Description
This lab implements and evaluates ensemble learning methods based on *Chapter 7 of Python Machine Learning (Second Edition) by Sebastian Raschka and Vahid Mirjalili*.  
The lab uses the **Iris** and **Wine** datasets to explore **majority voting, bagging, AdaBoost,** and a comprehensive comparison of classifiers and ensembles.

---

## Analysis Questions and Answers

### Majority Voting vs Individual Classifiers
**Q:** Compare the performance of the majority voting classifier with individual classifiers. Why does the ensemble typically perform better? Are there cases where it might perform worse?

**Performance Comparison:**  
In `part1_iris.py`, the majority voting classifier typically achieves a ROC AUC score of ~0.95 on the Iris dataset (classes 1 and 2, using sepal width and petal length).  
Individual classifiers perform as follows:  
- Logistic Regression: ~0.92  
- Decision Tree (depth=1): ~0.90  
- KNN (k=1): ~0.93  

The majority voting classifier outperforms individual classifiers due to its ability to combine diverse models.

**Why Ensembles Perform Better:**  
The ensemble leverages complementary strengths of the classifiers. Logistic Regression captures linear patterns, KNN excels at local patterns, and the Decision Tree (stump) provides simple splits. By aggregating predictions, the ensemble reduces variance and bias, mitigating individual model weaknesses.

**Cases Where Ensembles Might Perform Worse:**  
- Classifiers are highly correlated (e.g., similar models with redundant errors).  
- One classifier is significantly worse, skewing the majority vote.  
- The dataset is too small or simple, where a single strong classifier (e.g., KNN) suffices.

---

### Bagging Analysis
**Q:** How does changing the number of estimators in bagging affect performance?  
- Increasing the number of estimators (e.g., `n_estimators=500` in `part2_bagging.py`) improves performance by reducing variance.  
- More bootstrap samples create diverse trees, leading to more robust predictions.  
- Bagging achieves a test accuracy of 0.917 compared to 0.833 for a single decision tree.  
- Beyond ~200–500 estimators, additional trees yield diminishing returns due to cost and stabilized predictions.

**Q:** What is the effect of bootstrap sampling vs using the entire dataset?  
- Bootstrap sampling creates diverse training subsets, reducing overfitting.  
- Bagging’s test accuracy (0.917) is higher than the single decision tree’s (0.833), even though both achieve training accuracy of 1.000.  
- Training on the entire dataset leads to overfitting, while bootstrap sampling averages out errors.

**Q:** Why does bagging typically reduce overfitting compared to a single decision tree?  
- A single decision tree overfits (train=1.000, test=0.833).  
- Bagging trains multiple trees on bootstrap samples and averages their predictions, reducing variance.  
- Bagging’s test accuracy (0.917) confirms improved generalization.

---

### AdaBoost Insights
**Q:** How does the learning rate parameter affect AdaBoost's performance and convergence?  
- The learning rate controls the weight of each weak learner’s contribution.  
- A lower rate (e.g., 0.1) requires more estimators but avoids overfitting.  
- A higher rate (e.g., 1.0) speeds up convergence but may overfit noisy data.  
- In `part3_adaboost.py`, a lower rate (~0.1) achieves stable test accuracy (~0.90).

**Q:** Analyze the error convergence plot. Why does the test error sometimes increase after many iterations?  
- Training error decreases steadily to near zero.  
- Test error decreases initially but may increase after ~200–300 iterations.  
- This happens because AdaBoost overemphasizes misclassified samples and outliers, reducing generalization.  

**Q:** What makes decision stumps good base estimators for AdaBoost?  
- Stumps are weak learners with high bias but low variance.  
- Their simplicity prevents overfitting in early iterations.  
- Combining many stumps builds a strong classifier, improving test accuracy (~0.90 vs. ~0.85 for a single stump).

---

### Comparative Performance
**Q:** Which ensemble method performed best on the Iris dataset? Why?  
- In `part4_comparison.py`, Random Forest performs best, with mean accuracy of 0.97.  
- Random Forest combines bagging and feature randomness, producing diverse trees.  
- The Iris dataset’s clear separation allows it to outperform other models:  
  - Logistic Regression: 0.95  
  - KNN: 0.96  
  - Decision Tree: 0.94  
  - Voting Classifier: 0.96  
  - Bagging: 0.95  
  - AdaBoost: 0.94  
  - Majority Voting: 0.96  

**Q:** How does Random Forest relate to bagging?  
- Both use bootstrap sampling to train trees.  
- Random Forest adds randomness by selecting a subset of features at each split.  
- This increases diversity and reduces variance.  
- Random Forest typically outperforms Bagging (~0.97 vs. ~0.95).

**Q:** When would you choose one ensemble method over another?  
- **Majority Voting:** Combine diverse classifiers to leverage complementary strengths (e.g., fraud detection).  
- **Bagging:** Reduce overfitting for high-variance models like deep trees (e.g., image classification).  
- **AdaBoost:** Focus on hard-to-classify samples with weak learners (e.g., text classification).  
- **Random Forest:** General-purpose, balanced performance with minimal tuning (e.g., customer churn prediction).

---

### Practical Considerations
**Q:** What are the computational trade-offs between different ensemble methods?  
- **Majority Voting:** Moderate cost, depends on classifier diversity.  
- **Bagging:** High cost due to many trees, but parallelizable.  
- **AdaBoost:** Sequential and slower, not easily parallelized.  
- **Random Forest:** Slightly more expensive than bagging due to feature randomness, but robust.  

**Q:** How does ensemble size affect the bias-variance tradeoff?  
- Larger ensembles reduce variance but may increase bias if base learners are too weak.  
- Bagging improves test accuracy (0.917 vs. 0.833) by averaging predictions.  
- Smaller ensembles have higher variance but are cheaper to compute.

**Q:** In what real-world scenarios would each ensemble method be most appropriate?  
- **Majority Voting:** Fraud detection.  
- **Bagging:** Image classification.  
- **AdaBoost:** Text classification.  
- **Random Forest:** Customer churn prediction.

---
