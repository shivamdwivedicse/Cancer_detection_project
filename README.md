# Cancer_detection_project using Machine Learning


## Project Overview
This project predicts whether a Cancer is Malignant or Benign using Machine Learning models.

## Dataset
Provided (cancer_data.csv)

## Models Used
- Logistic Regression
- Random Forest
- SVM
- KNN
- Gradient Boosting

## Hyperparameter Tuning
GridSearchCV used for Logistic Regression.

Best parameter:

C = 0.5

## Threshold Analysis
Different probability thresholds were tested to reduce False Negatives.

## Final Model
LogisticRegression(
C=0.5,
max_iter=2000,
class_weight='balanced'
)

### Final Results

Confusion Matrix:

[[70,1],
 [1,42]]

Cross Validation Mean Score:

94.90%

## Tech Stack
Python, Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
