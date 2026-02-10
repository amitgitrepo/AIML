# Machine Learning Classification Models - Comparative Analysis

## Project Overview

This project implements and compares six different machine learning classification algorithms on a single dataset. The goal is to evaluate and compare the performance of traditional and ensemble learning methods using comprehensive evaluation metrics.

## Dataset

### Bank Marketing Dataset (with Social/Economic Context)

This project uses the Bank Marketing dataset from the UCI Machine Learning Repository, which is based on direct marketing campaigns (phone calls) of a Portuguese banking institution.

- **Source:** UCI Machine Learning Repository
- **Citation:** Moro et al., 2014 - "A Data-Driven Approach to Predict the Success of Bank Telemarketing"
- **Dataset Link:** http://archive.ics.uci.edu/ml/datasets/Bank+Marketing
- **Size:** 41,188 instances with 20 input features + 1 output variable
- **Target Variable:** Binary classification - Will the client subscribe to a bank term deposit? (yes/no)
- **Class Distribution:** Binary (imbalanced dataset)
- **Time Period:** May 2008 to November 2010

### Feature Categories

#### Bank Client Data (7 features)
1. **age** - Age of the client (numeric)
2. **job** - Type of job (categorical: admin, blue-collar, entrepreneur, housemaid, management, retired, self-employed, services, student, technician, unemployed, unknown)
3. **marital** - Marital status (categorical: divorced, married, single, unknown)
4. **education** - Education level (categorical: basic.4y, basic.6y, basic.9y, high.school, illiterate, professional.course, university.degree, unknown)
5. **default** - Has credit in default? (categorical: no, yes, unknown)
6. **housing** - Has housing loan? (categorical: no, yes, unknown)
7. **loan** - Has personal loan? (categorical: no, yes, unknown)

#### Last Contact Information (4 features)
8. **contact** - Contact communication type (categorical: cellular, telephone)
9. **month** - Last contact month of year (categorical: jan, feb, mar, ..., nov, dec)
10. **day_of_week** - Last contact day of the week (categorical: mon, tue, wed, thu, fri)
11. **duration** - Last contact duration in seconds (numeric)
    - *Note:* Highly affects output target; not known before call is performed. Use only for benchmark purposes.

#### Campaign Information (4 features)
12. **campaign** - Number of contacts performed during this campaign (numeric)
13. **pdays** - Days since client was last contacted from previous campaign (numeric; 999 = not previously contacted)
14. **previous** - Number of contacts performed before this campaign (numeric)
15. **poutcome** - Outcome of previous marketing campaign (categorical: failure, nonexistent, success)

#### Social and Economic Context (5 features)
16. **emp.var.rate** - Employment variation rate - quarterly indicator (numeric)
17. **cons.price.idx** - Consumer price index - monthly indicator (numeric)
18. **cons.conf.idx** - Consumer confidence index - monthly indicator (numeric)
19. **euribor3m** - Euribor 3 month rate - daily indicator (numeric)
20. **nr.employed** - Number of employees - quarterly indicator (numeric)

### Data Characteristics
- **Missing Values:** Several categorical attributes contain "unknown" labels
- **Imbalanced Classes:** The target variable is imbalanced (more "no" than "yes" responses)
- **Mixed Data Types:** Combination of numerical and categorical features
- **Temporal Nature:** Data ordered by date, allowing for time-based analysis

## Implemented Models

This project implements the following six classification algorithms:

### 1. Logistic Regression
A linear model for binary and multiclass classification that estimates probabilities using a logistic function.

### 2. Decision Tree Classifier
A tree-structured classifier that makes decisions based on feature values through a series of questions.

### 3. K-Nearest Neighbors (KNN) Classifier
An instance-based learning algorithm that classifies samples based on the majority class of their k nearest neighbors.

### 4. Naive Bayes Classifier
A probabilistic classifier based on Bayes' theorem with strong independence assumptions between features.
- **Variant Used:** Gaussian Naive Bayes (suitable for continuous numerical features)

### 5. Random Forest (Ensemble)
An ensemble method that combines multiple decision trees using bootstrap aggregating (bagging) to improve accuracy and reduce overfitting.

### 6. XGBoost (Ensemble)
An optimized gradient boosting framework that builds trees sequentially, with each tree correcting errors from previous ones.

## Evaluation Metrics

Each model is evaluated using the following six metrics:

| Metric | Description | Range |
|--------|-------------|-------|
| **Accuracy** | Ratio of correct predictions to total predictions | 0 to 1 |
| **AUC Score** | Area Under the ROC Curve - measures model's ability to distinguish between classes | 0 to 1 |
| **Precision** | Ratio of true positives to all positive predictions | 0 to 1 |
| **Recall** | Ratio of true positives to all actual positives (sensitivity) | 0 to 1 |
| **F1 Score** | Harmonic mean of precision and recall | 0 to 1 |
| **MCC Score** | Matthews Correlation Coefficient - balanced measure for imbalanced datasets | -1 to 1 |

## Project Structure

```
├── data/
│   └── bank.csv         # Subset dataset (4,119 instances)
├── streamlit_app.py
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

## Installation

### Prerequisites
- Python 3.11 or higher
- pip package manager

### Setup

1. Clone this repository:
```bash
git clone [your-repo-url]
cd [your-project-name]
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Required Libraries

```
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
jupyter
```

## Usage

### Running the Complete Pipeline

```python
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load data
data = pd.read_csv('data/bank-additional-full.csv', sep=';')

# Separate features and target
X = data.drop('y', axis=1)
y = data['y']

# Encode target variable (yes/no to 1/0)
le = LabelEncoder()
y = le.fit_transform(y)

# Handle categorical variables
categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 
                    'loan', 'contact', 'month', 'day_of_week', 'poutcome']
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and evaluate models
from src.models import train_all_models
from src.evaluation import evaluate_all_models

models = train_all_models(X_train_scaled, y_train)
results = evaluate_all_models(models, X_test_scaled, y_test)
```

### Individual Model Training

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Example: Training Logistic Regression
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)
```

## Results

### Model Performance Comparison

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|-------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.909 | 0.686 | 0.897 | 0.909 | 0.900 | 0.451 |
| Decision Tree | 0.901 | 0.716 | 0.897 | 0.901 | 0.899 | 0.450 |
| K-Nearest Neighbors | 0.892 | 0.671 | 0.883 | 0.892 | 0.887 | 0.444 |
| Naive Bayes | 0.847 | 0.733 | 0.885 | 0.847 | 0.862 | 0.433 |
| Random Forest | - | - | - | - | - | - |
| XGBoost | 0.904 | 0.741 | 0.903 | 0.904 | 0.903 | 0.452 |

*[Fill in your actual results after running the models]*

### Key Findings

- **Best Overall Model:** [Model name based on your results]
- **Best for Precision:** [Model name]
- **Best for Recall:** [Model name]
- **Best for Imbalanced Data (MCC):** [Model name]

## Methodology

### Data Preprocessing
1. **Loading Data:** Read CSV file with semicolon delimiter
2. **Target Encoding:** Convert binary target ('yes'/'no') to (1/0)
3. **Handling Missing Values:** Treat 'unknown' labels (either as separate category or impute)
4. **Encoding Categorical Variables:** One-hot encoding for categorical features
5. **Feature Scaling:** StandardScaler for numerical features
6. **Train-Test Split:** 80-20 ratio with stratification to maintain class distribution
7. **Duration Feature:** Optionally excluded for realistic predictive modeling (see note in dataset description)

### Model Training
- All models trained on the same training dataset
- Hyperparameters tuned using cross-validation (where applicable)
- Random state set to 42 for reproducibility
- Stratified sampling used due to class imbalance

### Evaluation
- All metrics calculated on the same test dataset
- Special attention to class imbalance using MCC and F1 scores
- ROC-AUC used to evaluate probability predictions
- Confusion matrices generated for each model

## Visualizations

The project includes the following visualizations:
- Model comparison bar charts
- ROC curves for all models
- Confusion matrices
- Feature importance plots (for tree-based models)
- Precision-Recall curves

## Hyperparameter Tuning

[Optional section - describe any hyperparameter tuning performed]

## Observations and Insights

### Expected Challenges with This Dataset

1. **Class Imbalance:** The dataset is imbalanced with more "no" responses than "yes"
   - MCC and F1 scores will be particularly important metrics
   - May need to apply techniques like SMOTE or class weighting

2. **Mixed Feature Types:** Combination of numerical and categorical features
   - Proper encoding and scaling are critical
   - Tree-based models may handle mixed types better

3. **Duration Feature:** Highly predictive but not available before the call
   - Should be excluded for realistic modeling
   - Including it may lead to unrealistically high performance

4. **Economic Context Features:** Social and economic indicators may have strong predictive power
   - These features make the dataset unique and valuable

### Key Questions to Explore

- Which features are most important for prediction?
- How do ensemble methods compare to simpler models on this imbalanced dataset?
- Does the inclusion of social/economic features significantly improve performance?
- Which model best balances precision and recall for marketing campaign optimization?

## Future Work

### Model Improvements
- Implement class balancing techniques (SMOTE, class weights, undersampling)
- Extensive hyperparameter optimization using GridSearchCV or RandomizedSearchCV
- Implement additional ensemble methods (AdaBoost, Gradient Boosting, Stacking)
- Deep learning models (Neural Networks) for comparison

### Feature Engineering
- Create interaction features between economic indicators
- Time-based features from month and day_of_week
- Aggregate features from campaign history
- Feature selection techniques to identify most important predictors

### Analysis Extensions
- Cost-benefit analysis for different threshold values
- Temporal analysis: Model performance across different time periods
- Feature importance analysis across all models
- Impact analysis of 'duration' feature inclusion vs. exclusion

### Deployment Considerations
- Real-time prediction pipeline without 'duration' feature
- Model interpretability for business stakeholders
- A/B testing framework for campaign optimization

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your license information]

## Contact

[Your contact information]

## Acknowledgments

### Dataset Citation
```
S. Moro, P. Cortez and P. Rita. 
A Data-Driven Approach to Predict the Success of Bank Telemarketing. 
Decision Support Systems, 2014.
DOI: 10.1016/j.dss.2014.03.001
```

**Dataset Creators:**
- Sérgio Moro (ISCTE-IUL)
- Paulo Cortez (University of Minho)
- Paulo Rita (ISCTE-IUL)

**Data Source:** UCI Machine Learning Repository

**Additional Data:** Social and economic context attributes from Banco de Portugal

### Libraries and Tools
- **scikit-learn:** Machine learning models and evaluation metrics
- **XGBoost:** Gradient boosting implementation
- **pandas & numpy:** Data manipulation and numerical computing
- **matplotlib & seaborn:** Data visualization

---

**Note:** This project is for educational purposes as part of [Course/Project Name].
