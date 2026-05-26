# Employee Resignation Prediction App

A Machine Learning web application built with Streamlit that predicts whether an employee is likely to resign based on HR analytics data.

---

# Project Overview

This project uses Machine Learning techniques to analyze employee information and predict employee attrition (resignation risk).

The application was developed using:

- Python
- Scikit-learn
- Imbalanced-learn
- Streamlit
- Data Tree Classifier
- Random Forest Classifier

---

# Dataset

Dataset used:

IBM HR Analytics Employee Attrition Dataset

Main target:
- `Attrition`
  - Yes → Employee likely to resign
  - No → Employee likely to stay

---

# Features Used

## Categorical Features
- BusinessTravel
- Department
- EducationField
- Gender
- JobRole
- MaritalStatus
- Over18
- OverTime

## Numerical Features
- Age
- DailyRate
- DistanceFromHome
- Education
- EnvironmentSatisfaction
- HourlyRate
- JobInvolvement
- JobLevel
- JobSatisfaction
- MonthlyIncome
- MonthlyRate
- NumCompaniesWorked
- PercentSalaryHike
- PerformanceRating
- RelationshipSatisfaction
- StockOptionLevel
- TotalWorkingYears
- TrainingTimesLastYear
- WorkLifeBalance
- YearsAtCompany
- YearsInCurrentRole
- YearsSinceLastPromotion
- YearsWithCurrManager

---

# Machine Learning Pipeline

The pipeline includes:

1. OneHotEncoder for categorical variables
2. SMOTEENN for handling imbalanced data
3. Random Forest Classifier

Pipeline structure:

```python
Pipeline([
    ('preprocess', preprocessor),
    ('smoteenn', SMOTEENN(random_state=42)),
    ('model', RandomForestClassifier())
])