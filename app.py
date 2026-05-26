import streamlit as st
import pandas as pd
import joblib

# Load trained pipeline
model = joblib.load("attrition_pipeline.pkl")
  

st.title("Employee Resignation Prediction")

st.write("Fill employee information to predict attrition risk.")

# =========================
# CATEGORICAL FEATURES
# =========================

BusinessTravel = st.selectbox(
    "Business Travel",
    ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel']
)

Department = st.selectbox(
    "Department",
    ['Research & Development', 'Sales', 'Human Resources']
)

EducationField = st.selectbox(
    "Education Field",
    ['Life Sciences', 'Medical', 'Marketing',
     'Technical Degree', 'Other', 'Human Resources']
)

Gender = st.selectbox(
    "Gender",
    ['Male', 'Female']
)

JobRole = st.selectbox(
    "Job Role",
    ['Sales Executive', 'Research Scientist',
     'Laboratory Technician', 'Manufacturing Director',
     'Healthcare Representative', 'Manager',
     'Sales Representative', 'Research Director',
     'Human Resources']
)

MaritalStatus = st.selectbox(
    "Marital Status",
    ['Married', 'Single', 'Divorced']
)

Over18 = st.selectbox(
    "Over 18",
    ['Y']
)

OverTime = st.selectbox(
    "Over Time",
    ['No', 'Yes']
)

# =========================
# NUMERICAL FEATURES
# =========================

Age = st.number_input("Age", min_value=18, max_value=65, value=30)

DailyRate = st.number_input("Daily Rate", value=800)

DistanceFromHome = st.number_input("Distance From Home", value=10)

Education = st.number_input("Education Level", min_value=1, max_value=5, value=3)

EmployeeCount = st.number_input("Employee Count", value=1)

EmployeeNumber = st.number_input("Employee Number", value=1001)

EnvironmentSatisfaction = st.number_input(
    "Environment Satisfaction",
    min_value=1,
    max_value=4,
    value=3
)

HourlyRate = st.number_input("Hourly Rate", value=60)

JobInvolvement = st.number_input(
    "Job Involvement",
    min_value=1,
    max_value=4,
    value=3
)

JobLevel = st.number_input(
    "Job Level",
    min_value=1,
    max_value=5,
    value=2
)

JobSatisfaction = st.number_input(
    "Job Satisfaction",
    min_value=1,
    max_value=4,
    value=3
)

MonthlyIncome = st.number_input("Monthly Income", value=5000)

MonthlyRate = st.number_input("Monthly Rate", value=15000)

NumCompaniesWorked = st.number_input(
    "Number of Companies Worked",
    value=2
)

PercentSalaryHike = st.number_input(
    "Percent Salary Hike",
    value=15
)

PerformanceRating = st.number_input(
    "Performance Rating",
    min_value=1,
    max_value=4,
    value=3
)

RelationshipSatisfaction = st.number_input(
    "Relationship Satisfaction",
    min_value=1,
    max_value=4,
    value=3
)

StandardHours = st.number_input("Standard Hours", value=80)

StockOptionLevel = st.number_input(
    "Stock Option Level",
    min_value=0,
    max_value=3,
    value=1
)

TotalWorkingYears = st.number_input(
    "Total Working Years",
    value=10
)

TrainingTimesLastYear = st.number_input(
    "Training Times Last Year",
    value=2
)

WorkLifeBalance = st.number_input(
    "Work Life Balance",
    min_value=1,
    max_value=4,
    value=3
)

YearsAtCompany = st.number_input(
    "Years At Company",
    value=5
)

YearsInCurrentRole = st.number_input(
    "Years In Current Role",
    value=3
)

YearsSinceLastPromotion = st.number_input(
    "Years Since Last Promotion",
    value=1
)

YearsWithCurrManager = st.number_input(
    "Years With Current Manager",
    value=4
)

# =========================
# BUILD INPUT DATAFRAME
# =========================

input_data = pd.DataFrame([{
    'Age': Age,
    'BusinessTravel': BusinessTravel,
    'DailyRate': DailyRate,
    'Department': Department,
    'DistanceFromHome': DistanceFromHome,
    'Education': Education,
    'EducationField': EducationField,
    'EmployeeCount': EmployeeCount,
    'EmployeeNumber': EmployeeNumber,
    'EnvironmentSatisfaction': EnvironmentSatisfaction,
    'Gender': Gender,
    'HourlyRate': HourlyRate,
    'JobInvolvement': JobInvolvement,
    'JobLevel': JobLevel,
    'JobRole': JobRole,
    'JobSatisfaction': JobSatisfaction,
    'MaritalStatus': MaritalStatus,
    'MonthlyIncome': MonthlyIncome,
    'MonthlyRate': MonthlyRate,
    'NumCompaniesWorked': NumCompaniesWorked,
    'Over18': Over18,
    'OverTime': OverTime,
    'PercentSalaryHike': PercentSalaryHike,
    'PerformanceRating': PerformanceRating,
    'RelationshipSatisfaction': RelationshipSatisfaction,
    'StandardHours': StandardHours,
    'StockOptionLevel': StockOptionLevel,
    'TotalWorkingYears': TotalWorkingYears,
    'TrainingTimesLastYear': TrainingTimesLastYear,
    'WorkLifeBalance': WorkLifeBalance,
    'YearsAtCompany': YearsAtCompany,
    'YearsInCurrentRole': YearsInCurrentRole,
    'YearsSinceLastPromotion': YearsSinceLastPromotion,
    'YearsWithCurrManager': YearsWithCurrManager
}])

# =========================
# PREDICTION
# =========================

if st.button("Predict"):

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1] * 100

    if prediction == 1:
        st.error(
            f"Employee likely to resign ({probability:.2f}%)"
        )
    else:
        st.success(
            f"Employee likely to stay ({100 - probability:.2f}%)"
        )