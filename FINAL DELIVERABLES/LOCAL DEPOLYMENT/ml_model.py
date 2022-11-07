# Importing packages

import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
# Loading the dataset

train_data = pd.read_csv("loan_prediction.csv")
data = train_data.copy()
data = data.rename(columns={'Self_Employed': 'Employment', 'Married': 'Marital_Status'})

data.drop(columns=['Loan_ID'], inplace=True)

# Replace the outliers

q1 = data.ApplicantIncome.quantile(0.27)
q3 = data.ApplicantIncome.quantile(0.49)
IQR = q3 - q1
upper_lim = q3 + 1.5 * IQR
lower_lim = q1 - 1.5 * IQR
data.ApplicantIncome = np.where(data.ApplicantIncome > upper_lim, data.ApplicantIncome.median(), data.ApplicantIncome)
data.ApplicantIncome = np.where(data.ApplicantIncome < lower_lim, data.ApplicantIncome.median(), data.ApplicantIncome)

q1 = data.CoapplicantIncome.quantile(0.25)
q3 = data.CoapplicantIncome.quantile(0.71)
IQR = q3 - q1
upper_lim = q3 + 1.5 * IQR
data.CoapplicantIncome = np.where(data.CoapplicantIncome > upper_lim, data.CoapplicantIncome.median(),
                                  data.CoapplicantIncome)

q1 = data.LoanAmount.quantile(0.20)
q3 = data.LoanAmount.quantile(0.60)
data.LoanAmount = np.where(data.LoanAmount > q3, data.LoanAmount.median(), data.LoanAmount)
data.LoanAmount = np.where(data.LoanAmount < q1, data.LoanAmount.median(), data.LoanAmount)

#Handling null values

data.Gender.fillna(data.Gender.mode()[0],inplace=True)
data.Marital_Status.fillna(data.Marital_Status.mode()[0],inplace=True)
data.Dependents.fillna(data.Dependents.mode()[0],inplace=True)
data.Employment.fillna(data.Employment.mode()[0],inplace=True)
data.LoanAmount.fillna(value=data.LoanAmount.median(),inplace=True)
data.Loan_Amount_Term.fillna(data.Loan_Amount_Term.median(),inplace=True)
data.Credit_History.fillna(data.Credit_History.mode()[0],inplace=True)


# Perform encoding on categorical columns

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
data.Gender = le.fit_transform(data.Gender)
data.Marital_Status = le.fit_transform(data.Marital_Status)
data.Dependents = le.fit_transform(data.Dependents)
data.Education = le.fit_transform(data.Education)
data.Employment = le.fit_transform(data.Employment)
data.Property_Area = le.fit_transform(data.Property_Area)
data.Loan_Status = le.fit_transform(data.Loan_Status)

data.Loan_Status = data.Loan_Status.map({1: 'Y', 0: 'N'})

# Split the data into dependent and independent variables

x = data.drop(columns=['Loan_Status'])
y = data.Loan_Status

# Scale the independent variables

from sklearn.preprocessing import scale

x_scaled = pd.DataFrame(scale(x), columns=x.columns)

# Split the data into training and testing

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.3, random_state=3)

# Build the model

"""Random Forest classification algorithm"""

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100,random_state=4,max_depth=5)
classifier.fit(x_train,y_train)
y_pred=classifier.predict([[1,1,3,0,1,1234,2345,23414,360,1,2]])
print(y_pred)
# Creating pickle file

pickle.dump(classifier, open("ml_model.pkl", "wb"))
