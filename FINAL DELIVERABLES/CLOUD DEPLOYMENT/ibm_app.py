from flask import Flask, request, render_template
import pickle
import requests
#NOTE: you must manually set API_KEY below using information retrieved from your IBM Cloud account.
API_KEY = "7mtMr5nCl5IrGwBhi0Q-WFgNz6C8WNAtTAQcrM3a3Qt2"
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":
API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

flask_app = Flask(__name__)

@flask_app.route("/")
def Home():
    return render_template("home.html")


@flask_app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        name = request.form['name']
        gender = request.form['gender']
        married = request.form['married']
        dependents = request.form['dependents']
        education = request.form['education']
        employment = request.form['employment']
        applicant_income = request.form['applicant_income']
        coapplicant_income = request.form['coapplicant_income']
        loan_amount = request.form['loan_amount']
        loan_amount_term = request.form['loan_amount_term']
        credit_history = request.form['credit_history']
        prop_area = request.form['prop_area']

        if gender == 'Male':
            gender = 1
        else:
            gender = 0

        if married == 'Yes':
            married = 1
        else:
            married = 0

        if dependents == '0':
            dependents = 0
        elif dependents == '1':
            dependents = 1
        elif dependents == '2':
            dependents = 2
        else:
            dependents = 3

        if education == 'Graduate':
            education = 0
        else:
            education = 1

        if employment == 'Yes':
            employment = 1
        else:
            employment = 0

        if prop_area == 'Rural':
            prop_area = 0
        elif prop_area == 'Semiurban':
            prop_area = 1
        else:
            prop_area = 2

        applicant_income = float(applicant_income)
        coapplicant_income = float(coapplicant_income)
        loan_amount = float(loan_amount)
        loan_amount_term = float(loan_amount_term)

        x = [[gender, married, dependents, education, employment, applicant_income,
            coapplicant_income, loan_amount, loan_amount_term, credit_history, prop_area]]

        #NOTE: manually define and pass the array(s) of values to be scored in the next line
        payload_scoring = {"input_data": [{"fields": [['gender', 'married', 'dependents', 'education', 'employment', 'applicant_income',
                                     'coapplicant_income', 'loan_amount', 'loan_amount_term', 'credit_history', 'prop_area']],
                                     "values": x}]}

        response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/a820ff74-422b-4477-a6ee-929e81de0327/predictions?version=2022-10-28', json=payload_scoring,
        headers={'Authorization': 'Bearer ' + mltoken})
        print("Scoring response")
        print(response_scoring.json())

        if prediction == 'Y':
            prediction = "Congrats {}, You are eligible to apply for loan".format(name)
        else:
            prediction = "Sorry {}, You are ineligible to apply for loan".format(name)

        return render_template('submit.html', prediction_text='{}'.format(prediction))

    else:
        return render_template('predict.html')

if __name__ == "__main__":
    flask_app.run(debug=True)
