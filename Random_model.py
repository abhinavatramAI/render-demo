# Import Flask and other necessary libraries
from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd
import warnings
import os
from datetime import datetime


warnings.filterwarnings("ignore")

# Initialize Flask app
app = Flask(__name__)

# Load the machine learning model
try:
    with open('VF_RF.sav', 'rb') as model_file:
        model = joblib.load(model_file)
except Exception as e:
    print("Error loading the model:", e)
    model = None


# Risk assessment function
def charge(a):
    if (a >= 0) and (a <= 4740):
        return 'Low Risk Customer'
    elif (a <= 9382):
        return 'Medium Risk Customer'
    elif (a <= 25381):
        return 'High Risk Customer'
    elif (a > 25381):
        return 'Very High Risk Customer'


# Define a function to append input details to an Excel file
# Define a function to append input details to an Excel file
def append_to_excel(age, sex, bmi, children, smoker, region, prediction, risk_message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data = {'Timestamp': [timestamp],
            'Age': [age],
            'Sex': ['Female' if sex == 0 else 'Male'],
            'BMI': [bmi],
            'Children': [children],
            'Smoker': ['No' if smoker == 0 else 'Yes'],
            'Region': [
                'Southwest' if region == 1 else 'Southeast' if region == 2 else 'Northwest' if region == 3 else 'Northeast'],
            'Prediction': [prediction],
            'Risk Message': [risk_message]}

    df = pd.DataFrame(data)
    if os.path.exists('user_inputs.xlsx'):
        existing_data = pd.read_excel('user_inputs.xlsx')
        updated_data = pd.concat([existing_data, df], ignore_index=True)
        updated_data.to_excel('user_inputs.xlsx', index=False)
    else:
        df.to_excel('user_inputs.xlsx', index=False)


# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')


# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('error.html', error_message="Model not found.")

    try:
        # Get input values from the form
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        bmi = int((request.form['bmi']))  # Convert BMI to integer
        children = int(request.form['children'])
        smoker = int(request.form['smoker'])
        region = int(request.form['region'])

        # Prepare input features as numpy array
        features = np.array([[age, sex, bmi, children, smoker, region]])

        # Make prediction
        prediction = int(model.predict(features)[0])

        # Get risk assessment message
        risk_message = charge(prediction)

        # Append input details to Excel file
        append_to_excel(age, sex, bmi, children, smoker, region, prediction, risk_message)

        # Pass input values and prediction results to the result template
        return render_template('predict.html', age=age, sex=sex, bmi=bmi, children=children, smoker=smoker,
                               region=region, prediction=prediction, risk_message=risk_message)
    except Exception as e:
        print("Error during prediction:", e)
        return render_template('error.html', error_message=str(e))


if __name__ == '__main__':
    # Run the app
    app.run(debug=True)
