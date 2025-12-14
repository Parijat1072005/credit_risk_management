from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# 1. Load the trained model
model = joblib.load('credit_risk_model.pkl')

# 2. Define the exact top features used in the notebook
TOP_FEATURES = [
    'PAY_TO_BILL_ratio', 
    'percent_fully_paid_months', 
    'pay_0', 
    'age', 
    'pay_amt2', 
    'pay_amt6', 
    'marriage', 
    'Bill_amt1', 
    'pay_amt1', 
    'LIMIT_BAL'
]

# 3. Define the Threshold optimized in your notebook (0.32)
THRESHOLD = 0.32

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        form_data = request.form.to_dict()
        
        # Convert to DataFrame
        df = pd.DataFrame([form_data])
        
        # Convert columns to numeric
        df = df.apply(pd.to_numeric)

        # --- FEATURE ENGINEERING (Replicating your notebook logic) ---
        
        # 1. Calculate percent_fully_paid_months
        # Your notebook logic: (train_data[pay_cols] == -1).sum(axis=1) / len(pay_cols)
        pay_cols = ['pay_0', 'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6']
        
        # We need to ensure these columns exist in the input to calculate the feature
        # (The form handles the inputs, here we calculate)
        df['percent_fully_paid_months'] = (df[pay_cols] == -1).sum(axis=1) / 6.0

        # 2. Calculate PAY_TO_BILL_ratio
        # Since the exact formula wasn't explicit in the snippet (it was loaded from CSV),
        # we calculate it as Total Pay / Total Bill (standard credit risk metric)
        bill_cols = [f'Bill_amt{i}' for i in range(1, 7)]
        pay_amt_cols = [f'pay_amt{i}' for i in range(1, 7)]
        
        total_bill = df[bill_cols].sum(axis=1)
        total_pay = df[pay_amt_cols].sum(axis=1)
        
        # Avoid division by zero
        df['PAY_TO_BILL_ratio'] = total_pay / (total_bill + 1e-6)

        # --- FILTER FOR MODEL INPUT ---
        # Select only the features the model was trained on
        input_data = df[TOP_FEATURES]

        # --- PREDICTION ---
        # Get probability of class 1 (Default)
        prob = model.predict_proba(input_data)[:, 1][0]
        
        # Apply your optimized threshold (0.32)
        prediction = 1 if prob > THRESHOLD else 0

        return render_template('result.html', prediction=prediction, probability=round(prob*100, 2))

    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)