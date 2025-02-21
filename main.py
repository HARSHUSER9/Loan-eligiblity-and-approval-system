from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load("loan_approval_model.pkl")
scaler = joblib.load("scaler.pkl")  # Make sure you saved your scaler during training


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input features from the form
        features = [
            float(request.form['ApplicantIncome']),
            float(request.form['CoapplicantIncome']),
            float(request.form['LoanAmount']),
            float(request.form['Loan_Amount_Term']),
            float(request.form['Credit_History']),
            int(request.form['Property_Area']),
            int(request.form['Self_Employed']),
            int(request.form['Married']),
            int(request.form['Education'])
        ]

        # Check if the user enabled premium mode
        is_premium = request.form.get('Premium') == '1'

        # Convert to numpy array and reshape
        input_data = np.array(features).reshape(1, -1)

        # Scale the input data
        scaled_input = scaler.transform(input_data)

        # Predict loan eligibility
        prediction = model.predict(scaled_input)
        loan_status = "Approved" if prediction[0] == 1 else "Rejected"

        # Response object
        response = {"Loan Approval": loan_status}

        if is_premium:
            response["Message"] = "Premium users get exclusive insights!"
            response["Recommended Interest Rate"] = "6.5% (Based on market trends)"
            response["Loan Improvement Tip"] = "Increasing your credit score to 750+ may boost approval chances."

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)})


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
