from flask import Flask, jsonify, request
from flask_cors import CORS
import pickle
import pandas as pd
import traceback

# Initialize Flask app
app = Flask(__name__)

# Enable CORS to allow requests from other machines/origins
CORS(app, resources={r"/*": {"origins": "*"}})

# Global variable to store the model
model = None
df=pd.read_csv("filtered_dataset.csv")
print(df.columns)
# Load the model only once when the application starts
def load_model():
    global model
    try:
        with open('healthcare_fraud_detection_model.pkl', 'rb') as file:
            model = pickle.load(file)
        print("Model loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

# Load the model at startup
load_model()

@app.route("/predict", methods=["POST"])
def predict():
    global model

    try:
        # Check if model is loaded
        if model is None:
            if not load_model():
                return jsonify({"error": "Model could not be load ed"}), 500

        # Get JSON data from request
        request_data = request.json

        # Ensure the input is a list
        if not isinstance(request_data, list):
            return jsonify({"error": "Invalid data format. Expected a list"}), 400

        # Define the expected feature names (from dataset)
        feature_names = [
            "PerProviderAvg_InscClaimAmtReimbursed",
            "PerProviderAvg_DeductibleAmtPaid",
            "PerProviderAvg_IPAnnualReimbursementAmt",
            "PerProviderAvg_IPAnnualDeductibleAmt",
            "PerProviderAvg_OPAnnualReimbursementAmt",
            "PerProviderAvg_OPAnnualDeductibleAmt",
            "PerProviderAvg_Age",
            "PerProviderAvg_NoOfMonths_PartACov",
            "PerProviderAvg_NoOfMonths_PartBCov",
            "PerProviderAvg_DurationofClaim",
            "PerProviderAvg_NumberofDaysAdmitted",
            "PerAttendingPhysician Avg_InscClaimAmtReimbursed",
            "PerAttendingPhysician Avg_DeductibleAmtPaid",
            "PerAttendingPhysician Avg_IPAnnualReimbursementAmt",
            "PerAttendingPhysician Avg_IPAnnualDeductibleAmt",
            "PerAttendingPhysician Avg_OPAnnualReimbursementAmt",
            "PerAttendingPhysician Avg_OPAnnualDeductibleAmt",
            "PerAttendingPhysician Avg_DurationofClaim"
        ]

        # Ensure input size matches the expected feature count
        if len(request_data) != len(feature_names):
            return jsonify({"error": f"Expected {len(feature_names)} values, got {len(request_data)}"}), 400

        # Convert list into DataFrame with correct column names
        input_data = pd.DataFrame([request_data], columns=feature_names)

        # Make prediction
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)[:, 1].tolist()
        print(prediction)
        # Return the prediction result
        return jsonify({
            "prediction": int(prediction[0]),
            "probability": prediction_proba[0],
            "status": "success"
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/")
def hello_world():
    return "Fraud Detection API"

if __name__ == "__main__":
    print("Starting Fraud Detection API server")
    app.run(host='0.0.0.0', port=5000, debug=True)