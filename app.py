from flask import Flask, request, jsonify
import pandas as pd
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the trained ML model
with open('fraud_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/', methods=['GET'])
def home():
    return "✅ Credit Card Fraud Detection API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON input as DataFrame
        input_data = pd.DataFrame([request.get_json()])

        # Predict class and probability
        prediction = model.predict(input_data)[0]
        confidence = float(model.predict_proba(input_data)[0][prediction])

        # Prepare response
        result = {
            "fraud": bool(prediction),
            "confidence": round(confidence, 3)
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
