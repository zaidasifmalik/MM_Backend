from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS from flask_cors
import joblib
import numpy as np
from flask_ngrok import run_with_ngrok  # Import run_with_ngrok

app = Flask(__name__)
CORS(app)  # Enable CORS for all origins

# Load the trained model
model = joblib.load('psl_predictor.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data sent from the Flutter app
        data = request.get_json()

        # Extract features from the data
        wickets = data['wickets']
        balls_left = data['balls_left']
        runs_left = data['runs_left']

        # Preprocess data if necessary (e.g., scaling, one-hot encoding)
        features = np.array([[wickets, balls_left, runs_left]])

        # Make prediction using the model
        prediction = model.predict_proba(features)

        # Interpret prediction
        team1_win_percentage = prediction[0][0] * 100
        team2_win_percentage = prediction[0][1] * 100

        # Return prediction percentages as JSON
        return jsonify({'team1_win_percentage': team1_win_percentage, 'team2_win_percentage': team2_win_percentage})
    except Exception as e:
        # Return error message if prediction fails
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run()
