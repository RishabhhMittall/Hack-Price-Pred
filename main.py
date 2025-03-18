from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("player_value_predictor.pkl")

# Initialize Flask app
app = Flask(__name__)

# Enable CORS
CORS(app)  # Allow all origins, methods, and headers (for development only)

# Define the expected input fields
expected_fields = [
    "short_name", "long_name", "overall", "potential", "wage_eur", "age", "height_cm",
    "weight_kg", "league_name", "club_name", "club_jersey_number", "nationality_name",
    "skill_moves", "pace", "shooting", "passing", "dribbling", "defending", "physic",
    "bmi", "club_position"
]

# Define a route for predictions
@app.route("/", methods=["POST"])
def predict():
    try:
        # Get JSON data from the request
        player_data = request.get_json()

        # Validate input data
        if not player_data:
            return jsonify({"error": "No input data provided"}), 400

        # Check if all required fields are present
        for field in expected_fields:
            if field not in player_data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        # Convert input data into a DataFrame
        input_data = pd.DataFrame([player_data])

        # Make a prediction
        prediction = model.predict(input_data)

        # Return the prediction
        return jsonify({"predicted_value_eur": prediction[0]})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)