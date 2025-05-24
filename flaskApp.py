from flask import Flask, request, jsonify
from model import StressModel

app = Flask(__name__)
model = StressModel()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = [
        data['sleep_hours'],
        data['exercise_hours'],
        data['screen_time'],
        data['social_interaction']
    ]
    prediction = model.predict(features)
    return jsonify({'stress_level': 'High' if prediction == 1 else 'Low'})

if __name__ == "__main__":
    app.run(debug=True)
