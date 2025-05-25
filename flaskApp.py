from flask import Flask, render_template, request
from model import StressModel

app = Flask(__name__)
model = StressModel()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    sleep_hours = float(request.form['sleep_hours'])
    exercise_hours = float(request.form['exercise_hours'])
    screen_time = float(request.form['screen_time'])
    social_interaction = float(request.form['social_interaction'])
    age = float(request.form['age'])
    work_hours = float(request.form['work_hours'])
    features = [sleep_hours, exercise_hours, screen_time, social_interaction, age ,work_hours]
    prediction = model.predict(features)
    prediction_text = 'High' if prediction == 1 else 'Low'
    return render_template('index.html', prediction=prediction_text)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
