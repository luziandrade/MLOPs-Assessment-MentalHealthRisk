from model import StressModel

def test_predict_low_stress():
    model = StressModel()
    features = [8.0, 2.0, 2.0, 0.5, 26, 8]
    prediction = model.predict(features)
    # only printing if the test fails
    print(f"Test Low Stress: Input={features} | Prediction={prediction}")
    assert prediction == 0

def test_predict_high_stress():
    model = StressModel()
    features = [4.0, 0.1, 10.0, 0.5, 26, 1.0]
    prediction = model.predict(features)
    # only printing if the test fails
    print(f"Test High Stress: Input={features} | Prediction={prediction}")
    assert prediction == 1
