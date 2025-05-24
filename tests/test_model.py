from model import StressModel

def test_predict_low_stress():
    model = StressModel()
    features = [8.0, 2.0, 2.0, 4.0]
    assert model.predict(features) == 0

def test_predict_high_stress():
    model = StressModel()
    features = [4.0, 0.1, 10.0, 0.5]
    assert model.predict(features) == 1
