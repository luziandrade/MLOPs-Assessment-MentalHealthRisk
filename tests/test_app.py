from flaskApp import app

def test_predict_endpoint():
    client = app.test_client()
    response = client.post('/predict', data={
        'age': 25,
        'sleep_hours': 4.0,
        'exercise_hours': 0.1,
        'screen_time': 10.0,
        'social_interaction': 0.5,
        'work_hours': 8.0
    }, follow_redirects=True)
    assert response.status_code == 200
    assert b'High' in response.data or b'Low' in response.data

def test_predict_missing_field():
    from flaskApp import app
    client = app.test_client()
    response = client.post('/predict', data={
        'age': 25,
        'sleep_hours': 6.0,
        # Missing exercise_hours
        'screen_time': 5.0,
        'social_interaction': 2.0,
        # Missing work_hours
    }, follow_redirects=True)
    assert response.status_code == 400 or b'error' in response.data
