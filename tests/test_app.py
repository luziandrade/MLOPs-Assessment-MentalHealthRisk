from flaskApp import app

def test_predict_endpoint():
    client = app.test_client()
    response = client.post('/predict', data={
        'sleep_hours': 4.0,
        'exercise_hours': 0.1,
        'screen_time': 10.0,
        'social_interaction': 0.5
    }, follow_redirects=True)
    assert response.status_code == 200
    assert b'High' in response.data or b'Low' in response.data
