import unittest
import app
import json

class TestFlaskApi(unittest.TestCase):
    def setUp(self):
        self.app = app.app.test_client()
        self.app.testing = True

    def test_predict(self):
        response = self.app.post('/predict', json={"text":"Apple stock price is going up!"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(response.get_data()), {"ORGS": ["Apple"]})