# +
import unittest
from api import app
import requests

class TestPredictionAPI(unittest.TestCase):
    def setUp(self):
        # Start the Flask app in testing mode
        app.testing = True
        self.app = app.test_client()

    def test_predict_endpoint(self):
        # Test the /predict endpoint with a sample JSON data
        sample_data = [{"x0":-0.065676,"x1":1.892277,"x2":4.8187408342,"x3":0.6403133273,"x4":1.9445617124,"x5":"friday","x6":0.208718,"x7":73.573314,"x8":4.929132,"x9":0.116004,"x10":14.685447,"x11":-2.358749, "x12":"$-4446.82", "x13":0.147213,"x14":10.495835,"x15":4.2431,"x16":0.432266,"x17":14.724543,"x18":-20.363074,"x19":0.535574,"x20":0.007602,"x21":3.547267,"x22":0.076528,"x23":3.705181,"x24":-97.753393,"x25":1.456519,"x26":0.592102,"x27":0.444456,"x28":14.833636,"x29":-2.140143,"x30":5.763837,"x31":"germany","x32":0.93468745,"x33":2.83856379,"x34":-1.26750059,"x35":0.17889836,"x36":1.95062105,"x37":3.12564016,"x38":-1.56780217,"x39":1.13668067,"x40":-1242.59,"x41":-645.34,"x42":1356.52,"x43":-806.75,"x44":0.26568817,"x45":0.11807431,"x46":-0.0617344591,"x47":-0.3463514448,"x48":0.2094155516,"x49":-1.5219293403,"x50":-0.9714326645,"x51":-0.39976832,"x52":0.9454604875,"x53":0.49824967,"x54":0.20915577,"x55":1.06980965,"x56":0.763710638,"x57":0.3624300861,"x58":-0.2548926661,"x59":0.3033352861,"x60":-3.1130897434,"x61":0.4084773658,"x62":1.8569750752,"x63":"78.38%","x64":9.46745541,"x65":3.62652714,"x66":0.32696533,"x67":14.75967038,"x68":-19.85977388,"x69":0.33548682,"x70":0.22087097,"x71":-4.96935696,"x72":0.32417187,"x73":3.19503611,"x74":-107.665301,"x75":0.5734412,"x76":1.90875842,"x77":0.63871216,"x78":None,"x79":-2.02714256,"x80":6.08478887,"x81":"October","x82":"Male","x83":0.3744631949,"x84":0.6821907509,"x85":1.0067103199,"x86":1.3175822827,"x87":-1.4605654149,"x88":-0.799771886,"x89":0.2210406342,"x90":0.3052426128,"x91":-0.099213,"x92":0.71223354,"x93":3.85348867,"x94":-91.6500528,"x95":0.4998613,"x96":2.80435772,"x97":0.62792117,"x98":-32.19004297,"x99":103.19259743}]
        response = self.app.post('/predict', json=sample_data)

        # Check the response status code
        self.assertEqual(response.status_code, 200)

        try:
            # Check if the response is an error
            data = response.get_json()
            if "error" in data:
                self.fail(f"Received an error response: {data['error']}")
        
            # Check the response format
            self.assertIsInstance(data, list)
            self.assertTrue(all(isinstance(entry, dict) for entry in data))
            # Check for the presence of expected keys in the response
            expected_keys = ['class_probability', 'predicted_class', 'input_variables']
            for entry in data:
                for key in expected_keys:
                    self.assertIn(key, entry)

        # Add more specific assertions based on your expected response format
        except ValueError:
            self.fail("Failed to parse response JSON")

        

    def test_predict_endpoint_invalid_input(self):
        # Test the /predict endpoint with invalid input
        invalid_data = "This is not a valid input"
        response = self.app.post('/predict', json=invalid_data)

        # Check the response status code
        self.assertEqual(response.status_code, 400)

        # Check the response format and content
        data = response.get_json()
        self.assertIn("error", data)
        self.assertIsInstance(data["error"], str)

    def test_predict_endpoint_missing_data(self):
        # Test the /predict endpoint with missing data
        missing_data = [{"x0": -0.065676, "x1": 1.892277, "x2": 4.8187408342}]  # Incomplete data
        response = self.app.post('/predict', json=missing_data)

        # Check the response status code
        self.assertEqual(response.status_code, 500)

        # Check the response format and content
        data = response.get_json()
        self.assertIn("error", data)
        self.assertIsInstance(data["error"], str)

if __name__ == '__main__':
    unittest.main()

