# +
from flask import Flask, request, jsonify, json
import joblib
import numpy as np
import pandas as pd
  
import logging

from custom_functions import feature_engineering, mean_imputation, create_dummies

# -


# Load the trained model
model = joblib.load("trained_model.joblib")


app = Flask(__name__)
app.json_encoder = json.JSONEncoder

# +
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        input_data = request.get_json()

        # Validate input data
        if not isinstance(input_data, (list, dict)):
            raise ValueError("Invalid input data format. Must be a JSON list or object.")

        # Handle single or batch predictions
        if isinstance(input_data, list):
            # Batch prediction
            input_df = pd.DataFrame(input_data)
        else:
            # Single prediction
            input_df = pd.DataFrame([input_data])

        # Formatting the data
        inp_data = feature_engineering(input_df)
        Cate_cols = inp_data.columns[inp_data.dtypes == 'object'].tolist()

        imputed_data = mean_imputation(inp_data, Cate_cols)
        final_data = create_dummies(inp_data, imputed_data, train=False)
        variables = ['x5_saturday', 'x81_July', 'x81_December', 'x31_japan', 'x81_October', 'x5_sunday',
                     'x81_February', 'x31_asia', 'x91', 'x81_May', 'x5_monday', 'x81_September', 'x81_March',
                     'x53', 'x81_November', 'x44', 'x81_June', 'x5_tuesday', 'x81_August', 'x81_January',
                     'x62', 'x31_germany', 'x58', 'x56', 'x82_Male']

        # Make predictions
        predicted_class = model.predict(final_data[variables])
        prediction_probab = 1 / (1 + np.exp(-predicted_class))

        # Prepare response
        response = []
        for i in range(len(prediction_probab)):
            result = {
                'class_probability': float(prediction_probab[i]),
                'predicted_class': int(predicted_class[i]),
                'input_variables': final_data.iloc[i].to_dict()
            }
            response.append(result)

        # Use Python's 'json' module and set appropriate content type
        return jsonify(response), 200, {'Content-Type': 'application/json'}

    except ValueError as ve:
        # Log validation error
        logger.error(f"Validation Error: {ve}")
        return jsonify({"error": str(ve)}), 400  # Bad Request

    except Exception as e:
        # Log any exception that occurs
        logger.error(f"Error: {e}")
        return jsonify({"error": str(e)}), 500  # Internal Server Error


if __name__ == '__main__':
    # Run the Flask app on port 1313
    app.run(port=1313, debug=True)

# +
# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Define the prediction route
# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Get JSON data from the request
#         input_data = request.get_json()

#         # Validate input data
#         if not isinstance(input_data, (list, dict)):
#             raise ValueError("Invalid input data format. Must be a JSON list or object.")

#         # Handle single or batch predictions
#         if isinstance(input_data, list):
#             # Batch prediction
#             input_df = pd.DataFrame(input_data)
#         else:
#             # Single prediction
#             input_df = pd.DataFrame([input_data])

#         # Formatting the data
#         inp_data = feature_engineering(input_df)
#         Cate_cols = inp_data.columns[inp_data.dtypes == 'object'].tolist()

#         imputed_data = mean_imputation(inp_data, Cate_cols)
#         final_data = create_dummies(inp_data, imputed_data, train=False)
#         variables = ['x5_saturday', 'x81_July', 'x81_December', 'x31_japan', 'x81_October', 'x5_sunday',
#                      'x81_February', 'x31_asia', 'x91', 'x81_May', 'x5_monday', 'x81_September', 'x81_March',
#                      'x53', 'x81_November', 'x44', 'x81_June', 'x5_tuesday', 'x81_August', 'x81_January',
#                      'x62', 'x31_germany', 'x58', 'x56', 'x82_Male']

#         # Make predictions
#         predicted_class = model.predict(final_data[variables])
#         prediction_probab = 1 / (1 + np.exp(-predicted_class))

#         # Prepare response
#         response = []
#         for i in range(len(prediction_probab)):
#             result = {
#                 'class_probability': float(prediction_probab[i]),
#                 'predicted_class': int(predicted_class[i]),
#                 'input_variables': final_data.iloc[i].to_dict()
#             }
#             response.append(result)

#         return flask_jsonify(response)

#     except ValueError as ve:
#         # Log validation error
#         logger.error(f"Validation Error: {ve}")
#         return flask_jsonify({"error": str(ve)}), 400  # Bad Request

#     except Exception as e:
#         # Log any exception that occurs
#         logger.error(f"Error: {e}")
#         return flask_jsonify({"error": str(e)}), 500  # Internal Server Error


# if __name__ == '__main__':
#     # Run the Flask app on port 1313
#     app.run(port=1313, debug=True)

# -

if __name__ == '__main__':
    # Run the Flask app on port 1313
#     app.run(port=1313, debug=True)
    app.run(host='0.0.0.0', port=1313, debug=True)
