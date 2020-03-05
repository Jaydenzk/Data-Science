from flask import Flask, jsonify, request, render_template
import os
from joblib import load
import pandas as pd
import pickle
# from .tokenizer import token
# from .wrangle import wrangle

"""
Airbnb API using XGB Model.
"""

# create app:


def create_app():
    app = Flask(__name__)

    # load pipeline pickle:
    model = pickle.load(open('xgb_reg_fourteen_features1.pickle', 'rb'))

    @app.route('/', methods=['GET'])
    def prediction():
        """
        Receives data in JSON format, creates dataframe with data,
        runs through predictive model, returns predicted price as JSON object.
        """

        # Receive data:
        listings = request.get_json(force=True)

        # Features used in predictive model:
        accommodates = listings["accommodates"]
        bathrooms = listings["bathrooms"]
        bedrooms = listings["bedrooms"]
        minimum_nights = listings["minimum_nights"]
        maximum_nights = listings["maximum_nights"]
        instant_bookable = listings["instant_bookable"]

        features = {'accommodates': accommodates,
                    'bathrooms': bathrooms,
                    'bedrooms': bedrooms,
                    'minimum_nights': minimum_nights,
                    'maximum_nights': maximum_nights,
                    'instant_bookable': instant_bookable}

        # Convert data into DataFrame:
        df = pd.DataFrame(listings, index=[1])

        # Make prediction for optimal price:
        prediction = str(model.predict(df))

        # Return JSON object:
        return jsonify(output)

    return app


if __name__ == "__main__":
    app.run(debug=True)
