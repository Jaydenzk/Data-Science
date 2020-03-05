from flask import Flask, jsonify, request, render_template
import os
from joblib import load
import pandas as pd
import pickle
from flask_cors import CORS
# from .tokenizer import token
# from .wrangle import wrangle

"""
Airbnb API using XGB Model.
"""


def create_app():
    """
    Create and configure the application.
    """
    app = Flask('__name__', instance_relative_config=True)

    CORS(app)  # prevents web errors

    model = pickle.load(open('xgb_reg_fourteen_features1.pickle', 'rb'))
    @app.route('/')
    def root():
        return render_template('base.html', optimal_price="")

    @app.route('/predict', methods=['GET'])
    def predict():

        # defining a dictionary to store data in
        data = {}

        # List of features to use in request
        PARAMETERS = [
            'bedrooms', 'bathrooms', 'accommodates',
            'instant_bookable', 'minimum_nights', 'maximum_nights'
        ]

        AMENITIES = [
            'high_end_electronics', 'high_end_appliances',
            'kitchen_luxury', 'child_friendly', 'privacy',
            'free_parking', 'smoking_allowed', 'pets_allowed'
        ]

        print('\n\nGetting the request data\n\n')

        # load the data
        for param in PARAMETERS:
            print(f'{param} type:', type(request.args[param]))
            print(f'{param}:', request.args[param])
            try:
                data[param] = [int(request.args[param])]
            except:
                data[param] = [request.args[param]]

        print('\n\nAmenities:\n')

        for amenity in AMENITIES:
            if amenity in request.args.keys():

                print(f'{amenity} present! Value: {request.args[amenity]}')
                data[amenity.replace(' ', '_')] = 1
            else:
                data[amenity.replace(' ', '_')] = 0

        for arg in request.args.keys():
            print(f'{arg}: {request.args[arg]}')

        print('\n\nConverting to dataframe\n\n')

        # convert data into dataframe to be passed through the model
        data_df = pd.DataFrame.from_dict(data)

        # more options can be specified also
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print('\n\n', data_df, '\n\n')

        #data_df = wrangle(data_df)

        print('\n\nmaking a prediction\n\n')

        # making a prediction by passing a dataframe through the model
        result = str(model.predict(data_df))
        # create a string response to display
        response = f'The optimal price is {result} Euros'

        print('\n\n' + response + '\n\n')

        # convert the dict to a JSON object and return it
        return result
    return app


if __name__ == "__main__":
    app.run(debug=True)
