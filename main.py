import pandas as pd
from nasa_model.predict import make_prediction

input_dict = {'name': ["msfadmfas"], "est_diameter_min": [1.1982708007], "est_diameter_max": [2.67941], "relative_velocity": [13569], "miss_distance": [54839744], "orbiting_body": ["Earth"], "sentry_object": [0], "absolute_magnitude": [16.73], 'id': [3]}

result = make_prediction(input_data=input_dict)
print(result)