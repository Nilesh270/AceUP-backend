from flask import Flask, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Define the list of subjects
subjects = ['Programming', 'Data Structures', 'Web Development', 'DBMS', 'Networking']

def load_models():
    models = {}
    for subject in subjects:
        with open(f"models/model_{subject}.pkl", "rb") as f:
            models[subject] = pickle.load(f)
    return models

# Load the trained models
models = load_models()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    # Validate input
    if not all(feature in data for feature in subjects):
        return jsonify({'error': 'Please provide scores for all subjects'}), 400

    # Ensure that the features are ordered and named correctly
    new_student_data = pd.DataFrame(data, index=[0])

    predicted_scores = {}
    for subject, model in models.items():
        predicted_scores[subject] = model.predict(new_student_data)[0]

    return jsonify(predicted_scores)

if __name__ == '__main__':
    app.run(debug=True)
