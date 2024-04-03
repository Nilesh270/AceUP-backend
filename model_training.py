import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score  
import pickle

def load_models():
    # Load data
    data = pd.read_csv("bca_student_data.csv")

    # Preprocessing (assuming 'Name', 'Gender', and 'Semester' are non-predictive)
    X = data.drop(columns=['Name', 'Gender', 'Semester','Placement Chances (%)'])
    subjects = X.columns.tolist()

    # Train individual models for each subject
    models = {}
    for subject in subjects:
        y = data[subject]  # Use the specified target variable

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a Random Forest model here (adjust parameters as needed)
        model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train, y_train)

        # Make predictions on the testing set
        y_pred = model.predict(X_test)

        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print("Subject:", subject)
        print("Mean Squared Error:", mse)
        print("R-squared Score:", r2)

        # Save the model
        model_path = f"models/model_{subject}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"Model saved to: {model_path}")

        models[subject] = model  # Store the model in the dictionary

    print("Models trained and saved successfully!")

    return models

# Call the function to load and save models
load_models()
