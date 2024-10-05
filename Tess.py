import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import streamlit as st

# Check if the file exists
csv_file_path = 'TESS_Project_Candidates_Yet_To_Be_Confirmed.csv'  # Adjust this path as necessary
if not os.path.exists(csv_file_path):
    st.error("File not found. Please check the file path.")
else:
    # Load data from CSV file
    df = pd.read_csv(csv_file_path)

    # Display the DataFrame
    st.write("DataFrame:")
    st.dataframe(df.head())  # Show the first 5 rows in the Streamlit app

    # Check unique values in the TESS Disposition column
    unique_values = df['TESS Disposition'].unique()
    st.write("Unique values in TESS Disposition:")
    st.write(unique_values)

    # Count occurrences of each class
    class_counts = df['TESS Disposition'].value_counts()
    st.write("Counts of each class in TESS Disposition:")
    st.write(class_counts)

    # Create Confirmed and Candidate columns
    df['Confirmed'] = df['TESS Disposition'].apply(lambda x: 1 if x in ['KP', 'CP'] else 0)
    df['Candidate'] = df['TESS Disposition'].apply(lambda x: 1 if x == 'PC' else 0)

    # Display unique values in the new columns
    st.write("Unique values in Confirmed column:")
    st.write(df['Confirmed'].unique())
    st.write("Unique values in Candidate column:")
    st.write(df['Candidate'].unique())

    # Check counts for Confirmed and Candidate
    st.write("Counts of Confirmed classes:")
    st.write(df['Confirmed'].value_counts())
    st.write("Counts of Candidate classes:")
    st.write(df['Candidate'].value_counts())

    # Check if we have enough data for model training
    if df['Confirmed'].value_counts().min() == 0:
        st.warning("Only one class detected in the target variable 'Confirmed'. Model cannot train with one class.")
    elif df['Candidate'].value_counts().min() == 0:
        st.warning("Only one class detected in the target variable 'Candidate'. Model cannot train with one class.")
    else:
        # Prepare data for modeling
        # Drop non-numeric and unnecessary columns
        X = df.drop(columns=['TESS Disposition', 'Confirmed', 'Candidate'])

        # Check data types of the features
        st.write("Data types of features:")
        st.write(X.dtypes)

        # Convert categorical columns to numeric if necessary
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))  # Encode the categorical columns

        # Ensure all data is numeric
        st.write("DataFrame after encoding categorical variables:")
        st.dataframe(X.head())  # Show the first 5 rows of the modified DataFrame

        # Prepare the target variable
        y = df['Confirmed']  # Target variable (Confirmed)

        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a RandomForest Classifier
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        st.success(f"Model Accuracy for Confirmed: {accuracy:.2f}")

        # Now, let's check the Candidate target variable
        y_candidate = df['Candidate']  # Target variable (Candidate)

        # Split the dataset for Candidate
        X_train_candidate, X_test_candidate, y_train_candidate, y_test_candidate = train_test_split(X, y_candidate, test_size=0.2, random_state=42)

        # Train a RandomForest Classifier for Candidate
        model_candidate = RandomForestClassifier()
        model_candidate.fit(X_train_candidate, y_train_candidate)

        # Make predictions
        y_pred_candidate = model_candidate.predict(X_test_candidate)

        # Calculate accuracy for Candidate
        accuracy_candidate = accuracy_score(y_test_candidate, y_pred_candidate)
        st.success(f"Model Accuracy for Candidate: {accuracy_candidate:.2f}")
