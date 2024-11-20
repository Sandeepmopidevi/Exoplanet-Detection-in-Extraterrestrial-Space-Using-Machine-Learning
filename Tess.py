import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import time

# Set the title for the Streamlit app
st.markdown(
    """
    <style>
        .centered-text {
            text-align: center;
        }
        .stDataFrame {
            margin-left: auto;
            margin-right: auto;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h1 class='centered-text'>Exoplanet Detection in Extraterrestrial Space Using TESS Data</h1>", unsafe_allow_html=True)

# Check if the file exists
csv_file_path = 'TESS_Project_Candidates_Yet_To_Be_Confirmed.csv'  # Adjust this path as necessary
if not os.path.exists(csv_file_path):
    st.error("File not found. Please check the file path.")
else:
    # Load data from CSV file
    df = pd.read_csv(csv_file_path)

    # Interactive Data Overview
    with st.expander("Explore the Dataset"):
        st.markdown("<h3 class='centered-text'>Dataset Overview</h3>", unsafe_allow_html=True)
        st.write("Preview of the DataFrame:")
        st.dataframe(df.head(), use_container_width=True)

        # Filter by column
        selected_column = st.selectbox("Select a column to view unique values:", df.columns)
        st.write(f"Unique values in {selected_column}:")
        st.write(df[selected_column].unique())

    # Analyze "TESS Disposition" column
    with st.expander("Analyze TESS Disposition Column"):
        st.markdown("<h3 class='centered-text'>TESS Disposition Analysis</h3>", unsafe_allow_html=True)
        unique_values = df['TESS Disposition'].unique()
        st.write("Unique values in TESS Disposition:")
        st.write(unique_values)

        class_counts = df['TESS Disposition'].value_counts()
        st.write("Counts of each class in TESS Disposition:")
        st.bar_chart(class_counts)

    # Feature Engineering
    df['Confirmed'] = df['TESS Disposition'].apply(lambda x: 1 if x in ['KP', 'CP'] else 0)
    df['Candidate'] = df['TESS Disposition'].apply(lambda x: 1 if x == 'PC' else 0)

    with st.expander("Feature Engineering"):
        st.markdown("<h3 class='centered-text'>Engineered Features</h3>", unsafe_allow_html=True)
        st.write("Unique values in Confirmed column:")
        st.write(df['Confirmed'].unique())
        st.write("Unique values in Candidate column:")
        st.write(df['Candidate'].unique())

        st.write("Counts of Confirmed classes:")
        st.bar_chart(df['Confirmed'].value_counts())
        st.write("Counts of Candidate classes:")
        st.bar_chart(df['Candidate'].value_counts())

    # Ensure there is enough data for modeling
    if df['Confirmed'].value_counts().min() == 0:
        st.warning("Only one class detected in the target variable 'Confirmed'. Model cannot train with one class.")
    elif df['Candidate'].value_counts().min() == 0:
        st.warning("Only one class detected in the target variable 'Candidate'. Model cannot train with one class.")
    else:
        # Prepare data for modeling
        X = df.drop(columns=['TESS Disposition', 'Confirmed', 'Candidate'])
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

        y_confirmed = df['Confirmed']
        y_candidate = df['Candidate']

        # Modeling for Confirmed
        st.markdown("<h3 class='centered-text'>Modeling: Confirmed</h3>", unsafe_allow_html=True)
        with st.spinner("Training the model for 'Confirmed'..."):
            progress = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress.progress(i + 1)

            X_train, X_test, y_train, y_test = train_test_split(X, y_confirmed, test_size=0.2, random_state=42)
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

        st.success(f"Model Accuracy for Confirmed: {accuracy:.2f}")

        # Modeling for Candidate
        st.markdown("<h3 class='centered-text'>Modeling: Candidate</h3>", unsafe_allow_html=True)
        with st.spinner("Training the model for 'Candidate'..."):
            progress = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress.progress(i + 1)

            X_train, X_test, y_train, y_test = train_test_split(X, y_candidate, test_size=0.2, random_state=42)
            model_candidate = RandomForestClassifier()
            model_candidate.fit(X_train, y_train)
            y_pred_candidate = model_candidate.predict(X_test)
            accuracy_candidate = accuracy_score(y_test, y_pred_candidate)

        st.success(f"Model Accuracy for Candidate: {accuracy_candidate:.2f}")
