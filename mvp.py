# Importing Libraries
import pandas as pd
import numpy as np
import streamlit as st
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load Dataset
df = pd.read_csv('Kepler_Project_Candidates_Yet_To_Be_Confirmed.csv')

# Rename columns
df = df.rename(columns={'kepid':'KepID','kepoi_name':'KOIName','kepler_name':'KeplerName', 'koi_disposition':'ExoplanetArchiveDisposition',
                        'koi_pdisposition':'DispositionUsingKeplerData','koi_score':'DispositionScore','koi_fpflag_nt':'NotTransit-LikeFalsePositiveFlag',
                        'koi_fpflag_ss':'koi_fpflag_ss','koi_fpflag_co':'CentroidOffsetFalsePositiveFlag','koi_fpflag_ec':'EphemerisMatchIndicatesContaminationFalsePositiveFlag',
                        'koi_period':'OrbitalPeriod[days','koi_period_err1':'OrbitalPeriodUpperUnc.[days','koi_period_err2':'OrbitalPeriodLowerUnc.[days',
                        'koi_time0bk':'TransitEpoch[BKJD','koi_time0bk_err1':'TransitEpochUpperUnc.[BKJD','koi_time0bk_err2':'TransitEpochLowerUnc.[BKJD',
                        'koi_impact':'ImpactParameter','koi_impact_err1':'ImpactParameterUpperUnc','koi_impact_err2':'ImpactParameterLowerUnc',
                        'koi_duration':'TransitDuration[hrs','koi_duration_err1':'TransitDurationUpperUnc.[hrs','koi_duration_err2':'TransitDurationLowerUnc.[hrs',
                        'koi_depth':'TransitDepth[ppm','koi_depth_err1':'TransitDepthUpperUnc.[ppm','koi_depth_err2':'TransitDepthLowerUnc.[ppm',
                        'koi_prad':'PlanetaryRadius[Earthradii','koi_prad_err1':'PlanetaryRadiusUpperUnc.[Earthradii','koi_prad_err2':'PlanetaryRadiusLowerUnc.[Earthradii',
                        'koi_teq':'EquilibriumTemperature[K','koi_teq_err1':'EquilibriumTemperatureUpperUnc.[K','koi_teq_err2':'EquilibriumTemperatureLowerUnc.[K',
                        'koi_insol':'InsolationFlux[Earthflux','koi_insol_err1':'InsolationFluxUpperUnc.[Earthflux','koi_insol_err2':'InsolationFluxLowerUnc.[Earthflux',
                        'koi_model_snr':'TransitSignal-to-Noise','koi_tce_plnt_num':'TCEPlanetNumber','koi_tce_delivname':'TCEDeliver','koi_steff':'StellarEffectiveTemperature[K',
                        'koi_steff_err1':'StellarEffectiveTemperatureUpperUnc.[K','koi_steff_err2':'StellarEffectiveTemperatureLowerUnc.[K','koi_slogg':'StellarSurfaceGravity[log10(cm/s**2)',
                        'koi_slogg_err1':'StellarSurfaceGravityUpperUnc.[log10(cm/s**2)','koi_slogg_err2':'StellarSurfaceGravityLowerUnc.[log10(cm/s**2)','koi_srad':'StellarRadius[Solarradii',
                        'koi_srad_err1':'StellarRadiusUpperUnc.[Solarradii','koi_srad_err2':'StellarRadiusLowerUnc.[Solarradii','ra':'RA[decimaldegrees','dec':'Dec[decimaldegrees',
                        'koi_kepmag':'Kepler-band[mag]'})

# Create target column
df['ExoplanetCandidate'] = df['DispositionUsingKeplerData'].apply(lambda x: 1 if x == 'CANDIDATE' else 0)

# Drop unnecessary columns
df.drop(columns=['KeplerName', 'KOIName', 'KepID', 'ExoplanetArchiveDisposition', 'DispositionUsingKeplerData'], inplace=True)

# Handle missing values (numeric columns only)
numeric_columns = df.select_dtypes(include=[np.number]).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

# Handle non-numeric columns by encoding them
non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
for col in non_numeric_columns:
    label_encoder = LabelEncoder()
    df[col] = label_encoder.fit_transform(df[col].astype(str))

# Verify dataset size
print(f"Dataset shape after preprocessing: {df.shape}")

# Define features and target
X = df.drop(columns=['ExoplanetCandidate'])
y = df['ExoplanetCandidate']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Train Random Forest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'exoplanet_model.pkl')

# Streamlit MVP
st.title("Exoplanet Detection MVP")
st.write("Enter the parameters below to check if it's an exoplanet.")

# Input fields for user
input_data = {}
for column in X.columns:
    input_data[column] = st.number_input(f"{column}", value=0.0)

# Prediction Button
if st.button("Predict"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    result = "Exoplanet Candidate" if prediction[0] == 1 else "Not an Exoplanet"
    st.write(f"Prediction: **{result}**")

# Evaluation of Model
y_pred = model.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy: {accuracy:.4f}")

# Confusion Matrix Visualization
st.write("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', ax=ax)
st.pyplot(fig)
