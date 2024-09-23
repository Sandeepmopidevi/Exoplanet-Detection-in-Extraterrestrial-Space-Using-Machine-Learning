# Importing Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn Packages
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report

# Additional Packages
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

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

# Create new columns
df['ExoplanetCandidate'] = df['DispositionUsingKeplerData'].apply(lambda x: 1 if x == 'CANDIDATE' else 0)
df['ExoplanetConfirmed'] = df['ExoplanetArchiveDisposition'].apply(lambda x: 2 if x == 'CONFIRMED' else 1 if x == 'CANDIDATE' else 0)

# Drop irrelevant columns
df.drop(columns=['KeplerName','KOIName','EquilibriumTemperatureUpperUnc.[K','KepID','ExoplanetArchiveDisposition','DispositionUsingKeplerData',
                 'NotTransit-LikeFalsePositiveFlag','koi_fpflag_ss','CentroidOffsetFalsePositiveFlag','EphemerisMatchIndicatesContaminationFalsePositiveFlag',
                 'TCEDeliver','EquilibriumTemperatureLowerUnc.[K'], inplace=True)

# Drop missing values
df.dropna(inplace=True)

# Clean column names to remove special characters
df.columns = df.columns.str.replace(r"[\[\]<>\.]", "", regex=True)

# Define Features and Target
features = df.drop(columns=['ExoplanetCandidate', 'ExoplanetConfirmed'])
target = df['ExoplanetCandidate']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.4, random_state=1)

# Function for evaluation and visualization
def evaluation(y_true, y_pred, model_name):
    acc = metrics.accuracy_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred)
    
    print(f'\nModel: {model_name}')
    print(f'Accuracy: {acc:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Precision: {precision:.4f}')
    
    # Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Logistic Regression
lr = LogisticRegression(C=100, max_iter=200, class_weight='balanced')
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
evaluation(y_test, y_pred, "Logistic Regression")

# K-Nearest Neighbors
knn = KNeighborsClassifier(leaf_size=8, metric='manhattan', weights='uniform')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
evaluation(y_test, y_pred, "K-Nearest Neighbors")

# Decision Tree Classifier
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
evaluation(y_test, y_pred, "Decision Tree Classifier")

# Random Forest Classifier
forest = RandomForestClassifier(n_estimators=100, criterion='gini')
forest.fit(X_train, y_train)
y_pred = forest.predict(X_test)
evaluation(y_test, y_pred, "Random Forest Classifier")

# Gradient Boosting Classifier
gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)
evaluation(y_test, y_pred, "Gradient Boosting Classifier")

# XGBoost Classifier
xgboost = xgb.XGBClassifier()
xgboost.fit(X_train, y_train)
y_pred = xgboost.predict(X_test)
evaluation(y_test, y_pred, "XGBoost Classifier")

# LightGBM Classifier
lightgbm = lgb.LGBMClassifier(verbose=0)
lightgbm.fit(X_train, y_train)
y_pred = lightgbm.predict(X_test)
evaluation(y_test, y_pred, "LightGBM Classifier")