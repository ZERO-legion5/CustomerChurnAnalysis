import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
import joblib

# Load data
data = pd.read_csv("churn.csv")
data_clean = data[data.columns[3:]].copy() # Remove RowNumber, CustomerId, Surname

# Preprocessing for Neural Network
for x in ["Geography", "Gender"]:
    le = LabelEncoder()
    data_clean[x] = le.fit_transform(data_clean[x])

x = data_clean.drop(columns=['Exited'])
y = data_clean['Exited']

scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)

# Save the scaler
joblib.dump(scaler, 'scaler.joblib')

# Split data
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=0)

# Apply SMOTE
smote = SMOTE(random_state=0)
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)

# Save background data for SHAP
background_data = x_train_res[np.random.choice(x_train_res.shape[0], 100, replace=False)]
joblib.dump(background_data, 'background_data.joblib')

# Build ANN Model
model = Sequential()
model.add(Dense(units=32, activation='relu', input_shape=(10,)))
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=32, activation='tanh'))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(x_train_res, y_train_res, batch_size=32, epochs=25, verbose=0)
model.save('ann_model.h5')

# --- NEW: Persona Clustering (Unsupervised) ---
# We use key features for clustering personas
cluster_features = ['Age', 'Balance', 'CreditScore', 'EstimatedSalary']
x_cluster = data_clean[cluster_features]
cluster_scaler = MinMaxScaler()
x_cluster_scaled = cluster_scaler.fit_transform(x_cluster)

kmeans = KMeans(n_clusters=4, random_state=0, n_init=10)
kmeans.fit(x_cluster_scaled)

# Save cluster model and its specific scaler
joblib.dump(kmeans, 'cluster_model.joblib')
joblib.dump(cluster_scaler, 'cluster_scaler.joblib')

print("Model, Scaler, and Clusters exported successfully!")
