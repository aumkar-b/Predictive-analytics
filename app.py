
import pandas as pd
import numpy as np
import io
import base64
import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt


# Load data
# df = pd.read_excel('dataset1.xlsx', engine='openpyxl', usecols='A:M', nrows=37)
df = pd.read_csv('dataset1.csv')

# Sidebar for filtering
st.sidebar.header("Please Filter Here:")
machineid = st.sidebar.multiselect("Select the Machine ID:", options=df["machineid"].unique(), default=df["machineid"].unique())

# Filter data based on selected machine IDs
df_selection = df[df["machineid"].isin(machineid)]

runtime = df_selection['total_machine_runtime'].mean()
runtime = round(runtime / (1000 * 60 * 60), 3)
downtime = df_selection['total_machine_downtime'].mean()
downtime = round(downtime / (1000 * 60 * 60), 3)
machine_availability = df_selection['machine_availability_percent'].mean()
machine_availability = round(machine_availability, 3)
planned_production_time = df_selection['planned_production_time'].mean()
planned_production_time = round(planned_production_time / (1000 * 60 * 60), 3)
total_parts_produced = df_selection['total_parts_produced'].mean()
total_parts_produced = round(total_parts_produced, 0)
oee = df_selection['oee'].mean()
oee = round(oee, 3)
machine_performance_percent = df_selection['machine_performance_percent'].mean()
machine_performance_percent = round(machine_performance_percent, 3)
machine_idle_time = df_selection['machine_idle_time'].mean()
machine_idle_time = round(machine_idle_time / (1000 * 60 * 60), 3)
actual_production_time = df_selection['actual_production_time'].mean()
actual_production_time = round(actual_production_time / (1000 * 60 * 60), 3)
availability_loss_time = df_selection['availability_loss_time'].mean()
availability_loss_time = round(availability_loss_time / (1000 * 60 * 60), 3)
capacity_utilized_percent = df_selection['capacity_utilized_percent'].mean()
capacity_utilized_percent = round(capacity_utilized_percent, 3)
plant_operating_time_percent = df_selection['plant_operating_time_percent'].mean()
plant_operating_time_percent = round(plant_operating_time_percent, 3)

# Display filtered data
# st.title(":bar_chart: PredictiGuard: Deep Insights Dashboard")
st.title(":bar_chart: Shopfloor Hidden Insights")
st.markdown('### Metrics')
#st.write(df_selection)
col1, col2, col3 = st.columns(3)
col1.metric("Machine Runtime", f"{runtime} Hours", "12%")
col2.metric("Downtime", f"{downtime} Hours","-8%")
col3.metric("Machine Availability", f"{machine_availability}%","23%")

col4, col5, col6 = st.columns(3)
col4.metric("Planned Production Time ", f"{planned_production_time} Hours", "12%")
col5.metric("Total Parts Produced", f"{total_parts_produced} Parts","-8%")
col6.metric("OEE ", f"{oee}%","18%")

col7, col8, col9 = st.columns(3)
col7.metric("Machine Performance", f"{machine_performance_percent}%","23%")
col8.metric("Machine Idle Time ", f"{machine_idle_time} Hours", "-9.8%")
col9.metric("Actual Production Time ", f"{actual_production_time} Hours","12%")

col10, col11, col12 = st.columns(3)
col10.metric("Availability Loss Time", f"{availability_loss_time} Hours","-8%")
col11.metric("Capacity Utilized Percent", f"{capacity_utilized_percent}%","23%")
col12.metric("Plant Operating Time Percent", f"{plant_operating_time_percent}%","23%")

#bar graph
# grouped_data = df.groupby('machineid').mean()
# st.bar_chart(grouped_data)


data = pd.read_csv('original_dataset.csv')


# Preprocessing and autoencoder model
columns_to_drop = ['date', 'timestamp', 'tenant_id', 'org_id', 'edge_id','department_id','unit_id', 'shift_id','machine_id']
data = data.drop(columns=columns_to_drop)

imputer = SimpleImputer(strategy='mean')
data.fillna(data.mean(), inplace=True)

# label_encoder = LabelEncoder()
# categorical_columns = ['machine_id']
# for col in categorical_columns:
#     data[col] = label_encoder.fit_transform(data[col])

scaler = StandardScaler()
numerical_columns = ['total_machine_runtime', 'machine_up_time', 'planned_production_time', 'machine_idle_time',
                     'actual_production_time', 'total_machine_downtime', 'total_planned_downtime', 'unplanned_downtime',
                     'total_parts_produced', 'actual_cycletime', 'time_between_job_parts', 'parts_per_minute',
                     'availability_loss_time', 'cycletime_loss', 'capacity_utilized_percent', 'machine_performance_percent',
                     'machine_availability_percent', 'availability_loss_percent', 'asset_utilization_percent',
                     'planned_downtime_percent']
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

X = data[numerical_columns]
input_dim = X.shape[1]
encoding_dim = 10

inputs = tf.keras.Input(shape=(input_dim,))
encoded = layers.Dense(encoding_dim, activation='relu')(inputs)
decoded = layers.Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = models.Model(inputs, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit autoencoder model
autoencoder.fit(X_train_scaled, X_train_scaled, epochs=25, batch_size=128, shuffle=True, validation_split=0.2)

# Reconstruction errors
reconstructions = autoencoder.predict(X_test_scaled)
mse = np.mean(np.power(X_test_scaled - reconstructions, 2), axis=1)
threshold = np.mean(mse) + 2 * np.std(mse)
anomalies = mse > threshold
df_test = pd.DataFrame(X_test_scaled, columns=numerical_columns)
df_test['anomaly_label'] = anomalies.astype(int)



# Plot reconstruction errors
fig1, ax1 = plt.subplots()
ax1.plot(df_test.index, mse, color='blue', linestyle='-')
ax1.axhline(y=threshold, color='red', linestyle='--', label='Threshold')
ax1.set_title('Reconstruction Errors')
ax1.set_xlabel('Data Point Index')
ax1.set_ylabel('Reconstruction Error')
ax1.legend()
st.pyplot(fig1)