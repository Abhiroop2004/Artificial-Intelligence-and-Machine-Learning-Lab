import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import time, warnings

warnings.filterwarnings('ignore') 
# Initialize synaptic weights
synaptic_weights = 2 * np.random.random((4, 1)) - 1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1-x)

def train(input: list[int], output: list[int], iterations: int):
    global synaptic_weights
    for i in range(iterations):
        prediction = learn(input)
        error = output - prediction
        factor = np.dot(input.T, error * sigmoid_derivative(prediction))
        synaptic_weights += factor

def learn(inputs): 
    return sigmoid(np.dot(inputs, synaptic_weights))

df = pd.read_csv(r"C:\Users\Hp\Downloads\LabEx.csv", index_col=0)
df = df.dropna()
df['Location'].replace(['Poor', 'Medium', 'Good'], [0, 1, 2], inplace=True)
df['Rent (per month) Rs.'] = df['Rent (per month) Rs.'].str.replace(',', '').astype(float)
df['Carpet Area (sq. ft.)'] = df['Carpet Area (sq. ft.)'].str.replace(',', '').astype(float)

df_input = df.drop(['Rent (per month) Rs.'], axis=1)
df_output = df['Rent (per month) Rs.']

scaler = StandardScaler()
df_input = scaler.fit_transform(df_input.values)
df_output = df_output / df_output.max()
iterations = 100

start = time.time()
train(df_input, df_output.to_numpy().reshape(-1, 1), iterations)
end = time.time()

print(f"Training Completed in {end-start} s, for {iterations} iterations")
bed = int(input("Enter number of beds: "))
carpet_area = int(input("Enter carpet area: "))
age = int(input("Enter age of house: "))
location = int(input("Enter type of location (0 for poor, 1 for medium, 2 for good): "))
prediction_input = np.array([[bed, carpet_area, age, location]])
prediction_input_scaled = scaler.transform(prediction_input)
prediction = learn(prediction_input_scaled)

predicted_rent = prediction[0][0] * df['Rent (per month) Rs.'].max()
print("Predicted Rent: Rs.", predicted_rent)