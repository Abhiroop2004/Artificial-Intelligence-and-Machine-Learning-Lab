import numpy as np
synaptic_weights = 2 * np.random.random((2, 1)) - 1

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

x_1 = int(input("Enter 1st number:"))
x_2 = int(input("Enter 2nd number:"))
iterations = int(input("Enter number of iterations: "))
input=np.array([[5, 4], [4, 5], [5, 5], [12, 5], [5, 12], [12, 12], [12,11], [11,12], [30,20], [20, 30], [20, 4], [5, 20]])
output=np.array([[1], [0], [0], [1], [0], [0], [1], [0], [1], [0], [1], [0]])
train(input, output, iterations)
prediction = learn(np.array([6,2]))
if (prediction > 0.5):
    print (1)
else:
    print (0)