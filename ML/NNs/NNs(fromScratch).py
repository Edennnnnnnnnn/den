import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
def sigmoid_derivative(sigma):
    return sigma * (1 - sigma)
    
X = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1]])
y = np.array([[0,1,0.3,0.5]]).T

# randomly initialize our weights with mean 0
np.random.seed(1)
w0 = 2 * np.random.random((3,4)) - 1
w1 = 2 * np.random.random((4,1)) - 1

nextPrintTime = 1
for i in range(1000000):
    # Forward Propagation:
    l0 = X
    l1 = sigmoid(l0 @ w0)
    l2 = sigmoid(l1 @ w1)
    l2_error = y - l2
    
    if i >= nextPrintTime:
        nextPrintTime *= 2
        print("Iteration {} Error: {}".format(i, np.mean(np.abs(l2_error))))
        
    # Backward Propagation:
    l2_gd = l2_error * sigmoid_derivative(l2)
    l1_error = l2_gd @ w1.T
    l1_gd = l1_error * sigmoid_derivative(l1)

    w1 += l1.T @ l2_gd
    w0 += l0.T @ l1_gd