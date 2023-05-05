#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


def rotate(data, degree):
    # data: M x 2
    theta = np.pi / 180 * degree
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])  # rotation matrix
    return np.dot(data, R.T)


def leastSquares(X, Y) -> np.array:
    # In this function, X is always the input, Y is always the output;
    # X: (M x (d+1)), Y: (M x 1), where d=1 here, closed form solution by matrix-vector representations only
    # Recall - Close-form solution for linear regression --> (X^T * X)^(-1) * X^T * Y, return weights w;
    X_T = np.transpose(X)

    # Dot product of X_t and X, (X^T * X):
    temp1 = np.dot(X_T, X)

    # Invert matrix of the dot product between X_T and X, (X^T * X)^(-1):
    temp1 = np.linalg.inv(temp1)

    # Dot product between X_T and Y, (X^T * Y):
    temp2 = np.dot(X_T, Y)

    # dot product between two parts, (X^T * X)^(-1) * X^T * Y:
    w = np.dot(temp1, temp2)
    return w


def model(X, w):
    # X: M x (d+1)
    # w: d+1
    # return y_hat: M x 1
    y_hat = np.dot(X, w)
    return y_hat


def generate_data(M, var1, var2, degree):
    # data generate involves two steps:
    # Step I: generating 2-D data, where two axis are independent
    # M (scalar): The number of data samples
    # var1 (scalar): variance of a
    # var2 (scalar): variance of b

    mu = [0, 0]

    Cov = [[var1, 0],
           [0,  var2]]

    data = np.random.multivariate_normal(mu, Cov, M)
    # shape: M x 2

    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], color="blue")
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.xlabel('a')
    plt.ylabel('b')
    plt.tight_layout()
    plt.savefig('data_ab_' + str(var2) + '.jpg')

    # Step II: rotate data by 45 degree counter-clockwise,
    # so that the two dimensions are in fact correlated

    data = rotate(data, degree)
    plt.tight_layout()
    plt.figure()
    # plot the data points
    plt.scatter(data[:, 0], data[:, 1], color="blue")
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.xlabel('x')
    plt.ylabel('y')

    # plot the line where data are mostly generated around
    X_new = np.linspace(-5, 5, 100, endpoint=True).reshape([100, 1])

    Y_new = np.tan(np.pi / 180 * degree) * X_new
    plt.plot(X_new, Y_new, color="blue", linestyle='dashed')
    plt.tight_layout()
    plt.savefig('data_xy_' + str(var2) + '_' + str(degree) + '.jpg')
    return data


###########################
# Main code starts here
###########################
# Settings
M = 5000
var1 = 1
var2 = 0.1
degree = 45

data = generate_data(M, var1, var2, degree)

##########
# Training the linear regression model predicting y from x (x2y)
Input = data[:, 0].reshape((-1, 1))  # M x d, where d=1
Output = data[:, 1].reshape((-1, 1))  # M x 1
# M x (d+1) augmented feature
Input_aug = np.concatenate([Input, np.ones([M, 1])], axis=1)

w_x2y = leastSquares(Input_aug, Output)  # (d+1) x 1, where d=1
print('Predicting y from x (x2y): weight=' +
      str(w_x2y[0, 0]), 'bias=', str(w_x2y[1, 0]))

# # Training the linear regression model predicting x from y (y2x)
Input = data[:, 1].reshape((-1, 1))  # M x d, where d=1

# M x (d+1) augmented feature
Input_aug = np.concatenate([Input, np.ones([M, 1])], axis=1)
Output = data[:, 0].reshape((-1, 1))  # M x 1

w_y2x = leastSquares(Input_aug, Output)  # (d+1) x 1, where d=1
print('Predicting x from y (y2x): weight=' +
      str(w_y2x[0, 0]), 'bias=', str(w_y2x[1, 0]))


# plot the data points
plt.figure()
X = data[:, 0].reshape((-1, 1))  # M x d, where d=1
Y = data[:, 1].reshape((-1, 1))  # M x d, where d=1

plt.scatter(X, Y, color="blue", marker='x')
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.xlabel('x')
plt.ylabel('y')

# plot the line where data are mostly generated around
X_new = np.linspace(-4, 4, 100, endpoint=True).reshape([100, 1])

Y_new = np.tan(np.pi / 180 * degree) * X_new
plt.plot(X_new, Y_new, color="blue", linestyle='dashed')

# plot the prediction of y from x (x2y)
# M x d, where d=1
X_new = np.linspace(-4, 4, 100, endpoint=True).reshape([100, 1])
# M x (d+1) augmented feature
X_new_aug = np.concatenate([X_new, np.ones([X_new.shape[0], 1])], axis=1)
plt.plot(X_new, model(X_new_aug, w_x2y), color="red", label="x2y")

# plot the prediction of x from y (y2x)
# M x d, where d=1
Y_new = np.linspace(-4, 4, 100, endpoint=True).reshape([100, 1])
# M x (d+1) augmented feature
Y_new_aug = np.concatenate([X_new, np.ones([X_new.shape[0], 1])], axis=1)
plt.plot(model(Y_new_aug, w_y2x), Y_new, color="green", label="y2x")
plt.legend()
plt.tight_layout()
plt.savefig('Regression_model_' + str(var2) + '_' + str(degree) + '.jpg')
