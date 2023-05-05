"""
By Eden Zhou
Apr. 15, 2023
"""

from utils import plot_data, generate_data
import numpy as np


def train_linear_regression(X, t):
	# dJ/d = 2·S·w - 2·X^T·t
	# W --> (X^T·X)^(-1)·X^T·t  <==>  W --> [w b]
	X_ = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
	X_T = X_.T
	W = np.linalg.inv(X_T @ X_) @ X_T @ t
	
	w = W[:-1]
	b = W[-1]
	return w, b


def predict_linear_regression(X, w, b):
	# t_hat = Xw^T + b
	t = X @ w.T + b
	t = np.where(t >= 0.5, 1, 0)
	return t

def get_accuracy(t, t_hat):
	acc = (t == t_hat).sum() / t.shape[0]
	return acc


def main():
	# Linear regression classifier
	X, t = generate_data("A")
	w, b = train_linear_regression(X, t)
	t_hat = predict_linear_regression(X, w, b)
	print("Accuracy of linear regression on dataset A:", get_accuracy(t_hat, t))
	plot_data(X, t, w, b, is_logistic=False, figure_name='dataset_A_linear.png')
	
	# Linear regression classifier
	X, t = generate_data("B")
	w, b = train_linear_regression(X, t)
	t_hat = predict_linear_regression(X, w, b)
	print("Accuracy of linear regression on dataset B:", get_accuracy(t_hat, t))
	plot_data(X, t, w, b, is_logistic=False, figure_name='dataset_B_linear.png')
	
	
main()
