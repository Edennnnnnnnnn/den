"""
By Eden Zhou
Apr. 15, 2023
"""

from utils import plot_data, generate_data
import numpy as np


def _sigmoidFunction(Z):
	return 1.0 / (1.0 + np.exp(-Z))

def train_logistic_regression(X, t):
	"""
	Given data, train your logistic classifier.
	Return weight and bias
	"""
	# Gradient Descent
	alpha = 0.1
	w = np.zeros(X.shape[1])
	b = 0.0
	epoch = 0
	while epoch < 1000:
		# t = Sigmoid(X·w^T + b)
		t_hat = _sigmoidFunction(X @ w.T + b)
		w -= alpha * (X.T @ (t_hat - t))
		b -= alpha * (np.sum(t_hat - t))
		epoch += 1
	return w, b


def predict_logistic_regression(X, w, b):
	"""
	Generate predictions by your logistic classifier.
	"""
	# t = Sigmoid(X·w^T + b)
	t = _sigmoidFunction(X @ w.T + b)
	t = np.where(t >= 0.5, 1, 0)
	
	return t

def get_accuracy(t, t_hat):
	"""
	Calculate accuracy,
	"""
	acc = (t == t_hat).sum() / t.shape[0]
	return acc


def main():
	
	# logistic regression classifier
	X, t = generate_data("A")
	w, b = train_logistic_regression(X, t)
	t_hat = predict_logistic_regression(X, w, b)
	print("Accuracy of linear regression on dataset A:", get_accuracy(t_hat, t))
	plot_data(X, t, w, b, is_logistic=True, figure_name='dataset_A_logistic.png')
	
	
	# logistic regression classifier
	X, t = generate_data("B")
	w, b = train_logistic_regression(X, t)
	t_hat = predict_logistic_regression(X, w, b)
	print("Accuracy of linear regression on dataset B:", get_accuracy(t_hat, t))
	plot_data(X, t, w, b, is_logistic=True, figure_name='dataset_B_logistic.png')
	
	
main()
