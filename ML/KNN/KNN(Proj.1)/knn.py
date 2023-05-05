import torch
from torchvision import datasets, transforms
import numpy as np


def knn(x_train, y_train, x_test, n_classes, device):
  """
  input:
    x_train: 60000 x 784 matrix: each row is a flattened image of an MNIST digit
    y_train: 60000 vector: label for x_train
    x_test: 1000 x 784 testing images
    n_classes: no. of classes in the classification task
    device: pytorch device on which to run the code
  return: 
    predicted y_test which is a 1000-sized vector
  """
  
  # Apply GPU "Cuba" Acceleration to torch; Convert numpy arrays to torch tensors;
  x_test_tensor = torch.tensor(x_test, dtype=torch.float, device=device)
  x_train_tensor = torch.tensor(x_train, dtype=torch.float, device=device)
  y_train_tensor = torch.tensor(y_train, dtype=torch.float, device=device)

  # Set the num k for nearest neighbors in checking;
  k = 3
  # Acquire the size of the trainning dataset;
  x_trainSize = x_train_tensor.size(dim=0)

  # Main Loop: for predicting each image in the test dataset (each row in x_test);
  y_test = []
  for i in range(len(x_test_tensor)):
    # Apply Torch Broadcasting to each image to fit the shape of the training set provided(for more convenient computing);
    x_test_i_tensor_bc = torch.broadcast_to(x_test_tensor[i], (x_trainSize, 784))
    # Cmpute the Euclidean distance b/w each "neighbor" and the target test image (row);
    rawDist = (x_test_i_tensor_bc - x_train_tensor) ** 2
    distances = rawDist.sum(axis=1) ** 0.5

    # Based on the k setting, get k nearest neighbors (from the trainset) around the target test image;
    knnIdx = distances.topk(k, largest=False)[1]
    
    # Add the prediction of knn (predictions in y_train) to the list predictor;
    predictor = []
    for idx in knnIdx:
      predictor.append(y_train_tensor[idx])
    # Find the one that appeared most frequently, convert it back to numpy array and add it to y_test;
    y_test.append(max(predictor, key=predictor.count).cpu().numpy())
  # Return the algorithms result in numpy array, y_test, the predictions of x_test;
  return y_test



  """
  Dimension of dataset: 
  Train: (60000, 784) (60000,) 
  Test: (1000, 784) (1000,)
  Running...
  Training on GPU: Tesla T4
  Correct Predictions: 965/1000 total 	Accuracy: 0.965000 	Time: 4.432128
  OrderedDict([   ('correct_predict', 965),
      ('accuracy', 0.965),
      ('run_time', 4.432128268000042),
      ('base_score', 100.0),
      ('overall_score', 100.0)])
  """
  