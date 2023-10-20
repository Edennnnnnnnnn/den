import os
import sys
import cv2
import numpy as np


class SpeckleNoise:
	def __init__(self, img:np.array, var:float, mean=0):
		"""
			This function initializes the setting for Speckle noising;
		:param
			- img, np.array, the origin image input;
			- var, float, the variance value used to create the gaussian distribution;
			- mean, float, the mean value used to create the gaussian distribution, default=0;
		"""
		self.image = img.astype('float')
		self.dimensions = self.image.shape     # Image dimensions: (height, width, channel);
		self.mean = mean
		self.variance = var
		
	def getNoisyImage(self) -> np.array:
		"""
			This function creates Speckle noise, by clipping and merging the noise with the origin image and finally
		return the noisy image;
		:return
			- self.image, np.array, the modified image after merging and clipping;
		"""
		gauss = np.random.normal(self.mean, self.variance, self.dimensions)
		for row in range(self.image.shape[0]):
			for col in range(self.image.shape[1]):
				self.image[row][col] *= (1 + gauss[row][col])
		return self.noiseClip(self.image)
	
	@staticmethod
	def noiseClip(I:np.array):
		"""
			This is a helper function that clips and fixes invalid pixel values (>255 and <0) in an (image) array to
		avoid incorrect results;
		:param
			- I, np.array, the clipping-needed (image) array;
		:return
			- I, np.array, the after-clipping (image) array;
		"""
		I[I > 255] = 255
		I[I < 0] = 0
		return I