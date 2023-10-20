import os
import sys
import cv2
import numpy as np


class GaussianNoise:
	def __init__(self, img:np.array, mean:float, var:float):
		"""
			This function initializes the setting for Gaussian noising;
		:param
			- img, np.array, the origin image input;
			- mean, float, the mean value used to create the gaussian distribution;
			- var, float, the variance value used to create the gaussian distribution;
		"""
		self.image = img.astype('float')
		self.dimensions = self.image.shape     # Image dimensions: (height, width, channel);
		self.mean = mean
		self.variance = var
		
	def getNoisyImage(self) -> np.array:
		"""
			This function creates Gaussian noise based on the BoxMullerTransform method, by zooming, clipping and merging
		the noise with the origin image and finally return the noisy image;
		:return
			- self.image, np.array, the modified image after gaussian noise merging;
		"""
		# Taking the Box-Muller Transform:
		u1 = np.random.uniform(size=self.dimensions)
		u2 = np.random.uniform(size=self.dimensions)
		Z = self._BoxMullerTransform(u1, u2)
		
		# Noise zooming and panning:
		Z *= self.variance
		Z += self.mean
		
		# Noise clipping and merging:
		self.image += self.noiseClip(Z)
		return self.image
	
	@staticmethod
	def _BoxMullerTransform(u1:np.array, u2:np.array) -> np.array:
		"""
			This is a helper function processing the BoxMullerTransform and return the result;
		:param
			- u1, np.array, the first uniform distribution with the similar size of the image;
			- u2, np.array, the second uniform distribution with the similar size of the image;
		:return
			- Z, np.array, the BoxMullerTransform result;
		"""
		return (-2 * np.log(u1)) ** 0.5 * np.cos(2 * np.pi * u2)
	
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