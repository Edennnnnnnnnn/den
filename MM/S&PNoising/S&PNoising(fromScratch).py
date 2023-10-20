import os
import sys
import cv2
import numpy as np


class SaltAndPepperNoise:
    def __init__(self, img, noiseProb, spRatio):
        """
            This function initializes the setting for Salt & Pepper noising;
        :param
            - img, np.array, the origin image input;
            - noiseProb, float, the probability to add noisy;
            - spRatio, float, the ratio decides the percents of salt noise and peper noise;
        """
        self.image = img.astype('float')
        self.dimensions = self.image.shape     # Image dimensions: (height, width, channel);
        self.prob = noiseProb
        self.ratio = spRatio   # (x > spRatio) --> peperNoise  AND  (x < spRatio) --> saltNoise;

    def getNoisyImage(self) -> np.array:
        """
           This function creates Salt & Pepper noise, by merging noise to pixels and finally return the noisy image;
        :return
           - self.image, np.array, the modified image after noise merging;
       """
        for row in range(self.dimensions[0]):
            for col in range(self.dimensions[1]):
                    if np.random.rand() < self.prob:
                        if np.random.rand() < self.ratio:
                            self.image[row][col] = 255
                        else:
                            self.image[row][col] = 0
        return self.image
    