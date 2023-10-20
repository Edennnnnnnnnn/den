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


def main(argv):
    """
        The overall controller function for the whole process;
    :param:
        - argv, list, inputs from the command line, be used to locate data input and define hyperparameter values;
    """
    """ Basic Settings """
    # Env set-up:
    cv2.setUseOptimized(True)

    # Noise selections:
    applyNoise = {'Gaussian': False,
                  'SP': False,
                  'Speckle': False}


    """ Command Processing """
    try:
        # Noise selection:
        noiseTypeInput = str(argv[1])
        applyNoise[noiseTypeInput] = True

        # Image input:
        imageNameInput = str(argv[2])
        imageInput = cv2.imread(imageNameInput, cv2.IMREAD_GRAYSCALE)  # 0 --> gray-scale  AND  1 --> Color-scale
    except:
        sys.stderr.write("\n>> CommandError: Invalid command entered, please check the command and retry;")
        exit()


    """ Applying Noise Builders """
    noisyImageOutput = None

    # Get image with Gaussian Noise:
    if applyNoise['Gaussian']:
        noisyImageOutput = GaussianNoise(img=imageInput, mean=0, var=50).getNoisyImage()

    # Get image with Salt & Peper Noise:
    if applyNoise['SP']:
        noisyImageOutput = SaltAndPepperNoise(img=imageInput, noiseProb=0.1, spRatio=0.5).getNoisyImage()

    # Get image with Speckle Noise:
    if applyNoise['Speckle']:
        noisyImageOutput = SpeckleNoise(img=imageInput, var=3).getNoisyImage()


    """ Result Output """
    cv2.imwrite(f'{os.path.splitext(imageNameInput)[0]}_{noiseTypeInput}.png', noisyImageOutput)



if __name__ == '__main__':
    main(sys.argv)
