import os
import sys
import cv2
import numpy as np


class MeanFilter:
    def __init__(self, img:np.array, kernelSize:int):
        """
            This function initializes the setting for Mean Filtering;
        :params
            - img, np.array, the noisy image input;
            - kernelSize, int, decides the size of the convolution kernel;
        """
        self.image = img.astype('float')
        self.dimensions = self.image.shape     # Image dimensions: (height, width, channel);
        self.n = kernelSize
        self.kernel = None

    def _createKernel(self):
        """
            This function called initially to build the convolution kernel based on the given size;
        """
        self.kernel = np.ones((self.n, self.n), dtype=float) / pow(self.n, 2)

    def _performConvolution(self, r:int, c:int, a:int, b:int) -> float:
        """
            This function performs the convolution process based on parameters given;
        :param
            - r, int, the x-axis value of the current image pixel coordination;
            - c, int, the y-axis value of the current image pixel coordination;
            - a, int, the x-axis value of the convolution kernel coordination;
            - b, int, the y-axis value of the convolution kernel coordination;
        :return
            - convSum, float, the sum of the convolution performance;
        """
        assert a > 0 and b > 0
        # * Current image pixel coordination: (r, c)
        # * Current Conv pixel relative coordination (w.r.t. image pixel coord): (i, j)
        # * Range of the Conv kernel: (a, b)
        convSum = 0
        # Loop over the whole kernel pixels w.r.t. the corresponding image pixel coordination:
        i = int(-a)
        print(f" * Checking image pixel ({r}, {c})")
        while i <= a:
            j = int(-b)
            while j <= b:
                # Boundary Handling: Check the validation of Conv pixel coordination:
                correspondCoord_x = r + i
                correspondCoord_y = c + j
                if (correspondCoord_x < 0) or (correspondCoord_x >= self.dimensions[0]) or (correspondCoord_y < 0) or (correspondCoord_y >= self.dimensions[1]):
                    j += 1
                    continue
                # If valid, computing the kernel-pixel-based conv value and adding it to the image-pixel-based convSum;
                print(f"\t\t * Checking kernel pixel ({i}, {j})")
                kernelValue = self.kernel[i, j]
                imageValue = self.image[correspondCoord_x, correspondCoord_y]
                convSum += kernelValue * imageValue
                j += 1
            i += 1
        return convSum

    def getFilteredImage(self) -> np.array:
        """
            This controller function that process the mean filtering;
        :return
            - imageFiltered, np.array, the image after process the mean filtering;
        """
        # Create a conv kernel based on the given size:
        self._createKernel()

        # Initializing the convolution settings:
        imageFiltered = np.copy(self.image)
        a = np.ceil((self.n - 1) / 2)
        b = np.ceil((self.n - 1) / 2)

        # Performing convolution on each pixel of the given image:
        for row in range(self.dimensions[0]):
            for col in range(self.dimensions[1]):
                imageFiltered[row][col] = self._performConvolution(r=row, c=col, a=a, b=b)
        return imageFiltered


def main(argv):
    """
        The overall controller function for the whole process;
    :param:
        - argv, list, inputs from the command line, be used to locate data input and define hyperparameter values;
    """
    """ Basic Setting """
    # Env set-up:
    cv2.setUseOptimized(True)

    # Default hyperparameter:
    kernelSize = 7

    """ Command Processing """
    try:
        # Noisy image input:
        noisyImageNameInput = str(argv[1])
        noisyImageInput = cv2.imread(noisyImageNameInput, cv2.IMREAD_GRAYSCALE)  # 0 --> gray-scale  AND  1 --> Color-scale

        # Define kernel size (if needed):
        if len(argv) > 2:
            kernelSize = int(argv[2])

    except:
        sys.stderr.write("\n>> CommandError: Invalid command entered, please check the command and retry;")
        exit()


    """ Apply Mean Filter """
    filteredImageOutput = MeanFilter(img=noisyImageInput, kernelSize=kernelSize).getFilteredImage()


    """ Result Output """
    cv2.imwrite(f'{os.path.splitext(noisyImageNameInput)[0]}_Mean_n={str(kernelSize)}.png', filteredImageOutput)


if __name__ == '__main__':
    main(sys.argv)
