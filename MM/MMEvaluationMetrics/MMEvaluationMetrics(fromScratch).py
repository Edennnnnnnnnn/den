import sys
import cv2
import numpy as np


class MetricEvaluating:
    def __init__(self, img_1:np.array, img_2:np.array, L: int, K1:float, K2:float):
        """
            This function initializes the setting for Mean Filtering;
        :params
            - img_1, np.array, the first image input;
            - img_2, np.array, the second image input;
            - L, int, the maximal picture intensity value;
            - K1, float, the weight for computing the first constant variable;
            - K1, float, the weight for computing the second constant variable;
        """
        # Warning: two images must have the same size;
        assert img_1.shape == img_2.shape
        self.image_1 = img_1.astype('float')
        self.image_2 = img_2.astype('float')
        self.dimensions = img_1.shape     # Image dimensions: (height, width, channel), two images must have same size;

        self.L = L
        self.C1 = pow((K1 * L), 2)
        self.C2 = pow((K2 * L), 2)

    def getSSIMIndex(self) -> float:
        """
            This function processes the pattern to compute the SSIM index of the given images;
        :return
            - SSIMIndex, float, the index value computed;
        """
        # Compute the mean of images:
        mean_1 = np.mean(self.image_1)
        mean_2 = np.mean(self.image_2)

        # Compute the variance of images and the covariance between:
        var_1 = np.var(self.image_1)
        var_2 = np.var(self.image_2)
        covar = np.mean((self.image_1 - mean_1) * (self.image_2 - mean_2))

        # Compute the SSMI index:
        numerator = (2 * mean_1 * mean_2 + self.C1) * (2 * covar + self.C2)
        denominator = (pow(mean_1, 2) + pow(mean_2, 2) + self.C1) * (var_1 + var_2 + self.C2)
        return numerator / denominator

    def getPSNRIndex(self) -> float:
        """
            This function processes the pattern to compute the PSNR index of the given images;
        :return
            - PSNRIndex, float, the index value computed;
        """
        # Compute the MSE and then the PSNR index:
        MSE = np.mean(pow((self.image_1 - self.image_2), 2))
        if MSE == 0:
            return np.inf
        return 20 * np.log10(self.L / np.sqrt(MSE))


def main(argv):
    """
        The overall controller function for the whole process;
    :param:
        - argv, list, inputs from the command line, be used to locate data input and define hyperparameter values;
    """
    """ Basic Settings """
    # Env set-up:
    cv2.setUseOptimized(True)

    # Default hyperparameters:
    l = 255
    k1 = 0.01
    k2 = 0.03

    # Evaluation selections:
    applyEvaluation = {'SSIM': False,
                       'PSNR': False}


    """ Command Processing """
    try:
        # Evaluation selection:
        evaluationTypeInput = str(argv[1])
        if evaluationTypeInput == "ALL":
            applyEvaluation['SSIM'] = True
            applyEvaluation['PSNR'] = True
        else:
            applyEvaluation[evaluationTypeInput] = True

        # Images input:
        ImageNameInput_1 = str(argv[2])
        imageInput_1 = cv2.imread(ImageNameInput_1, cv2.IMREAD_GRAYSCALE)  # 0 --> gray-scale  AND  1 --> Color-scale
        ImageNameInput_2 = str(argv[3])
        imageInput_2 = cv2.imread(ImageNameInput_2, cv2.IMREAD_GRAYSCALE)  # 0 --> gray-scale  AND  1 --> Color-scale
    except:
        sys.stderr.write("\n>> CommandError: Invalid command entered, please check the command and retry;")
        exit()


    """ Apply Evaluation Metrics """
    # Initialize evaluators:
    evaluator = MetricEvaluating(imageInput_1, imageInput_2, l, k1, k2)
    print(f">> Checking: \n\t* Image1=[{ImageNameInput_1}] \n\t* Image2=[{ImageNameInput_2}]")

    # Process the evaluation by computing the SSIM index:
    if applyEvaluation['SSIM']:
        SSIMIndex = evaluator.getSSIMIndex()
        print(f"\t\t > SSIM Index = {SSIMIndex}")

    # Process the evaluation by computing the PSNR index:
    if applyEvaluation['PSNR']:
        PSNRIndex = evaluator.getPSNRIndex()
        print(f"\t\t > PSNR Index = {PSNRIndex}")



if __name__ == '__main__':
    main(sys.argv)