import os
import sys
import cv2
import numpy as np
# import bm3d

def init(img, blk_size, beta_kaiser):
    """
        Initialize arrays to store the filtered image and weight, and construct a 2D Kaiser window.
    Parameters:
        - img: The input image represented as a 2D array.
        - blk_size: The size of the block to be used in filtering.
        - beta_kaiser: The beta parameter for the Kaiser window, which affects the shape of the window.
    Returns:
        - img_matrix: A 2D array of zeros with the same shape as the input image, intended to store the filtered image.
        - weight_matrix: A 2D array of zeros with the same shape as the input image, intended to store the weight coefficients.
        - kaiser_window_2d: A 2D array representing the Kaiser window, which will be used in the filtering process.
    """
    img_shape = img.shape
    img_matrix = np.zeros(img_shape, dtype=float)
    weight_matrix = np.zeros(img_shape, dtype=float)

    # Generate a 1D Kaiser window with the given beta parameter
    kaiser_window_1d = np.kaiser(blk_size, beta_kaiser)
    # Construct a 2D Kaiser window by taking the outer product of the 1D window with itself
    kaiser_window_2d = np.outer(kaiser_window_1d, kaiser_window_1d)

    return img_matrix, weight_matrix, kaiser_window_2d


def locate_blk(i, j, blk_step, blk_size, img_width, img_height):
    """
        Ensure the current block does not exceed the image boundaries.
    Parameters:
        - i: The row index in the image where block matching begins.
        - j: The column index in the image where block matching begins.
        - blk_step: The step size for moving the block in both horizontal and vertical directions.
        - blk_size: The size of the block used for matching.
        - img_width: The width of the input image.
        - img_height: The height of the input image.
    Returns:
        - block_point: A NumPy array containing the x and y coordinates of the top-left corner of the block, ensuring that the block does not exceed image boundaries.
    """
    if i * blk_step + blk_size < img_width:
        point_x = i * blk_step
    else:
        point_x = img_width - blk_size

    if j * blk_step + blk_size < img_height:
        point_y = j * blk_step
    else:
        point_y = img_height - blk_size

    block_point = np.array((point_x, point_y), dtype=int)
    return block_point


def define_search_window(noisy_img, block_point, window_size, blk_size):
    """
        Uses the top-left coordinate of the block to return a coordinate tuple (x, y) for defining the top-left corner of the Search Window.
    Parameters:
        - noisy_img: The noisy image.
        - block_point: The top-left coordinate (x, y) of the current block being processed.
        - window_size: The size of the search window.
        - blk_size: The size of the block.
    """
    point_x, point_y = block_point  # Current block's top-left coordinate

    # Calculate the top-left and bottom-right coordinates of the search window
    lx = point_x + blk_size / 2 - window_size / 2  
    ly = point_y + blk_size / 2 - window_size / 2  
    rx = lx + window_size  
    ry = ly + window_size  

    # Check if the search window goes beyond the image boundaries
    if lx < 0:
        lx = 0
    elif rx >= noisy_img.shape[0]:
        lx = noisy_img.shape[0] - window_size
        rx = noisy_img.shape[0] - 1
    if ly < 0:
        ly = 0
    elif ry >= noisy_img.shape[1]:
        ly = noisy_img.shape[1] - window_size
        ry = noisy_img.shape[1] - 1

    return np.array((lx, ly), dtype=int)


def fast_block_matching(noisy_img, block_point, config):
    """
        Fast block matching
    Parameters:
        - noisy_img: Image with noise
        - block_point: Top-left corner coordinates of the current block
        - config: Configuration dictionary containing various parameters needed for matching
    Returns:
        - final_similar_blocks: Final set of similar blocks
        - block_positions: Positions of similar blocks
        - count: Number of similar blocks
    """
    # Unpack configuration parameters:
    blk_size = config['blk_size']
    step_length = config['step_length']
    threshold = config['first_match_threshold']
    max_match = config['max_matched_cnt']
    window_size = config['search_window']
    
    # Initialize arrays for storing the positions and DCT values of similar blocks:
    block_positions = np.zeros((max_match, 2), dtype=int)
    final_similar_blocks = np.zeros((max_match, blk_size, blk_size), dtype=float)

    # Calculate how many possible similar blocks are there in the search window:
    block_count = (window_size - blk_size) // step_length + 1
    temp_positions = np.zeros((block_count ** 2, 2), dtype=int)
    similar_blocks = np.zeros((block_count ** 2, blk_size, blk_size), dtype=float)
    distances = np.zeros(block_count ** 2, dtype=float)

    # Perform DCT on the current block
    target_blk = noisy_img[block_point[0]: block_point[0] + blk_size, block_point[1]: block_point[1] + blk_size]
    dct_target_blk = cv2.dct(target_blk.astype(np.float64))
    # Save the DCT transformed target block and its position:
    block_positions[0, :] = block_point
    final_similar_blocks[0, :, :] = dct_target_blk
    
    # Define the position of the search window:
    search_window_origin = define_search_window(noisy_img, block_point, window_size, blk_size)
    max_x, max_y = search_window_origin[0] + block_count, search_window_origin[1] + block_count
    matched_count = 0 # Number of similar blocks found;
    # Loop through each possible similar block (located by block_tl_points) within the search window:
    cur_x = search_window_origin[0]
    while cur_x < max_x:
        cur_y = search_window_origin[1]
        while cur_y < max_y:
            # Pick the block from the image and compute its dct:
            cur_blk = noisy_img[cur_x: cur_x + blk_size, cur_y: cur_y + blk_size]
            dct_cur_blk = cv2.dct(cur_blk.astype(np.float64))
            # Compute the relevant block distance:
            cur_dist = np.linalg.norm(pow((dct_target_blk - dct_cur_blk), 2) / pow(blk_size, 2))

            # Check validation, if invalid then skipping "too similar" blocks:
            if (cur_dist <= 0) or (cur_y >= threshold):
                cur_y += 1
                continue

            # Store block data in arrays defined:
            temp_positions[matched_count, :] = (cur_x, cur_y)
            similar_blocks[matched_count, :, :] = dct_cur_blk
            # Record distance data in the array defined:
            distances[matched_count] = cur_dist

            # Continue the loop by updating hyperparams:
            matched_count += 1
            cur_y += step_length
        cur_x += step_length

    # Slice to get the distances of actually found similar blocks:
    distances = distances[:matched_count]
    # Sort distances to find the most similar blocks:
    sorted_indices = np.argsort(distances)
    
    # Determine the final number of similar blocks:
    count = min(matched_count, max_match)
    # Save the found similar blocks and their positions:
    for i in range(1, count):
        final_similar_blocks[i, :, :] = similar_blocks[sorted_indices[i-1], :, :]
        block_positions[i, :] = temp_positions[sorted_indices[i-1], :]
    return final_similar_blocks, block_positions, count


def _3d_filtering(similar_blocks, hard_threshold):
    """
        Perform 3D transform and filtering
    Parameters:
        - similar_blocks: Set of similar blocks, already in frequency domain
        - hard_threshold: Hard threshold for frequency domain filtering
    Returns:
        - filtered_blocks: Filtered set of similar blocks
        - nonzero_count: Number of non-zero elements
    """
    nonzero_count = 0  # Number of non-zero elements;
    block_shape = similar_blocks.shape
    # Loop through the block for filtering:
    for i in range(block_shape[1]):
        for j in range(block_shape[2]):
            # Pick each column (w.r.t. z-axis) out from the similar_blocks:
            cur_col = cv2.dct(similar_blocks[:, i, j])

            # Apply the hard threshold filter (transfer all elements less than the threshold to 0 and count the num):
            cur_col[np.abs(cur_col[:]) < hard_threshold] = 0
            nonzero_count += cur_col.nonzero()[0].size

            # Perform the inverse-DCT process on the filtered block and then put it back:
            similar_blocks[:, i, j] = cv2.idct(cur_col)[0]
    return similar_blocks, nonzero_count


def aggregate_hard_threshold(similar_blocks, block_positions, basic_img, weight_img, nonzero_count, block_count, kaiser_window, sigma):
    """
        Perform weighted accumulation on the stack output after 3D transform and filtering to get the preliminary filtered image.
    The weighted accumulation is done by multiplying with a Kaiser window and the weight of non-zero items.
    Parameters:
        - similar_blocks: A set of similar blocks, represented in the frequency domain
        - block_positions: Positions of blocks
        - basic_img: Basic image
        - weight_img: Weight image
        - nonzero_count: Number of non-zero items
        - block_count: Number of blocks
        - kaiser_window: Kaiser window
        - sigma: Standard deviation
    """
    # Ensure the number of non-zero items is at least 1
    if nonzero_count < 1:
        nonzero_count = 1
    
    # Calculate block weight
    block_shape = similar_blocks.shape
    block_weight = (1. / (sigma ** 2 * nonzero_count)) * kaiser_window
    for i in range(block_count):
        position = block_positions[i, :]
        temp_img = block_weight * cv2.idct(similar_blocks[i, :, :])
        
        # Update the basic image and weight image
        basic_img[position[0]:position[0] + block_shape[1], position[1]:position[1] + block_shape[2]] += temp_img
        weight_img[position[0]:position[0] + block_shape[1], position[1]:position[1] + block_shape[2]] += block_weight


def bm3d_first_step(noisy_img, config):
    """
        Basic Denoising
    """
    # Initialize parameters
    width, height = noisy_img.shape
    block_size = config['blk_size']
    block_step = config['step_length']

    # Calculate the number of blocks to search based on the block step:
    width_num = (width - block_size) // block_step + 1
    height_num = (height - block_size) // block_step + 1
    print(f"* Total #block ({width_num}, {height_num})")

    # Initialize arrays
    basic_img, weight_matrix, kaiser_window = init(noisy_img, block_size, config["beta_kaiser"])

    # Loop through blocks, +2 is to avoid insufficient space at the edge:
    for i in range(width_num):
        for j in range(height_num):
            print(f"\t>> Processing block ({i}, {j})")
            # Get the top-left corner coordinate of the current reference block
            block_point = locate_blk(i, j, block_step, block_size, width, height)
            print("\t\t> Block Located")

            # Find similar blocks, their positions, and the count of similar blocks
            similar_blocks, positions, count = fast_block_matching(noisy_img, block_point, config)
            print("\t\t> Block Matched")

            # Apply 3D filtering to similar blocks and get the count of non-zero statistics
            filtered_blocks, non_zero_count = _3d_filtering(similar_blocks, config['3d_hard_threshold'])
            print("\t\t> 3D Filtering Done")

            # Aggregate using hard threshold
            aggregate_hard_threshold(filtered_blocks, positions, basic_img, weight_matrix, non_zero_count, count, kaiser_window, config['sigma'])
            print("\t\t> Aggregation Done")

    # Normalize the basic image
    basic_img /= weight_matrix
    basic = np.matrix(basic_img, dtype=int).astype(np.uint8)
    return basic


def bm3dFullVer(image):
    """
        The full version of the BM3D Filter, provided by the 3-rd party library;
    """
    # image = bm3d.bm3d(imageNoisy, sigma_psd=30 / 255, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)
    return image


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
    sigma = 25
    threshold_hard_3d = 2.7 * sigma
    blk_size = 8
    step_length = 1
    first_match_threshold = 125 * blk_size ** 2
    max_matched_cnt = 16
    search_window = 16
    beta_kaiser = 1.5

    config = {
        'sigma': sigma,
        'first_match_threshold': first_match_threshold,
        'blk_size': blk_size,
        'step_length': step_length,
        'max_matched_cnt': max_matched_cnt,
        'search_window': search_window,
        '3d_hard_threshold': threshold_hard_3d,
        'beta_kaiser': beta_kaiser
    }


    """ Command Processing """
    try:
        # Noisy image input:
        noisyImageNameInput = str(argv[1])
        noisyImageInput = cv2.imread(noisyImageNameInput, cv2.IMREAD_GRAYSCALE)      # 0 --> gray-scale  AND  1 --> Color-scale
    except:
        sys.stderr.write("\n>> CommandError: Invalid command entered, please check the command and retry;")
        exit()


    """ Take Filtering """
    # Counting time and starting the filter:
    e1 = cv2.getTickCount()
    filteredImageOutput = bm3d_first_step(noisyImageInput, config)
    # filteredImageOutput = bm3dFullVer(noisyImageInput)          # Try the full version of BM3D;
    e2 = cv2.getTickCount()
    time = (e2 - e1) / cv2.getTickFrequency()
    print("The Processing time of the First step is %f s" % time)


    """ Result Output """
    cv2.imwrite(f'{os.path.splitext(noisyImageNameInput)[0]}_BM3D.png', filteredImageOutput)



if __name__ == '__main__':
    main(sys.argv)