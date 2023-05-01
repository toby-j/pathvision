import numpy as np

'''
Calculate the percentage of overlap of gradients given a segment and the target area
'''


def calculate_overlap(original_base_image, masked_gradients) -> float:
    total_pixel_sum = np.sum(np.asarray(original_base_image), axis=2)
    total_pixel_weight = np.sum(total_pixel_sum)
    # Next calculate the sum of the crop
    masked_pixel_sum = np.sum(masked_gradients, axis=2)
    masked_pixel_weight = np.sum(masked_pixel_sum)
    # Based on these numbers, calculate percentage of overlap
    return ((total_pixel_weight - masked_pixel_weight) / total_pixel_weight) * 100
