import numpy as np

'''
Calculate the percentage of overlap of gradients given a segment and the target area
'''


def calculate_overlap(original_base_image, masked_gradients) -> float:
    # first we get the total of the gradients over the whole mask
    original_base_image = np.asarray(original_base_image)


    total_pixel_sum = np.sum(original_base_image)
    total_pixel_weight = np.sum(total_pixel_sum)
    # Next calculate the sum of the crop
    masked_pixel_sum = np.sum(masked_gradients)
    masked_pixel_weight = np.sum(masked_pixel_sum)
    # Based on these numbers, calculate percentage of overlap
    print("Total image sum: {}".format(total_pixel_weight))
    print("Crop sum: {}".format(masked_pixel_weight))

    overlap = (masked_pixel_weight / total_pixel_weight) * 100
    print(overlap)
    return overlap
