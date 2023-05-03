import numpy as np

'''
Calculate the percentage of overlap of gradients given a segment and the mask area

We have to run the segmentor again, because crop on origin has the black background. To
increase performance we should use ['crop'] instead of ['crops_on_origin'] but we'd have to
scale it to the size of the origin image and in correct coordinates. We're only interested in
stuff in the bounding box, so we repeat the process, but just for pixels in the bounding
box. 

:param segmenter: Image segmentation model
:param raw_gradients: Gradient 
:param crop: 
:return: Percentage of overlap.
'''

def calculate_overlap(segmenter, raw_gradients, crop) -> float:
    # Get mask using the bounding box crop
    outputs = segmenter(np.array(crop))
    instances = outputs["instances"].to("cpu")
    raw_mask = instances.pred_masks[0].numpy().squeeze()
    # Invert the mask, so the pixels outside are True.
    raw_mask[:] = ~raw_mask
    # Replace where the mask is False, with a 0 in the same location in raw_gradients
    segment = np.where(np.array(raw_mask), np.array(raw_gradients), 0)
    # We now have just the gradients in a 2D vector of the pixels outside the bounding box
    sum_mask_segment = np.sum(segment)
    # Find what percentage the outside pixels make up of the full gradient image by summing both 2D vectors
    total_sum = np.sum(raw_gradients)
    # What percentage are the gradients outside the segment of the full gradient vector
    percentage_overlap = (sum_mask_segment / total_sum) * 100
    return percentage_overlap
