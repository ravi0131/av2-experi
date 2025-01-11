from shapely.geometry import Polygon
import numpy as np

def filter_bboxes_on_area(corners: np.ndarray, percentile=0.6, area_upper_threshold=80) ->np.ndarray:
    """
    Filter out bounding boxes that have an area less than a certain threshold
    
    Args:
        corners: A numpy array of shape (n, 4, 2) containing the four corners of the bounding boxes
        percentile: The percentile of the area to use as the threshold
        area_threshold: The threshold area in square units
        
    Returns:
        A numpy array of shape (m, 4, 2) containing the bounding boxes that meet the threshold
    """
    areas = [Polygon(corners[i]).area for i in range(corners.shape[0])]
    area_lower_bound = np.percentile(areas, percentile)
    
    
    filtered_corners = []
    for idx, corner in enumerate(corners):
        poly = Polygon(corner)
        if poly.area > area_lower_bound and poly.area <= area_upper_threshold:
            filtered_corners.append(corner)
            
    filtered_corners = np.stack(filtered_corners)
    
    return filtered_corners