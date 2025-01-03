import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import matplotlib.axes as mpl_axes
from .utils import *
from  .rectangle_fitting_liso import fit_2d_box_modest, minimum_bounding_rectangle
class Bboxer:
    def __init__(self):
        self.clustered_points = None
        self.clustered_labels = None

    def cluster(self, points, ax: mpl_axes= None, visualize=False):
        """
        Perform DBSCAN clustering on the non-ground points.
        Args:
        - points: a nx2 numpy array of non-ground LiDAR points in the frame
        - ax(optional): Matplotlib axes for plotting. If not passed, will automatically create one
        - visualize(optional): A boolean flag to visualize the clustered points 
        """
        xy_points = points
        eps = 0.4
        min_samples = 8
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(xy_points)
        
        self.clustered_points = xy_points
        self.clustered_labels = labels
        if visualize:
            if ax is None:
                clustered_points_viz(xy_points, labels)
            if ax is not None:
                clustered_points_viz(clustered_points=xy_points, clustered_labels=labels, ax=ax)

    def estimate_bboxes_from_clusters(self, clustered_points, clustered_labels):
        """
        Estimate bounding boxes for each cluster using the minimum bounding rectangle approach.
        
        Args:
            clustered_points (np.ndarray): N x 2 array of (x, y) points.
            clustered_labels (np.ndarray): N-length array of cluster labels for the points.
        
        Returns:
            List[Dict]: A list of dictionaries where each dict contains 'bbox', 'angle', and 'area' for each cluster.
        """
        unique_labels = set(clustered_labels)
        rects = []

        for label in unique_labels:
            if label == -1:  # Skip noise points (if DBSCAN labels noise as -1)
                continue

            # Select points that belong to this cluster
            cluster_points = clustered_points[clustered_labels == label]

            # Apply the bounding box estimation
            bbox, angle, area = minimum_bounding_rectangle(cluster_points)

            # Store the result
            rects.append({
                "bbox": bbox,
                "angle": angle,
                "area": area
            })

        return rects
    
    def estimate_bboxes_from_clusters_modest(self, clustered_points, clustered_labels, fit_method="min_zx_area_fit"):
        """
        Estimate bounding boxes for each cluster using fit_2d_box_modest function.
        
        Args:
            clustered_points (np.ndarray): N x 2 array of (x, y) points.
            clustered_labels (np.ndarray): N-length array of cluster labels for the points.
            fit_method (str): Method for bounding box fitting, options include 'min_zx_area_fit', 'PCA', 'variance_to_edge' and 'closeness_to_edge'
        
        Returns:
            List[Dict]: A list of dictionaries where each dict contains 'bbox', 'box_length', 'box_width', and 'ry' for each cluster.
        """
        unique_labels = set(clustered_labels)
        rects = []

        for label in unique_labels:
            if label == -1:  # Skip noise points (if DBSCAN labels noise as -1)
                continue

            # Select points that belong to this cluster
            cluster_points = clustered_points[clustered_labels == label]

            # Add a z-dimension (set to 0) to make it Nx3, as required by fit_2d_box_modest
            cluster_points_3d = np.hstack((cluster_points, np.zeros((cluster_points.shape[0], 1))))

            # Apply the bounding box estimation
            box_center, box_length, box_width, ry = fit_2d_box_modest(cluster_points_3d, fit_method=fit_method)

            # Create the bounding box (center and dimensions)
            rects.append({
                "box_center": box_center,
                "box_length": box_length,
                "box_width": box_width,
                "ry": ry
            })

        return rects