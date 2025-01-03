import matplotlib.axes as mpl_axes
import numpy as np
from .rectangle_fitting2 import RectangleData
from typing import List, Dict
import matplotlib.pyplot as plt


def clustered_points_viz(clustered_points: np.ndarray, clustered_labels: np.ndarray, ax: mpl_axes.Axes = None):
    """
    Visualize the clustered points in 2D space.
    Args:
    - clustered_points: A nx2 numpy array of clustered points
    - clustered_labels: A nx1 numpy array of cluster labels for each point
    - ax: Matplotlib Axes object to plot on (optional)
    """
    import matplotlib.pyplot as plt
    # Create a colormap instance
    cmap = plt.get_cmap("tab20")  # Use 'tab20' for more colors
    unique_labels = np.unique(clustered_labels)
    colors = [cmap(i / len(unique_labels)) for i in range(len(unique_labels))]

    # Use provided axes or create new ones
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Visualize the 2D clusters
    for label, color in zip(unique_labels, colors):
        if label == -1:
            # Noise points
            color = [0, 0, 0, 1]  # Black color for noise
        mask = (clustered_labels == label)
        ax.scatter(clustered_points[mask, 0], clustered_points[mask, 1], color=color, label=f'Cluster {label}', s=10)

    ax.set_title('2D DBSCAN Clustering')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # Adjust legend position and appearance as needed
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)
    ax.set_aspect('equal')
    # Remove tight_layout and figure adjustments
    # plt.tight_layout()  # Not needed when using ax
    # Do not create new figures or call plt.show() here

def visualize_clusters_with_bboxes(clustered_points, clustered_labels, rects):
    """
    Visualize clusters and their corresponding bounding boxes without showing the legend or angles.
    
    Args:
        clustered_points (np.ndarray): N x 2 array of (x, y) points.
        clustered_labels (np.ndarray): N-length array of cluster labels for the points.
        rects (List[Dict]): List of dictionaries containing bounding boxes, angles, and areas for each cluster.
    """
    # Create a color map for clusters
    unique_labels = set(clustered_labels)
    colors = plt.cm.get_cmap("tab10", len(unique_labels))

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot each cluster
    for label in unique_labels:
        if label == -1:
            # Optionally skip noise points if using DBSCAN
            continue

        # Select points for this cluster
        cluster_points = clustered_points[clustered_labels == label]

        # Plot the points of the cluster (using same color but without labels)
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], s=10, color=colors(label))

    # Plot bounding boxes
    for rect in rects:
        bbox = rect["bbox"]

        # Create a polygon from the bounding box points
        polygon = plt.Polygon(bbox, fill=None, edgecolor='r', linewidth=2)
        ax.add_patch(polygon)

    # Add labels and remove legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Clusters and Bounding Boxes')
    plt.grid(True)
    plt.show()
    
def get_bbox_corners(box_center, box_length, box_width, ry):
    """
    Calculate the 4 corners of the bounding box based on the center, length, width, and rotation angle.
    
    Args:
        box_center (np.ndarray): (x, y) coordinates of the box center.
        box_length (float): Length of the box.
        box_width (float): Width of the box.
        ry (float): Rotation angle in radians.
    
    Returns:
        np.ndarray: 4x2 array representing the coordinates of the 4 corners of the bounding box.
    """
    # Rotation matrix
    rotation_matrix = np.array([[np.cos(ry), -np.sin(ry)], 
                                [np.sin(ry), np.cos(ry)]])
    
    # Half-dimensions
    half_length = box_length / 2
    half_width = box_width / 2
    
    # Define the four corners relative to the center in anti-clockwise order
    corners = np.array([[half_length, half_width],
                        [-half_length, half_width],
                        [-half_length, -half_width],
                        [half_length, -half_width]])
    
    # Rotate the corners and translate them to the box center
    rotated_corners = np.dot(corners, rotation_matrix.T) + box_center
    
    return rotated_corners

def visualize_bboxes_for_modest_fitting(clustered_points, clustered_labels, rects):
    """
    Visualize clusters and their corresponding bounding boxes without showing the legend or angles.
    
    Args:
        clustered_points (np.ndarray): N x 2 array of (x, y) points.
        clustered_labels (np.ndarray): N-length array of cluster labels for the points.
        rects (List[Dict]): List of dictionaries containing box center, length, width, and ry for each cluster.
    """
    # Create a color map for clusters
    unique_labels = set(clustered_labels)
    colors = plt.cm.get_cmap("tab10", len(unique_labels))

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot each cluster
    for label in unique_labels:
        if label == -1:
            # Optionally skip noise points if using DBSCAN
            continue

        # Select points for this cluster
        cluster_points = clustered_points[clustered_labels == label]

        # Plot the points of the cluster (using same color but without labels)
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], s=10, color=colors(label))

    # Plot bounding boxes
    for rect in rects:
        # Get bounding box corners
        bbox_corners = get_bbox_corners(rect["box_center"], rect["box_length"], rect["box_width"], rect["ry"])

        # Create a polygon from the bounding box points
        polygon = plt.Polygon(bbox_corners, fill=None, edgecolor='r', linewidth=2)
        ax.add_patch(polygon)

    # Add labels and remove legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Clusters and Bounding Boxes')
    plt.grid(True)
    plt.show()


    
def visualize_bboxes_for_modest_fitting2(clustered_points, clustered_labels, rects):
    """
    Visualize clusters and their corresponding bounding boxes without showing the legend or angles.
    
    Args:
        clustered_points (np.ndarray): N x 2 array of (x, y) points.
        clustered_labels (np.ndarray): N-length array of cluster labels for the points.
        rects (N,4,2: numpy array of bounding box corners for each cluster.
    """
    # Create a color map for clusters
    unique_labels = set(clustered_labels)
    colors = plt.cm.get_cmap("tab10", len(unique_labels))

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot each cluster
    for label in unique_labels:
        if label == -1:
            # Optionally skip noise points if using DBSCAN
            continue

        # Select points for this cluster
        cluster_points = clustered_points[clustered_labels == label]

        # Plot the points of the cluster (using same color but without labels)
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], s=10, color=colors(label))

    # Plot bounding boxes
    for rect in rects:
        # Create a polygon from the bounding box points
        polygon = plt.Polygon(rect, fill=None, edgecolor='r', linewidth=2)
        ax.add_patch(polygon)

    # Add labels and remove legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Clusters and Bounding Boxes')
    plt.grid(True)
    plt.show()

