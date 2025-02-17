import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

def visualize_point_cloud(velo):
    """
    Helper method to visualize a point cloud using open3d
    
    Args:
        velo: A numpy array of shape (n, 3) containing the lidar scan
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(velo)
    o3d.visualization.draw_geometries([pcd])
    

def visualize_point_cloud_dark(velo):
    """
    Helper method to visualize a point cloud using open3d with a dark mode theme.

    Args:
        velo: A numpy array of shape (n, 3) containing the lidar scan.
    """
    # Create a point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(velo)

    # Create a visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Point Cloud Visualization", width=800, height=600)
    
    # Set dark background
    vis.get_render_option().background_color = [0, 0, 0]  # Black background
    
    # Add the point cloud to the visualizer
    vis.add_geometry(pcd)

    # Customize point size and color if needed
    render_option = vis.get_render_option()
    render_option.point_size = 2.0  # Adjust the point size if necessary
    render_option.point_color_option = o3d.visualization.PointColorOption.ZCoordinate  # Color by Z-coordinate

    # Run the visualizer
    vis.run()
    vis.destroy_window()

    
# function to plot a point cloud in bev perspective. The first three columns are x, y, z. The rest should be ignored
# Only plot nothing else
def plot_bev(velo):
    """
    Helper method to plot a point cloud in a bird's eye view perspective
    
    Args:
        velo: A numpy array at least of shape  (n, 3) containing the lidar scan. 
                Actually doesn't matter what shape[1] is, as long as the first 3 columns are x, y, z.
                4th and 5th columns are ignored.
    """
    plt.figure(figsize=(10, 10))
    plt.scatter(velo[:, 0], velo[:, 1], s=1, c=velo[:, 2], cmap='viridis')
    plt.axis('equal')
    plt.show()

def plot_bev_dark(velo):
    """
    Helper method to plot a point cloud in a bird's eye view perspective with a dark mode theme.

    Args:
        velo: A numpy array at least of shape (n, 3) containing the lidar scan.
              Actually doesn't matter what shape[1] is, as long as the first 3 columns are x, y, z.
              4th and 5th columns are ignored.
    """
    # Set up dark mode theme
    plt.style.use('dark_background')
    
    # Plot the data
    plt.figure(figsize=(10, 10), facecolor='black')
    plt.scatter(velo[:, 0], velo[:, 1], s=1, c=velo[:, 2], cmap='viridis')
    plt.axis('equal')
    plt.xlabel('X', color='white')
    plt.ylabel('Y', color='white')
    plt.tick_params(axis='both', colors='white')
    plt.show()

def filter_roi(velo):
    """
    Helper method to filter out points that are not in the region of interest
    
    Args:
        velo: A numpy array of shape (n, 5) containing the lidar scan. Actually doesn't matter what shape[1] is, 
                as long as the first 3 columns are x, y, z. 4th and 5th columns are ignored.
    Returns:
        A numpy array of shape (n, 4) containing the lidar scan
    """
    geom = {
        'L1': -40.0,
        'L2': 40.0,
        'W1': 0.0,
        'W2': 70.0,
        'H1': -2.5,
        'H2': 1.0,
        'input_shape': (800, 700, 36),
        'label_shape': (200, 175, 7)
    }
    q = (geom['W1'] < velo[:, 0]) & (velo[:, 0] < geom['W2']) & \
        (geom['L1'] < velo[:, 1]) & (velo[:, 1] < geom['L2']) & \
        (geom['H1'] < velo[:, 2]) & (velo[:, 2] < geom['H2'])

    indices = np.where(q)[0]
    return velo[indices, :]

def get_bbox_corners(params: Tuple) -> np.ndarray:
    """
    Calculate the four corners of each bounding box from parameters.

    Args:
        bbox_params: A tuples containing (cx, cy, length, width, angle)
                    where cx,cy is center, length/width are dimensions, angle in radians

    Returns:
        corners: A (4,2) numpy array containing the four corners of the bounding box
    """

    cx, cy, length, width, ry = params
    
    # Half-dimensions
    half_length = length / 2
    half_width = width / 2
    
    # Rotation matrix
    rotation_matrix = np.array([
        [np.cos(ry), -np.sin(ry)],
        [np.sin(ry),  np.cos(ry)]
    ])
    
    # Local frame corners in clockwise order
    local_corners = np.array([
        [-half_length,  -half_width],
        [-half_length, half_width],
        [ half_length, half_width],
        [ half_length,  -half_width]
    ])
    
    # Transform to global frame
    global_corners = (rotation_matrix @ local_corners.T).T + np.array([cx, cy])
    return global_corners

def plot_point_cloud_and_bboxes(points: np.ndarray, bboxes: np.ndarray, fig_size= (10, 10)):
    """
    Plots a 2D bird's eye view (BEV) of a point cloud and bounding boxes.

    Args:
        points (np.ndarray): A numpy array of shape (N, 2) representing the point cloud (x, y).
        bboxes (np.ndarray): A numpy array of shape (M, 4, 2) representing bounding boxes 
                             where each box is defined by 4 corners (x, y).
    """
    # Check input dimensions
    assert points.shape[1] == 2, "Points must have shape (N, 2)."
    assert bboxes.shape[1:] == (4, 2), "Bounding boxes must have shape (M, 4, 2)."

    plt.figure(figsize=fig_size)
    
    # Plot the point cloud
    plt.scatter(points[:, 0], points[:, 1], s=1, color='blue', label='Point Cloud', alpha=0.5)

    # Plot each bounding box
    for bbox in bboxes:
        # Append the first corner at the end to close the bounding box
        bbox_closed = np.vstack([bbox, bbox[0]])
        plt.plot(bbox_closed[:, 0], bbox_closed[:, 1], color='red', linewidth=2)

    # Add labels, legends, and grid
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Bird\'s Eye View of Point Cloud and Bounding Boxes')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.axis('equal')  # Maintain aspect ratio
    plt.show()