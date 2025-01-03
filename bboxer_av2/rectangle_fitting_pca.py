import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.axes as mpl_axes

class Rectangle:
    def __init__(self, centroid, length, width, theta):
        self.centroid = centroid
        self.length = length
        self.width = width
        self.theta = theta
    
    def get_corners(self):
        # Calculate the four corners of the rectangle based on centroid, length, width, and theta
        dx = self.length / 2
        dy = self.width / 2

        # Rotation matrix for the given theta
        rotation_matrix = np.array([[np.cos(self.theta), -np.sin(self.theta)],
                                    [np.sin(self.theta),  np.cos(self.theta)]])

        # Corners in the local rectangle frame before rotation
        local_corners = np.array([[-dx, -dy],
                                  [ dx, -dy],
                                  [ dx,  dy],
                                  [-dx,  dy]])

        # Rotate and translate the corners
        rotated_corners = np.dot(local_corners, rotation_matrix.T) + self.centroid
        return rotated_corners
    
    def plot(self, ax: mpl_axes.Axes):
        # Get the corners of the rectangle
        corners = self.get_corners()
        # Close the rectangle by repeating the first point
        corners = np.vstack([corners, corners[0]])

        # Plot the rectangle
        ax.plot(corners[:, 0], corners[:, 1], 'r-')
        #ax.scatter(self.centroid[0], self.centroid[1], color='blue', label="Centroid")
        ax.scatter(self.centroid[0], self.centroid[1], color='blue')

        # # Optionally, you can label the rectangle properties
        # ax.text(self.centroid[0], self.centroid[1], f"l={self.length:.2f}, w={self.width:.2f}, θ={self.theta:.2f}",
        #         fontsize=10, color='green')

class LShapeFitting():

    def fitting(self, ox, oy):
        rects = []
        for i in range(len(ox)):
            x = ox[i]
            y = oy[i]
            rect = self.compute_rectangle_from_cluster(x, y)
            rects.append(rect)
        return rects, None
    
    def compute_rectangle_from_cluster(self,x_points, y_points):
        """
        Compute the bounding rectangle for a cluster of points given x and y coordinates.
        
        :param x_points: numpy array of x coordinates of the cluster points.
        :param y_points: numpy array of y coordinates of the cluster points.
        :return: Rectangle object representing the bounding box.
        """
        # Combine x and y points into a 2D array for PCA
        cluster_points = np.column_stack((x_points, y_points))

        # Same PCA steps as before
        pca = PCA(n_components=2)
        pca.fit(cluster_points)
        
        # Get the principal components
        eigenvectors = pca.components_
        centroid = np.mean(cluster_points, axis=0)
        
        # Project points onto the PCA axes (principal components)
        projected_points = np.dot(cluster_points - centroid, eigenvectors.T)

        # Find the min and max projections on the principal components (to get the bounding box dimensions)
        min_proj = np.min(projected_points, axis=0)
        max_proj = np.max(projected_points, axis=0)

        # Length (l) and Width (w) of the bounding box
        length = max_proj[0] - min_proj[0]
        width = max_proj[1] - min_proj[1]

        # Calculate the angle (theta) of the first principal component w.r.t the x-axis
        theta = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0])

        # Return a Rectangle instance
        return Rectangle(centroid, length, width, theta)


