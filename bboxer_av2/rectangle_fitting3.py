import matplotlib.pyplot as plt
import numpy as np
import itertools
from enum import Enum
from scipy.spatial import KDTree
import concurrent.futures

show_animation = True


class LShapeFitting():

    class Criteria(Enum):
        AREA = 1
        CLOSENESS = 2
        VARIANCE = 3

    def __init__(self):
        # Parameters
        self.criteria = self.Criteria.VARIANCE  # According to paper, variance criterion is the best
        self.min_dist_of_closeness_crit = 0.01  # [m]
        self.dtheta_deg_for_serarch = 1.0  # [deg]
        self.R0 = 3.0  # [m] range segmentation param
        self.Rd = 0.001  # [m] range segmentation param

    def fitting(self, ox: np.ndarray, oy: np.ndarray):
        """
        Args:
        ox(nx1): an array of x coordinates
        oy(nx1): an array of y coordinates 
        """
        # Adaptive Range Segmentation using KD-Tree
        index_sets = self._adaptive_range_segmentation(ox, oy)

        # Rectangle search using parallelization
        rects = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._rectangle_search, [ox[i] for i in ids], [oy[i] for i in ids]) for ids in index_sets]
            rects = [future.result() for future in concurrent.futures.as_completed(futures)]

        return rects, index_sets

    def _calc_area_criterion(self, c1, c2):
        c1_max = np.max(c1)
        c2_max = np.max(c2)
        c1_min = np.min(c1)
        c2_min = np.min(c2)

        alpha = -(c1_max - c1_min) * (c2_max - c2_min)

        return alpha

    def _calc_closeness_criterion(self, c1, c2):
        c1_max = np.max(c1)
        c2_max = np.max(c2)
        c1_min = np.min(c1)
        c2_min = np.min(c2)

        D1 = np.minimum(np.abs(c1_max - c1), np.abs(c1 - c1_min))
        D2 = np.minimum(np.abs(c2_max - c2), np.abs(c2 - c2_min))

        d = np.maximum(np.minimum(D1, D2), self.min_dist_of_closeness_crit)
        beta = np.sum(1.0 / d)

        return beta

    def _calc_variance_criterion(self, c1, c2):
        c1_max = np.max(c1)
        c2_max = np.max(c2)
        c1_min = np.min(c1)
        c2_min = np.min(c2)

        D1 = np.minimum(np.abs(c1_max - c1), np.abs(c1 - c1_min))
        D2 = np.minimum(np.abs(c2_max - c2), np.abs(c2 - c2_min))

        E1 = D1[D1 < D2]
        E2 = D2[D2 <= D1]

        V1 = -np.var(E1) if E1.size > 0 else 0.0
        V2 = -np.var(E2) if E2.size > 0 else 0.0

        gamma = V1 + V2

        return gamma

    def _rectangle_search(self, x, y):
        X = np.array([x, y]).T

        dtheta = np.deg2rad(self.dtheta_deg_for_serarch)
        minp = (-float('inf'), None)
        for theta in np.arange(0.0, np.pi / 2.0, dtheta):

            e1 = np.array([np.cos(theta), np.sin(theta)])
            e2 = np.array([-np.sin(theta), np.cos(theta)])

            c1 = X @ e1.T
            c2 = X @ e2.T

            # Select criteria
            if self.criteria == self.Criteria.AREA:
                cost = self._calc_area_criterion(c1, c2)
            elif self.criteria == self.Criteria.CLOSENESS:
                cost = self._calc_closeness_criterion(c1, c2)
            elif self.criteria == self.Criteria.VARIANCE:
                cost = self._calc_variance_criterion(c1, c2)

            if minp[0] < cost:
                minp = (cost, theta)

        # calculate best rectangle
        sin_s = np.sin(minp[1])
        cos_s = np.cos(minp[1])

        c1_s = X @ np.array([cos_s, sin_s]).T
        c2_s = X @ np.array([-sin_s, cos_s]).T

        rect = RectangleData()
        rect.a[0] = cos_s
        rect.b[0] = sin_s
        rect.c[0] = np.min(c1_s)
        rect.a[1] = -sin_s
        rect.b[1] = cos_s
        rect.c[1] = np.min(c2_s)
        rect.a[2] = cos_s
        rect.b[2] = sin_s
        rect.c[2] = np.max(c1_s)
        rect.a[3] = -sin_s
        rect.b[3] = cos_s
        rect.c[3] = np.max(c2_s)

        return rect
    
    def _adaptive_range_segmentation(self, ox: np.ndarray, oy: np.ndarray):
        """
        Args:
        ox (nx1): an array of x coordinates
        oy (nx1): an array of y coordinates
        """
        # Combine ox and oy into a single n x 2 array of points
        points = np.column_stack((ox, oy))
        
        # Initialize the KD-Tree and other variables
        tree = KDTree(points)
        clusters = []  # To hold the final clusters
        visited = set()  # To track visited points
        
        # Iterate over all points
        for i, point in enumerate(points):
            if i not in visited:
                # Initialize a new cluster and a queue for breadth-first expansion
                cluster = set()
                queue = [i]
                
                while queue:
                    # Pop the next point index from the queue
                    idx = queue.pop(0)
                    if idx not in visited:
                        # Mark the point as visited and add it to the cluster
                        visited.add(idx)
                        cluster.add(idx)
                        
                        # Calculate the adaptive radius R for this point
                        R = self.R0 + self.Rd * np.linalg.norm(points[idx])
                        
                        # Find all points within this adaptive radius
                        neighbors = tree.query_radius([points[idx]], r=R)[0]
                        
                        # Add any unvisited neighbors to the queue for further processing
                        for neighbor in neighbors:
                            if neighbor not in visited:
                                queue.append(neighbor)
                
                # Once the queue is empty, we have found all the points in the current cluster
                clusters.append(cluster)
        
        # Return the clusters as a list of sets of point indices
        return clusters


class RectangleData():

    def __init__(self):
        self.a = [None] * 4
        self.b = [None] * 4
        self.c = [None] * 4

        self.rect_c_x = [None] * 5
        self.rect_c_y = [None] * 5


    def plot(self, ax):
        """
        Plot the rectangle on the provided axes.
        
        Parameters:
        ax : matplotlib.axes.Axes
            The axes on which to plot the rectangle.
        """
        self.calc_rect_contour()
        ax.plot(self.rect_c_x, self.rect_c_y, "-r")  # Plotting on the provided Axes


    def calc_rect_contour(self):
        self.rect_c_x[0], self.rect_c_y[0] = self.calc_cross_point(self.a[0:2], self.b[0:2], self.c[0:2])
        self.rect_c_x[1], self.rect_c_y[1] = self.calc_cross_point(self.a[1:3], self.b[1:3], self.c[1:3])
        self.rect_c_x[2], self.rect_c_y[2] = self.calc_cross_point(self.a[2:4], self.b[2:4], self.c[2:4])
        self.rect_c_x[3], self.rect_c_y[3] = self.calc_cross_point([self.a[3], self.a[0]], [self.b[3], self.b[0]], [self.c[3], self.c[0]])
        self.rect_c_x[4], self.rect_c_y[4] = self.rect_c_x[0], self.rect_c_y[0]

    def calc_cross_point(self, a, b, c):
        x = (b[0] * -c[1] - b[1] * -c[0]) / (a[0] * b[1] - a[1] * b[0])
        y = (a[1] * -c[0] - a[0] * -c[1]) / (a[0] * b[1] - a[1] * b[0])
        return x, y

