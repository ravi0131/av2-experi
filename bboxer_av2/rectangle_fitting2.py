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
        index_sets = self._adoptive_range_segmentation(ox, oy)

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

            if minp[0] < cost:  # if the cost that I calculated in this loop is greater than the one in previous, then  make this the new 'minp'. Why? => We want the maximum cost
                minp = (cost, theta)

        # calculate best rectangle
        sin_s = np.sin(minp[1])
        cos_s = np.cos(minp[1])

        c1_s = X @ np.array([cos_s, sin_s]).T  # project all points on first edge
        c2_s = X @ np.array([-sin_s, cos_s]).T # project all points on second edge

        rect = RectangleData()
        rect.orientation = minp[1]  #save orientation of the rectangle
        rect.a[0] = cos_s
        rect.b[0] = sin_s
        rect.c[0] = np.min(c1_s)   # store the projection with lowest coordinate w.r.t. first edge
        rect.a[1] = -sin_s
        rect.b[1] = cos_s
        rect.c[1] = np.min(c2_s)
        rect.a[2] = cos_s
        rect.b[2] = sin_s
        rect.c[2] = np.max(c1_s)
        rect.a[3] = -sin_s
        rect.b[3] = cos_s
        rect.c[3] = np.max(c2_s)
        
        # Calculate length and width
        rect.length = np.max(c1_s) - np.min(c1_s)  # Length along the first axis
        rect.width = np.max(c2_s) - np.min(c2_s)   # Width along the second axis
        
        # Calculate centroid (midpoint of projections)
        centroid_x = (np.max(c1_s) + np.min(c1_s)) / 2
        centroid_y = (np.max(c2_s) + np.min(c2_s)) / 2
        rect.centroid = (centroid_x, centroid_y)

        return rect

    def _adoptive_range_segmentation(self, ox: np.ndarray, oy: np.ndarray):
        """
        Args:
        ox(nx1): an array of x coordinates
        oy(nx1): an array of y coordinates 
        """
        points = list(zip(ox, oy))
        tree = KDTree(points)
        clusters = []
        visited = set()

        for i, point in enumerate(points):
            if i not in visited:
                R = self.R0 + self.Rd * np.linalg.norm(point)
                indices = tree.query_ball_point(point, R)
                cluster = set(indices)
                clusters.append(cluster)
                visited.update(indices)

        # Merge overlapping clusters
        # For first iteration , merged_clusters is empty and we simply add the first cluster to it
        # For the rest of the iterations, we take a cluster from 'clusters' and compare it to all the clusters in 'merged_clusters'
        # If they both have common points, then we update the 'mc' from 'merged_clusters' to include the extra points (set union)
        # If not, we simply add this cluster to the list of 'merged_clusters'
        # And the loop goes so on and so forth
        # This means we return a list of distinct clusters for a given set of points (aka subsets are disjoint)
        merged_clusters = []
        while clusters:
            cluster = clusters.pop(0)
            merged = False
            for mc in merged_clusters:
                if cluster & mc:  # If intersection of sets 'cluster' and 'mc' is not empty 
                    mc.update(cluster)
                    merged = True
                    break 
            if not merged:
                merged_clusters.append(cluster)

        return merged_clusters


class RectangleData():

    def __init__(self):
        self.a = [None] * 4
        self.b = [None] * 4
        self.c = [None] * 4

        self.rect_c_x = [None] * 5
        self.rect_c_y = [None] * 5
        
        self.orientation = 0 #angle in radians w.r.t to x-axis in counter-clockwise direction
        self.length = 0
        self.width = 0
        self.centroid = (0,0)
        
        self.calculated_points_flag = False

    import matplotlib.axes as mpl_axes
    def plot(self, ax: mpl_axes):
        """
        Plot the rectangle on the provided axes.
        
        Parameters:
        ax : matplotlib.axes.Axes
            The axes on which to plot the rectangle.
        """
        if self.calculated_points_flag == False:
            self.calc_rect_contour()
        #ax.scatter(self.centroid[0], self.centroid[1], color='blue', label="Centroid")  #to see centroids
        ax.plot(self.rect_c_x, self.rect_c_y, "-r")  # Plotting on the provided Axes
        # ax.text(self.centroid[0], self.centroid[1], f"l={self.length:.2f}, w={self.width:.2f}, θ={self.orientation:.2f}",
        #          fontsize=10, color='green')


    def calc_rect_contour(self):
        self.calculated_points_flag = True
        self.rect_c_x[0], self.rect_c_y[0] = self.calc_cross_point(self.a[0:2], self.b[0:2], self.c[0:2])
        self.rect_c_x[1], self.rect_c_y[1] = self.calc_cross_point(self.a[1:3], self.b[1:3], self.c[1:3])
        self.rect_c_x[2], self.rect_c_y[2] = self.calc_cross_point(self.a[2:4], self.b[2:4], self.c[2:4])
        self.rect_c_x[3], self.rect_c_y[3] = self.calc_cross_point([self.a[3], self.a[0]], [self.b[3], self.b[0]], [self.c[3], self.c[0]])
        self.rect_c_x[4], self.rect_c_y[4] = self.rect_c_x[0], self.rect_c_y[0]
        # print(f"rect_c_x: {self.rect_c_x}, rect_c_y: {self.rect_c_y}")

    def calc_cross_point(self, a, b, c):
        x = (b[0] * -c[1] - b[1] * -c[0]) / (a[0] * b[1] - a[1] * b[0])
        y = (a[1] * -c[0] - a[0] * -c[1]) / (a[0] * b[1] - a[1] * b[0])
        return x, y

    def filter_self(self, min_area=0.4, length=15, width=15):
        if self.length < length or self.width < width:
            return False
        if self.length * self.width < min_area:
            return False
        return True
    
    def contains(self, point):
        """
        Check if a given point (x, y) is inside the rectangle using a ray-casting algorithm.
        The input `point` is expected to be a numpy array or list containing at least two elements [x, y].
        
        Parameters:
        - point: A numpy array or list with at least two elements [x, y]
        
        Returns:
        - True if the point is inside the rectangle, False otherwise
        """
        x, y = point[:2]  # Ensure only the first two elements (x, y) are unpacked
        
        # Rectangle corners (already computed in calc_rect_contour)
        x_coords = self.rect_c_x[:4]
        y_coords = self.rect_c_y[:4]
        
        # Check if the point is inside the polygon using the ray-casting algorithm
        n = len(x_coords)
        inside = False
        
        for i in range(n):
            j = (i + 1) % n
            xi, yi = x_coords[i], y_coords[i]
            xj, yj = x_coords[j], y_coords[j]
            
            # Debugging print statements to check values
            # print(f"xi: {xi}, yi: {yi}, xj: {xj}, yj: {yj}, x: {x}, y: {y}")
            
            # Ensure values are not None before comparison
            if None in (xi, yi, xj, yj, x, y):
                raise ValueError(f"One or more values are None: xi={xi}, yi={yi}, xj={xj}, yj={yj}, x={x}, y={y}")
            
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
        
        return inside
