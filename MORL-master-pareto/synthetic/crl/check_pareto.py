import numpy as np
from scipy import spatial
from functools import reduce

# test points

class Pareto(object):

    def __init__(self):

        self.pareto_groups = []

    def filter_(self, pts, pt):
        """
        Get all points in pts that are not Pareto dominated by the point pt
        """
        weakly_worse   = (pts <= pt).all(axis=-1)
        strictly_worse = (pts < pt).any(axis=-1)
        return pts[~(weakly_worse & strictly_worse)]


    def get_pareto_undominated_by(self, pts1, pts2=None):
        """
        Return all points in pts1 that are not Pareto dominated
        by any points in pts2
        """
        if pts2 is None:
            pts2 = pts1
        return reduce(self.filter_, pts2, pts1)


    def get_pareto_frontier(self, Q, length):
        """
        Iteratively filter points based on the convex hull heuristic
        """

        Store = True
        pts = Q.cpu().numpy()
        deque_list = []

        # loop while there are points remaining

            # brute force if there are few points:

        if len(self.pareto_groups) < length:
            self.pareto_groups.append(pts)
        else:
            pts_ = np.reshape(pts, (1, -1))

            pareto_np = np.array(self.pareto_groups)
            # compute vertices of the convex hull
            pareto_np = np.r_[pareto_np,pts_]
            hull_vertices = spatial.ConvexHull(pareto_np).vertices

            # get corresponding points
            hull_pts = pareto_np[hull_vertices]

            # get points in pts that are not convex hull vertices
            nonhull_mask = np.ones(pareto_np.shape[0], dtype=bool)
            nonhull_mask[hull_vertices] = False

            # get points in the convex hull that are on the Pareto frontier
            pareto  = self.get_pareto_undominated_by(hull_pts)
            print(pareto)

            Store = pts_ in pareto


            if Store:
                self.pareto_groups.append(pts)

            for i in range(len(pareto)):
                c = np.where(self.pareto_groups == pareto[i])
                deque_list.append(c[0][0])
                self.pareto_groups = pareto
                self.pareto_groups=self.pareto_groups.tolist()



        return Store, deque_list

    # --------------------------------------------------------------------------------
    # previous solutions
    # --------------------------------------------------------------------------------





    def dominates(row, rowCandidate):
        return all(r >= rc for r, rc in zip(row, rowCandidate))


