import numpy as np
from scipy import spatial
from functools import reduce

# test points


class Pareto(object):
    def __init__(self):

        self.pareto_groups = []

    def filter_(self,pts, pt):
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


    def get_pareto_frontier(self, pts):
        """
        Iteratively filter points based on the convex hull heuristic
        """
        pts = pts.detach().numpy()
        pts = np.array(list(set([tuple(t) for t in pts])))
        # loop while there are points remaining
        while pts.shape[0]:
            # brute force if there are few points:
            if pts.shape[0] < 10:
                self.pareto_groups.append(self.get_pareto_undominated_by(pts))
                break

            # compute vertices of the convex hull
            hull_vertices = spatial.ConvexHull(pts).vertices

            # get corresponding points
            hull_pts = pts[hull_vertices]

            # get points in pts that are not convex hull vertices
            nonhull_mask = np.ones(pts.shape[0], dtype=bool)
            nonhull_mask[hull_vertices] = False
            pts = pts[nonhull_mask]

            # get points in the convex hull that are on the Pareto frontier
            pareto   = self.get_pareto_undominated_by(hull_pts)
            self.pareto_groups.append(pareto)

            # filter remaining points to keep those not dominated by
            # Pareto points of the convex hull
            pts = self.get_pareto_undominated_by(pts, pareto)
            if len(self.pareto_groups)< 20:
                print("pareto point is:",self.pareto_groups)

        return np.vstack(self.pareto_groups)

    # --------------------------------------------------------------------------------
    # previous solutions
    # --------------------------------------------------------------------------------

    def is_pareto_efficient_dumb(self, costs):
        """
        :param costs: An (n_points, n_costs) array
        :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
        """
        is_efficient = np.ones(costs.shape[0], dtype = bool)
        for i, c in enumerate(costs):
            is_efficient[i] = np.all(np.any(costs>=c, axis=1))
        return is_efficient


    def is_pareto_efficient(self, costs):
        """
        :param costs: An (n_points, n_costs) array
        :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
        """
        is_efficient = np.ones(costs.shape[0], dtype = bool)
        for i, c in enumerate(costs):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(costs[is_efficient]<=c, axis=1)  # Remove dominated points
        return is_efficient


    def dominates(self, row, rowCandidate):
        return all(r >= rc for r, rc in zip(row, rowCandidate))


    def cull(self, pts, dominates):
        dominated = []
        cleared = []
        remaining = pts
        while remaining:
            candidate = remaining[0]
            new_remaining = []
            for other in remaining[1:]:
                [new_remaining, dominated][dominates(candidate, other)].append(other)
            if not any(dominates(other, candidate) for other in new_remaining):
                cleared.append(candidate)
            else:
                dominated.append(candidate)
            remaining = new_remaining
        return cleared, dominated

    # --------------------------------------------------------------------------------
    # benchmarking
    # --------------------------------------------------------------------------------

    # to accomodate the original non-numpy solution


