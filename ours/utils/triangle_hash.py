import numpy as np


class TriangleHash:
    def __init__(self, triangles, resolution):
        self.spatial_hash = [[] for _ in range(resolution*resolution)]
        self.resolution = resolution
        self._build_hash(triangles)

    def _build_hash(self, triangles):
        assert(triangles.shape[1] == 3)
        assert(triangles.shape[2] == 2)

        n_tri = triangles.shape[0]
        bbox_min = np.zeros(2)
        bbox_max = np.zeros(2)

        for i_tri in range(n_tri):
            for j in range(2):
                bbox_min[j] = min(triangles[i_tri, 0, j], triangles[i_tri, 1, j], triangles[i_tri, 2, j])
                bbox_max[j] = max(triangles[i_tri, 0, j], triangles[i_tri, 1, j], triangles[i_tri, 2, j])
                bbox_min[j] = min(max(bbox_min[j], 0), self.resolution - 1)
                bbox_max[j] = min(max(bbox_max[j], 0), self.resolution - 1)

            for x in range(int(bbox_min[0]), int(bbox_max[0]) + 1):
                for y in range(int(bbox_min[1]), int(bbox_max[1]) + 1):
                    spatial_idx = self.resolution * x + y
                    self.spatial_hash[spatial_idx].append(i_tri)  # TODO CANNOT PUSH BACK

    def query(self, points):
        assert(points.shape[1] == 2)
        n_points = points.shape[0]

        points_indices = []
        tri_indices = []

        for i_point in range(n_points):
            x = int(points[i_point, 0])
            y = int(points[i_point, 1])
            if not (0 <= x <= self.resolution and 0 <= y < self.resolution):
                continue

            spatial_idx = self.resolution * x + y
            for i_tri in self.spatial_hash[spatial_idx]:
                points_indices.append(i_point)
                tri_indices.append(i_tri)

        points_indices_np = np.zeros(len(points_indices), dtype=np.int32)
        tri_indices_np = np.zeros(len(tri_indices), dtype=np.int32)

        points_indices_view = points_indices_np
        tri_indices_view = tri_indices_np

        for k in range(len(points_indices)):
            points_indices_view[k] = points_indices[k]

        for k in range(len(tri_indices)):
            tri_indices_view[k] = tri_indices[k]

        return points_indices_np, tri_indices_np
