import json

import open3d as o3d
from pathlib import Path
import numpy as np
import tqdm

if __name__ == '__main__':
    root = Path('data/MCD')
    with (root / 'build_datasets/train_test_dataset.json').open('r') as f:
        t = json.load(f)

    diameters = []
    names = []
    for el in tqdm.tqdm({e[1] for e in t['train_models_train_views']}):
        triangles = np.load((root / (el + 'triangles.npy')))
        vertices = np.load((root / (el + 'vertices.npy')))


        mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(vertices), triangles=o3d.utility.Vector3iVector(triangles))
        pc = np.array(mesh.sample_points_uniformly(100_000).points)

        if not 'ycb' in el:
            diameters += [np.max(pc.max(0) - pc.min(0))]
            names += [el]

    print()