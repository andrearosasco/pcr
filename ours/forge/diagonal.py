import json
import os
import shutil

import numpy as np
import cv2
from sklearn.neighbors import KDTree

from ours.datasets.ShapeNetPOVRemoval import ShapeNet

from configs import DataConfig


def normalizeOBJ(obj, out, stats=None):
    """
    Normalizes OBJ to be centered at origin and fit in unit cube
    """
    if os.path.isfile(out):
        return
    if not stats:
        stats = obj2stats(obj)
    diag = stats['max'] - stats['min']
    norm = 1 / np.linalg.norm(diag)
    c = stats['centroid']
    outmtl = os.path.splitext(out)[0] + '.mtl'
    with open(obj, 'r') as f, open(out, 'w') as fo:
        for line in f:
            if line.startswith('v '):
                v = np.fromstring(line[2:], sep=' ')
                vNorm = (v - c) * norm
                vNormString = 'v %f %f %f\n' % (vNorm[0], vNorm[1], vNorm[2])
                fo.write(vNormString)
            elif line.startswith('mtllib '):
                fo.write('mtllib ' + os.path.basename(outmtl) + '\n')
            else:
                fo.write(line)

    return stats

def obj2stats(obj):
    """
    Computes statistics of OBJ vertices and returns as {num,min,max,centroid}
    """
    minVertex = np.array([np.Infinity, np.Infinity, np.Infinity])
    maxVertex = np.array([-np.Infinity, -np.Infinity, -np.Infinity])
    aggVertices = np.zeros(3)
    numVertices = 0
    with open(obj, 'r') as f:
        for line in f:
            if line.startswith('v '):
                v = np.fromstring(line[2:], sep=' ')
                aggVertices += v
                numVertices += 1
                minVertex = np.minimum(v, minVertex)
                maxVertex = np.maximum(v, maxVertex)
    centroid = aggVertices / numVertices
    info = {}
    info['numVertices'] = numVertices
    info['min'] = minVertex
    info['max'] = maxVertex
    info['centroid'] = centroid
    return info


if __name__ == '__main__':
    config = DataConfig()
    config.dataset_path = '..\\' + config.dataset_path
    dataset = ShapeNet(config, mode="easy/train")
    for label, partial, mesh, samples, occupancy in dataset:
        vertices = mesh[0]
        center = np.mean(vertices, axis=0)
        kdtree = KDTree(vertices)
        dist, idx = kdtree.query(center.reshape(1, -1), k=1)
        vertices[idx]

    stats = obj2stats('../../data/ShapeNetCore.v2/02691156/1bea1445065705eb37abdc1aa610476c/models/model_normalized.obj')
    diag = stats['max'] - stats['min']
    print(max(diag))
