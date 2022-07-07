from pathlib import Path

import numpy as np
import open3d as o3d

if __name__ == '__main__':
    data_dir = Path('data/YCB/data')
    split_dir = Path('./data/YCB')

    (split_dir / 'valid.txt').touch()

    lines = []
    for dir in data_dir.glob('*'):

        if (dir / 'google_16k/textured.obj').exists():
            mesh = o3d.io.read_triangle_mesh(str(dir / 'google_16k/textured.obj'))

            np.save(str(dir / 'google_16k/models/model_vertices.npy'), np.array(mesh.vertices))
            np.save(str(dir / 'google_16k/models/model_triangles.npy'), np.array(mesh.triangles))

    #         lines.append(f'{dir.stem}/google_16k')
    #
    #
    # classes = Path('./data/YCB/classes.txt')
    # classes.touch()
    #
    #
    # with classes.open('w+') as out:
    #     for i, l in enumerate(lines):
    #         print(f'{i} {l[:-11]}', file=out)

    # with (split_dir / 'valid.txt').open('w+') as out:
    #     print(*lines, sep='\n', end='', file=out)

