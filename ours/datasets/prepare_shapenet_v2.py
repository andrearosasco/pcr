import os
from pathlib import Path
import open3d as o3d
import tqdm
from nltk.corpus import wordnet

# # After the dataset has been unzipped delete the voxel files
# # (They are big and useless)
# sp = Path('..') / '..' / 'data' / 'ShapeNetCore.v2'
# for file in filter(lambda x: x.suffix == '.binvox', list(sp.rglob('*'))):
#     file.unlink(missing_ok=True)
#
# # Create a file mapping labels to directories names and classes names
# with (sp / 'classes.txt').open('w') as f:
#     dirs = filter(lambda x: x.is_dir(), list(sp.glob('*')))
#     for i, dir in enumerate(dirs):
#         class_name = wordnet.synset_from_pos_and_offset("n", int(dir.name)).name().split(".")[0]
#         print('\n' if i != 0 else '', file=f, end='')
#         print(f'{i} {dir.name} {class_name}', file=f, end='')

# sp = Path('..') / '..' / 'data' / 'ShapeNetCore.v2'
# for file in sp.rglob('*'):
#     if file.name == 'images':
#         file.rename(file.parent / 'imgs')

sp = Path('..') / '..' / 'data' / 'ShapeNetCore.v2'
# for file in tqdm.tqdm(list(sp.rglob('*'))):
for file in sp.rglob('*'):
    if file.suffix == '.obj':
        try:
            tm = o3d.io.read_triangle_mesh(str(file), True)
            complete_pcd = tm.sample_points_uniformly(10)
        except Exception as e:
            with Path('./out').open('w+') as f:
                print(e.__str__(), file=f)
                print(file, file=f)


