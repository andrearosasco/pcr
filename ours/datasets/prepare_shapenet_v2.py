import os
import shutil
from pathlib import Path
import open3d as o3d
import tqdm
from nltk.corpus import wordnet

# Some directories do not contain any mesh
sp = Path('..') / '..' / 'data' / 'ShapeNetCore.v2'

for cls in sp.glob('*'):
    for inst in cls.glob('*'):
        if not (inst / 'models/model_normalized.obj').exists():
            shutil.rmtree(inst)

# # # After the dataset has been unzipped delete the voxel files
# # (They are big and useless, just like you)
# sp = Path('..') / '..' / 'data' / 'ShapeNetCore.v2'
# for file in filter(lambda x: x.suffix == '.binvox', list(sp.rglob('*'))):
#     file.unlink(missing_ok=True)
#
# # # Create a file mapping labels to directories names and classes names
# with (sp / 'classes.txt').open('w') as f:
#     dirs = filter(lambda x: x.is_dir(), list(sp.glob('*')))
#     for i, dir in enumerate(dirs):
#         class_name = wordnet.synset_from_pos_and_offset("n", int(dir.name)).name().split(".")[0]
#         print('\n' if i != 0 else '', file=f, end='')
#         print(f'{i} {dir.name} {class_name}', file=f, end='')
#
# # To load just the mesh without attempting to read the textures we make them unreachable
# # Loading certain bad textures makes open3d crash without any possibility of catching the error
# sp = Path('..') / '..' / 'data' / 'ShapeNetCore.v2'
# for file in sp.rglob('*'):
#     if file.name == 'images':
#         file.rename(file.parent / 'imgs')
