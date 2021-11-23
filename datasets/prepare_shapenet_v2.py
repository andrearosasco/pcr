import os
import shutil
from pathlib import Path
import open3d as o3d
import numpy as np
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel(0))
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
sp = Path('..') / '..' / 'data' / 'ShapeNetCore.v2'
for file in filter(lambda x: x.suffix == '.binvox', list(sp.rglob('*'))):
    file.unlink(missing_ok=True)


# # To load just the mesh without attempting to read the textures we make them unreachable
# # Loading certain bad textures makes open3d crash without any possibility of catching the error

sp = Path('..') / '..' / 'data' / 'ShapeNetCore.v2'
i = 0
for cls in sp.glob('*'):
    for inst in cls.glob('*'):
        if (inst / 'images').exists():
            i += 1
            (inst / 'images').rename((inst / 'imgs'))
print('Modified', i, 'instances')

# sp = Path('..') / '..' / 'data' / 'ShapeNetCore.v2'
# i = 0
# j = 0
# for cls in sp.glob('*'):
#     for inst in cls.glob('*'):
#         tm = o3d.io.read_triangle_mesh(str(inst / 'models/model_normalized.obj') , False)
#         if tm.is_watertight():
#             i += 1
#         else:
#             j += 1
#
# print(i)
# print(j)

# # # Create a file mapping labels to directories names and classes names
# sp = Path('..') / '..' / 'data' / 'ShapeNetCore.v2'
# with (sp / 'classes.txt').open('w') as f:
#     dirs = filter(lambda x: x.is_dir(), list(sp.glob('*')))
#     for i, dir in enumerate(dirs):
#         no_examples = len(list(dir.glob('*')))
#         class_name = wordnet.synset_from_pos_and_offset("n", int(dir.name)).name().split(".")[0]
#         print('\n' if i != 0 else '', file=f, end='')
#         print(f'{i} {dir.name} {class_name} {no_examples}', file=f, end='')