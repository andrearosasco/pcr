import os
from pathlib import Path
from nltk.corpus import wordnet

# After the dataset has been unzipped delete the voxel files
# (They are big and useless)
sp = Path('..') / '..' / 'data' / 'ShapeNetCore.v2'
for file in filter(lambda x: x.suffix == '.binvox', list(sp.rglob('*'))):
    file.unlink(missing_ok=True)

# Create a file mapping labels to directories names and classes names
with (sp / 'classes.txt').open('w') as f:
    dirs = filter(lambda x: x.is_dir(), list(sp.glob('*')))
    for i, dir in enumerate(dirs):
        class_name = wordnet.synset_from_pos_and_offset("n", int(dir.name)).name().split(".")[0]
        print('\n' if i != 0 else '', file=f, end='')
        print(f'{i} {dir.name} {class_name}', file=f, end='')
