from pathlib import Path

# After the dataset has been unzipped delete the voxel files
# (They are big and useless)
sp = Path('..') / '..' / 'data' / 'ShapeNetCore.v2'
for file in filter(lambda x: x.suffix == '.binvox', list(sp.rglob('*'))):
    file.unlink()