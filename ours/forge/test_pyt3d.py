import pytorch3d.io

if __name__ == '__main__':
    verts, faces, aux = pytorch3d.io.load_obj('', load_textures=True)
    pass