# https://github.com/autonomousvision/occupancy_networks/blob/master/scripts/sample_mesh.py
import os
from zipfile import ZipFile
from nltk.corpus import wordnet
from io import BytesIO
import shutil
import tqdm
from configs.cfg1 import DataConfig

if __name__ == "__main__":

    # Create output folder
    data_path = DataConfig.raw_path
    out_path = DataConfig.prep_path
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.mkdir(out_path)
    i = 0

    # Open zip file
    with ZipFile(data_path) as zip_ref:
        # Create dictionary with the index of the class and relative name
        list_of_files = zip_ref.namelist()
        id_objects = {}
        for elem in list_of_files:
            # -1 to take the last element in the path, 0 to take the filename and not the extension
            filename = os.path.splitext(os.path.normpath(elem).split(os.sep)[-1])[0]
            if filename.isnumeric():
                id_objects[filename] = wordnet.synset_from_pos_and_offset("n", int(filename)).name().split(".")[0]

        # Iterate over the 51 classes
        for index in id_objects.keys():
            print("Class: ", id_objects[index])
            zip_data = BytesIO(zip_ref.read("ShapeNetCore.v1/" + index + ".zip"))
            # Open the zip which contains the objects of that class
            with ZipFile(zip_data) as data:
                list_of_files = data.namelist()
                objects = []
                for elem in list_of_files:
                    if elem.endswith(".obj"):
                        objects.append(elem)
                # For each object of that class
                class_path = out_path + os.sep + id_objects[index]
                for elem in tqdm.tqdm(objects):
                    # Create output directory
                    object_path = class_path + os.sep + str(i)
                    os.mkdir(object_path)
                    # Extract mesh and move it in the right folder
                    path = data.extract(elem)
                    os.rename(path, object_path + os.sep + "model.obj")
                    path2 = data.extract(elem[:-4] + ".mtl")
                    os.rename(path2, object_path + os.sep + "model.mtl")
                    # Get images if exist
                    img_path = elem.split("/")[0] + "/" + elem.split("/")[1] + "/" + "images/"
                    if img_path in list_of_files:
                        os.mkdir(object_path + os.sep + "images")
                        images = [elem for elem in list_of_files if elem.startswith(img_path) and elem != img_path]
                        for image in images:
                            path3 = data.extract(image)
                            os.rename(path3, object_path + os.sep + "images" + os.sep + path3.split(os.sep)[-1])
                    # Write label
                    with open(object_path + os.sep + "label.txt", "w") as out_label:
                        out_label.write(id_objects[index])
                    i += 1

                    # Remove object directory
                    shutil.rmtree(os.sep.join(path.split(os.sep)[:-1]))

                # Remove class directory
                shutil.rmtree(os.sep.join(path.split(os.sep)[:-2]))
