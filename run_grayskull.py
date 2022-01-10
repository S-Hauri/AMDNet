"""Touch up the conda recipe from grayskull using conda-souschef."""
import os
from os.path import join
from chardet import detect
from souschef.recipe import Recipe

import amdnet as module


def convert_file(srcfile, trgfile=None):
    if trgfile is None:
        trgfile = srcfile
    # get file encoding type
    def get_encoding_type(file):
        with open(file, "rb") as f:
            rawdata = f.read()
        return detect(rawdata)["encoding"]

    from_codec = get_encoding_type(srcfile)

    # add try: except block for reliability
    try:
        with open(srcfile, "r", encoding=from_codec) as f, open(
            trgfile, "w", encoding="utf-8"
        ) as e:
            text = f.read()  # for small files, for big use chunks
            e.write(text)

        os.remove(srcfile)  # remove old encoding file
        os.rename(trgfile, srcfile)  # rename new encoding
    except UnicodeDecodeError:
        print("Decode Error")
    except UnicodeEncodeError:
        print("Encode Error")


files = os.popen("git ls-tree --full-tree -r --name-only HEAD").read().split("\n")

for file in files:
    convert_file(file)

os.system("grayskull pypi {0}=={1}".format(module.__name__, module.__version__))

fpath = join(module.__name__, "meta.yaml")
fpath2 = join("scratch", "meta.yaml")
my_recipe = Recipe(load_file=fpath)
my_recipe["requirements"]["host"].append("flit")
my_recipe.save(fpath)
my_recipe.save(fpath2)
