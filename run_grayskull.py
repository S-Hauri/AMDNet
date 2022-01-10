"""Touch up the conda recipe from grayskull using conda-souschef."""
import os
from os.path import join

from souschef.recipe import Recipe

import amdnet as module

os.system("grayskull pypi {0}=={1}".format(module.__name__, module.__version__))

fpath = join(module.__name__, "meta.yaml")
fpath2 = join("scratch", "meta.yaml")
my_recipe = Recipe(load_file=fpath)
my_recipe["requirements"]["host"].append("flit")
my_recipe.save(fpath)
my_recipe.save(fpath2)
