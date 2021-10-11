import pickle as pkl
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm

from model.utils.preprocessing import get_graph_from_schnet_data
from monty.serialization import loadfn
from model.hierarch_model import HierarchGraphModel
from tensorflow.keras.models import load_model
from model.layers import _CUSTOM_OBJECTS

parser = argparse.ArgumentParser(description='Evaluate AMD-Net')
parser.add_argument('--material_file',
                    help='dataset with material information', type=str)
parser.add_argument('--motif_file',
                    help='motif information for each material', type=str)
parser.add_argument('--load_name',
                    help='file name of the model', type=str)
parser.add_argument('--predict', default='band_gap',
                    help='attribute to predict (band_gap or formation_energy_per_atom)', type=str)

args = parser.parse_args()

load_name = args.load_name
material_file = args.material_file
motif_file = args.motif_file
predict = args.predict

#load model
config = loadfn(load_name+'.json')
keras_model = load_model(load_name, custom_objects=_CUSTOM_OBJECTS)
model = HierarchGraphModel(**config, model=keras_model)
atom_graph_converter = config['atom_graph_converter']

# load data
(materials, splits) = pkl.load( open(material_file, 'rb') )
train_mids, validate_mids, test_mids = splits

motif_data = pd.DataFrame( pd.read_pickle(motif_file) )
motif_data.set_index('_mid', inplace=True)

print('testing')
# test
e = []
oxide_pred = []
oxide_gt = []
for mid in tqdm(test_mids):
    if mid in motif_data.index:
        try:
            label = motif_data.loc[mid, "_" + predict][0, 0]
            motif_graph, _ = get_graph_from_schnet_data(motif_data.loc[mid])
            atom_graph = atom_graph_converter.convert(materials.loc[mid, 'structure'])
            
            p = model.predict_hierarchical(atom_graph, motif_graph)
            t = materials.loc[mid, predict]
            p = float(p)
            if predict == 'band_gap':
                p = max(0, p)
            e.append(p-label)
            oxide_pred.append(p)
            oxide_gt.append(t)
        except:
            # don't add this
            print(f'could not process {mid}. check for isolated atoms')
            continue
e = np.array(e)
print(e.shape)
print(predict, ': oxide test MAE: ', np.mean(np.abs(e)))
