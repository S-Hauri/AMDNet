from model.hierarch_model import Hierarch_MEGNetModel, HierarchGraphModel
from model.data.graph import GaussianDistance
from model.data.crystal import CrystalGraph
from model.utils.preprocessing import get_graph_from_schnet_data
from model.layers import _CUSTOM_OBJECTS
from tensorflow.keras.backend import relu
from tensorflow.keras.models import load_model

import pandas as pd
import numpy as np
import pickle as pkl
import shutil
import os
import glob
import argparse

from numpy.random import seed
import tensorflow as tf
seed(999)
tf.random.set_seed(42)

"""
Parse input arguments
"""
parser = argparse.ArgumentParser(description='Train AMD-Net')
parser.add_argument('--material_file',
                    help='dataset with material information', type=str)
parser.add_argument('--motif_file',
                    help='motif information for each material', type=str)
parser.add_argument('--save_name',
                    help='where to save the model', type=str)
parser.add_argument('--predict', default='band_gap',
                    help='attribute to predict (band_gap or formation_energy_per_atom)', type=str)
parser.add_argument('--epochs', default=1000,
                    help='maximum numbers of epochs', type=int)
parser.add_argument('--patience', default=100,
                    help='stop training if no improvement for number of epochs', type=int)
parser.add_argument('--learning_rate', default=0.00005,
                    help='learning rate in training', type=float)
parser.add_argument('--batch_size', default=32,
                    help='batch size during training', type=int)
parser.add_argument('--atom_cutoff', default=5.,
                    help='cutoff for atom distance that are considered connected in the graph', type=float)
parser.add_argument('--motif_cutoff', default=11.,
                    help='cutoff for motif distance that are considered connected in the graph', type=float)
parser.add_argument('--rbf_edge_dim_atom', default=100,
                    help='dimension of RBF (radial basis function) for atoms', type=int)
parser.add_argument('--rbf_edge_dim_motif', default=100,
                    help='dimension of RBF (radial basis function) for motifs', type=int)

args = parser.parse_args()

training_patience = args.patience
max_epochs = args.epochs
predict = args.predict
atom_cutoff = args.atom_cutoff
motif_cutoff = args.motif_cutoff
rbf_edge_dim_atom = args.rbf_edge_dim_atom
rbf_edge_dim_motif = args.rbf_edge_dim_motif
learning_rate = args.learning_rate
batch_size = args.batch_size
material_file = args.material_file
motif_file = args.motif_file
save_name = args.save_name

def load_and_save_best(model):
    # load best model
    list_of_files = glob.glob('callback/*')
    latest_file = max(list_of_files, key=os.path.getctime)
    print(latest_file)
    loaded_model = load_model(latest_file, custom_objects=_CUSTOM_OBJECTS)
    config = {
        'atom_graph_converter': model.atom_graph_converter,
        'motif_graph_converter': model.motif_graph_converter,
        'target_scaler': model.target_scaler,
        'metadata': model.metadata, 
        'model': loaded_model
    }
    best_model = HierarchGraphModel(**config)
    
    best_model.save_model(save_name)
    
    return best_model

     
# load data
(materials, splits) = pkl.load( open(material_file, 'rb') )
train_mids, validate_mids, test_mids = splits

atom_graph_converter = CrystalGraph(cutoff=atom_cutoff, 
    bond_converter=GaussianDistance(np.linspace(0, atom_cutoff, rbf_edge_dim_atom), 0.5))

motif_graph_converter = CrystalGraph(cutoff=motif_cutoff, 
    bond_converter=GaussianDistance(np.linspace(0, motif_cutoff, rbf_edge_dim_motif), 0.5))


print('get data')
motif_data = pd.DataFrame( pd.read_pickle(motif_file) )
motif_data.set_index('_mid', inplace=True)

print('convert data')
train_graphs_motif = []
train_graphs_atom = []
train_targets = []
mids = train_mids
for mid in mids:
    if mid in motif_data.index:
        try:
            label = motif_data.loc[mid, "_" + predict][0, 0]            
            motif_graph, motif_dim = get_graph_from_schnet_data(motif_data.loc[mid])            
            atom_graph = atom_graph_converter.convert( materials.loc[mid, 'structure'] )
            
            train_targets.append(label)
            train_graphs_motif.append(motif_graph)
            train_graphs_atom.append(atom_graph)
        except:
            # isolated atoms, don't use this material
            print(f'could not process {mid}. check for isolated atoms')
            continue

validate_graphs_motif = []
validate_graphs_atom = []
validate_targets = []
mids = validate_mids
for mid in mids:
    if mid in motif_data.index:
        try:
            label = motif_data.loc[mid, "_" + predict][0, 0]
            graph, motif_dim = get_graph_from_schnet_data(motif_data.loc[mid])            
            atom_graph = atom_graph_converter.convert( materials.loc[mid, 'structure'] )
            
            validate_targets.append(label)
            validate_graphs_motif.append(graph)
            validate_graphs_atom.append(atom_graph)
        except:
            # isolated atoms, don't use this material
            continue
print('initializing models...')

model = Hierarch_MEGNetModel(nfeat_edge_atom=rbf_edge_dim_atom,
                             nfeat_node_atom=None,
                             nfeat_edge_motif=rbf_edge_dim_motif,
                             nfeat_node_motif=motif_dim,
                             n_output_embedding = 16,
                             n_last_layer = 32,
                             lr = learning_rate,
                             act = relu,
                             atom_graph_converter=atom_graph_converter,
                             motif_graph_converter=motif_graph_converter)

# before training: delete callback folder to make sure early stopping works
if os.path.exists('callback/'):
    shutil.rmtree('callback/')
os.makedirs('callback/')

# train
history = model.train_from_graphs(train_graphs_atom, train_graphs_motif, train_targets,
                        validate_graphs_atom, validate_graphs_motif, validate_targets,
                        epochs=max_epochs, batch_size = batch_size, 
                        patience = training_patience, verbose=2)

best_model = load_and_save_best(model)


