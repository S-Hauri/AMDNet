# AMDNet
Code base for AMDNet described in https://doi.org/10.1126/sciadv.abf1754
 
## Description

Incorporation of physical principles in a machine learning (ML) architecture is a fundamental step toward the continued development of artificial intelligence for inorganic materials. As inspired by the Pauling’s rule, we propose that structure motifs in inorganic crystals can serve as a central input to a machine learning framework. We demonstrated that the presence of structure motifs and their connections in a large set of crystalline compounds can be converted into unique vector representations using an unsupervised learning algorithm. To demonstrate the use of structure motif information, a motif-centric learning framework is created by combining motif information with the atom-based graph neural networks to form an atom-motif dual graph network (AMDNet), which is more accurate in predicting the electronic structures of metal oxides such as bandgaps. The work illustrates the route toward fundamental design of graph neural network learning architecture for complex materials by incorporating beyond-atom physical principles.

Architecture:

<img src="https://user-images.githubusercontent.com/51958221/139077101-4bd41f24-f209-4a51-8f7b-579cec81eb77.png" width="500">

AMDNet architecture and materials property predictions.
(A) Demonstration of the learning architecture of the proposed atom-motif dual graph network (AMDNet) for the effective learning of electronic structures and other material properties of inorganic crystalline materials. (B) Comparison of predicted and actual bandgaps [from density functional theory (DFT) calculations] and (C) comparison of predicted and actual formation energies (from DFT calculations) in the test dataset with 4515 compounds.

The code is partially base from the paper "Graph Networks as a Universal Machine Learning Framework for Molecules and Crystals" by Chen et al. https://github.com/materialsvirtuallab/megnet

## Usage

To get started, make sure you are using the same tensorflow and keras versions described in requirements.txt. Furthermore, you should manually download the data files because of the large file sizes.

To train AMDNet from scratch, run
python train_AMDnet.py --material_file data/material_data.pkl --motif_file data/motif_graph.pkl --save_name save/new_model.hdf5

To test the pretrained network, run
python evaluate_AMDnet.py  --material_file data/material_data.pkl --motif_file data/motif_graph.pkl --load_name save/new_model.hdf5

Other parameters: <br>
--material_file: dataset with material information <br>
--motif_file: motif information for each material <br>
--save_name: where to save the model <br>
--predict: attribute to predict (band_gap or formation_energy_per_atom) <br>
--epochs: maximum numbers of epochs <br>
--patience: stop training if no improvement for number of epochs <br>
--learning_rate: learning rate in training <br>
--batch_size: batch size during training <br>
--atom_cutoff: cutoff for atom distance that are considered connected in the graph <br>
--motif_cutoff: cutoff for motif distance that are considered connected in the graph <br>
--rbf_edge_dim_atom: dimension of RBF (radial basis function) for atoms <br>
--rbf_edge_dim_motif: dimension of RBF (radial basis function) for motifs <br>

Due to version changes and limited compatibility to older versions of tensorflow and keras, we can not provide the models used to recreate the results in the publication. However, the provided AMD model performs better than the one used in the publication with the same train/validation/test split. We observe an MAE on the test set of 0.41 (an improvement over the published 0.44).

In some cases, the training does not converge and stops after NaN. In this case, the learning rate is reduced and training proceeds from the best solution (this is the same as in the original source code from MEGNet). In cases where this stops the training early (after less than 200 epochs), we recommend reducing the learning rate and retrying from scratch.
