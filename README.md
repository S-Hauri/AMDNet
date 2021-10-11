# AMDNet
 Code base for AMDNet described in https://doi.org/10.1126/sciadv.abf1754

To train AMDNet from scratch, run
python train_AMDnet.py --material_file data/material_data.pkl --motif_file data/motif_graph.pkl --save_name save/bandgap_AMDNet.hdf5

To test the pretrained network, run
python evaluate_AMDnet.py  --material_file data/material_data.pkl --motif_file data/motif_graph.pkl --load_name save/bandgap_AMDNet.hdf5
