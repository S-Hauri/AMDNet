import os


def test_train_AMDNet():
    os.system(
        "train_AMDnet --material_file data/material_data.pkl --motif_file data/motif_graph.pkl --save_name save/new_model.hdf5"
    )

