from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Input, Concatenate, Add, Embedding, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.backend import int_shape
from tensorflow.keras.models import Model
from amdnet.layers import MEGNetLayer, Set2Set
from amdnet.activations import softplus2
from amdnet.callbacks import ModelCheckpointMAE, ManualStop, ReduceLRUponNan
from amdnet.data.graph import HierarchGraphBatchDistanceConvert
from amdnet.data.crystal import CrystalGraph
from amdnet.utils.preprocessing import DummyScaler
import numpy as np
import os
from monty.serialization import dumpfn, loadfn


class HierarchGraphModel:
    """
    Composition of keras model and converter class for transfering structure
    object to input tensors. We add methods to train the model from
    (structures, targets) pairs

    Args:
        model: (keras model)
        graph_converter: (object) a object that turns a structure to a graph,
            check `model.data.crystal`
        target_scaler: (object) a scaler object for converting targets, check
            `model.utils.preprocessing`
        metadata: (dict) An optional dict of metadata associated with the model.
            Recommended to incorporate some basic information such as units,
            MAE performance, etc.

    """

    def __init__(self,
                 model,
                 atom_graph_converter,
                 motif_graph_converter,
                 target_scaler=DummyScaler(),
                 metadata=None,
                 **kwargs):
        self.model = model
        self.atom_graph_converter = atom_graph_converter
        self.motif_graph_converter = motif_graph_converter
        self.target_scaler = target_scaler
        self.metadata = metadata or {}

    def __getattr__(self, p):
        return getattr(self.model, p)

    def train_from_graphs(self,
                          train_graphs_atom,
                          train_graphs_motif,
                          train_targets,
                          validation_graphs_atom=None,
                          validation_graphs_motif=None,
                          validation_targets=None,
                          epochs=1000,
                          batch_size=128,
                          verbose=1,
                          callbacks=None,
                          prev_model=None,
                          lr_scaling_factor=0.5,
                          patience=500,
                          **kwargs
                          ):

        # load from saved model
        if prev_model:
            self.load_weights(prev_model)
        is_classification = 'entropy' in self.model.loss
        monitor = 'val_acc' if is_classification else 'val_mae'
        mode = 'max' if is_classification else 'min'
        dirname = kwargs.pop('dirname', 'callback')
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        if callbacks is None:
            # with this call back you can stop the model training by `touch STOP`
            callbacks = [ManualStop()]
        train_nb_atoms = [len(i['atom']) for i in train_graphs_atom]
        train_targets = [self.target_scaler.transform(i, j) for i, j in zip(train_targets, train_nb_atoms)]

        if validation_graphs_atom is not None:
            filepath = os.path.join(dirname, '%s_{epoch:05d}_{%s:.6f}.hdf5' % (monitor, monitor))
            val_nb_atoms = [len(i['atom']) for i in validation_graphs_atom]
            validation_targets = [self.target_scaler.transform(i, j) for i, j in zip(validation_targets, val_nb_atoms)]
            inp_atom = self.atom_graph_converter.get_flat_data(validation_graphs_atom, None)
            inp_motif = self.motif_graph_converter.get_flat_data(validation_graphs_motif, validation_targets)
            val_inputs = inp_atom + inp_motif

            val_generator = self._create_generator(*val_inputs,
                                                   batch_size=batch_size)
            steps_per_val = int(np.ceil(len(validation_graphs_atom) / batch_size))
            callbacks.extend([ReduceLRUponNan(filepath=filepath,
                                              monitor=monitor,
                                              mode=mode,
                                              factor=lr_scaling_factor,
                                              patience=patience,
                                              )])
            callbacks.extend([ModelCheckpointMAE(filepath=filepath,
                                                 monitor=monitor,
                                                 mode=mode,
                                                 save_best_only=True,
                                                 save_weights_only=False,
                                                 val_gen=val_generator,
                                                 steps_per_val=steps_per_val,
                                                 target_scaler=self.target_scaler)])
        else:
            val_generator = None
            steps_per_val = None
        
        # ATOM AND MOTIF CONVERT SEPERATELY AND THEN CONCATENATE (TUPLE OUTPUT)
        inp_atom = self.atom_graph_converter.get_flat_data(train_graphs_atom, None)
        inp_motif = self.motif_graph_converter.get_flat_data(train_graphs_motif, train_targets)        
        train_inputs = inp_atom + inp_motif
        # check dimension match
        assert len(train_graphs_atom) == len(train_graphs_motif), \
            print('Training size of atom (%i) and motif (%i) graphs do not match!'
                  %(len(train_graphs_atom), len(train_graphs_motif)))
        train_generator = self._create_generator(*train_inputs, batch_size=batch_size)
        steps_per_train = int(np.ceil(len(train_graphs_atom) / batch_size))
        history = self.fit(train_generator, steps_per_epoch=steps_per_train,
                           validation_data=val_generator, validation_steps=steps_per_val,
                           epochs=epochs, verbose=verbose, callbacks=callbacks, **kwargs)
        return history

    def predict_hierarchical(self, atom_graph, motif_graph):
        """
        Predict property from graph

        Args:
            graph: a graph dictionary, see megnet.data.graph

        Returns:
            predicted target value

        """
        atom_inp = self.atom_graph_converter.graph_to_input(atom_graph)
        motif_inp = self.motif_graph_converter.graph_to_input(motif_graph)
        
        inp = atom_inp + motif_inp # concatenate
        #shapes = [k.shape for k in inp]
        #print(shapes)
        return self.target_scaler.inverse_transform(self.predict(inp).ravel(), len(atom_graph['atom']))

    def convert_inputs(self, atom_graph, motif_graph):
        atom_inp = self.atom_graph_converter.graph_to_input(atom_graph)
        motif_inp = self.motif_graph_converter.graph_to_input(motif_graph)
        
        inp = atom_inp + motif_inp # concatenate
        return inp

    def _create_generator(self, *args, **kwargs):
        kwargs.update({'atom_distance_converter': self.atom_graph_converter.bond_converter})
        kwargs.update({'motif_distance_converter': self.motif_graph_converter.bond_converter})
        return HierarchGraphBatchDistanceConvert(*args, **kwargs)

    def save_model(self, filename):
        """
        Save the model to a keras model hdf5 and a json config for additional
        converters

        Args:
            filename: (str) output file name

        Returns:
            None
        """
        self.model.save(filename)
        dumpfn(
            {
                'atom_graph_converter': self.atom_graph_converter,
                'motif_graph_converter': self.motif_graph_converter,
                'target_scaler': self.target_scaler,
                'metadata': self.metadata
            },
            filename + '.json'
        )

    @classmethod
    def from_file(cls, filename):
        """
        Class method to load model from
            filename for keras model
            filename.json for additional converters

        Args:
            filename: (str) model file name

        Returns
            GraphModel
        """
        configs = loadfn(filename + '.json')
        from keras.models import load_model
        from model.layers import _CUSTOM_OBJECTS
        model = load_model(filename, custom_objects=_CUSTOM_OBJECTS)
        configs.update({'model': model})
        return HierarchGraphModel(**configs)

    @classmethod
    def from_url(cls, url):
        """
        Download and load a model from a URL. E.g.
        https://github.com/materialsvirtuallab/megnet/blob/master/mvl_models/mp-2019.4.1/formation_energy.hdf5

        Args:
            url: (str) url link of the model

        Returns:
            GraphModel
        """
        import urllib.request
        fname = url.split("/")[-1]
        urllib.request.urlretrieve(url, fname)
        urllib.request.urlretrieve(url + ".json", fname + ".json")
        return cls.from_file(fname)


class Hierarch_MEGNetModel(HierarchGraphModel):
    """
    Construct a graph network model with or without explicit atom features
    if n_feature is specified then a general graph model is assumed,
    otherwise a crystal graph model with z number as atom feature is assumed.

    Args:
        nfeat_edge: (int) number of bond features
        nfeat_global: (int) number of state features
        nfeat_node: (int) number of atom features
        nblocks: (int) number of MEGNetLayer blocks
        lr: (float) learning rate
        n1: (int) number of hidden units in layer 1 in MEGNetLayer
        n2: (int) number of hidden units in layer 2 in MEGNetLayer
        n3: (int) number of hidden units in layer 3 in MEGNetLayer
        nvocal: (int) number of total element
        embedding_dim: (int) number of embedding dimension
        nbvocal: (int) number of bond types if bond attributes are types
        bond_embedding_dim: (int) number of bond embedding dimension
        ngvocal: (int) number of global types if global attributes are types
        global_embedding_dim: (int) number of global embedding dimension
        npass: (int) number of recurrent steps in Set2Set layer
        ntarget: (int) number of output targets
        act: (object) activation function
        l2_coef: (float or None) l2 regularization parameter
        is_classification: (bool) whether it is a classification task
        loss: (object or str) loss function
        metrics: (list or dict) List or dictionary of Keras metrics to be evaluated by the model during training and
            testing
        dropout: (float) dropout rate
        graph_converter: (object) object that exposes a "convert" method for structure to graph conversion
        target_scaler: (object) object that exposes a "transform" and "inverse_transform" methods for transforming the
            target values
        optimizer_kwargs (dict): extra keywords for optimizer, for example clipnorm and clipvalue
    """

    def __init__(self,
                 nfeat_edge_atom=None,
                 nfeat_node_atom=None,
                 nfeat_edge_motif=None,
                 nfeat_node_motif=None,
                 nfeat_global=2,
                 nblocks=3,
                 lr=1e-3,
                 n1=64,
                 n2=32,
                 n3=16,
                 n_output_embedding = 32,
                 n_last_layer = 64,
                 nvocal=95,
                 embedding_dim=16,
                 nbvocal=None,
                 bond_embedding_dim=None,
                 ngvocal=None,
                 global_embedding_dim=None,
                 npass=3,
                 ntarget=1,
                 act=softplus2,
                 is_classification=False,
                 loss="mse",
                 metrics=None,
                 l2_coef=None,
                 dropout=None,
                 atom_graph_converter=None,
                 motif_graph_converter=None,
                 target_scaler=DummyScaler(),
                 optimizer_kwargs=None,
                 dropout_on_predict=False
                 ):

        atom_inputs, atom_out = megnet_block(nfeat_edge=nfeat_edge_atom,
                                  nfeat_global=nfeat_global,
                                  nfeat_node=nfeat_node_atom,
                                  nblocks=nblocks,
                                  n1=n1, n2=n2, n3=n3, nvocal=nvocal,
                                  embedding_dim=embedding_dim, nbvocal=nbvocal,
                                  bond_embedding_dim=bond_embedding_dim,
                                  ngvocal=ngvocal,
                                  global_embedding_dim=global_embedding_dim,
                                  npass=npass, ntarget=n_output_embedding, act=act,
                                  l2_coef=l2_coef, dropout=dropout,
                                  dropout_on_predict=dropout_on_predict)
        
        motif_inputs, motif_out = megnet_block(nfeat_edge=nfeat_edge_motif,
                                  nfeat_global=nfeat_global,
                                  nfeat_node=nfeat_node_motif,
                                  nblocks=nblocks,
                                  n1=n1, n2=n2, n3=n3, nvocal=nvocal,
                                  embedding_dim=embedding_dim, nbvocal=nbvocal,
                                  bond_embedding_dim=bond_embedding_dim,
                                  ngvocal=ngvocal,
                                  global_embedding_dim=global_embedding_dim,
                                  npass=npass, ntarget=n_output_embedding, act=act,
                                  l2_coef=l2_coef, dropout=dropout,
                                  dropout_on_predict=dropout_on_predict)
        
        if l2_coef is not None:
            reg = l2(l2_coef)
        else:
            reg = None
        
        merged = Concatenate(axis=-1)([atom_out, motif_out])
        last = Dense(n_last_layer, activation=act, kernel_regularizer=reg)(merged)
        #last = Dense(n_last_layer, activation=act, kernel_regularizer=reg)(atom_out)
        
        if is_classification:
            final_act = 'softmax'
        else:
            final_act = None
        out = Dense(ntarget, activation=final_act, kernel_regularizer=reg)(last)
        
        inputs = atom_inputs + motif_inputs # concatenate lists
        model = Model(inputs=inputs, outputs=out)

        # Compile the model with the optimizer
        loss = 'sparse_categorical_crossentropy' if is_classification else loss

        self.lr = lr
        self.loss = loss
        self.metrics = metrics
        self.optimizer_kwargs = optimizer_kwargs
        opt_params = {'lr': lr}
        if optimizer_kwargs is not None:
            opt_params.update(optimizer_kwargs)
        model.compile(Adam(**opt_params), loss, metrics=metrics)

        if atom_graph_converter is None:
            raise Exception('atom_graph_converter not defined')
            
        if motif_graph_converter is None:
            raise Exception('motif_graph_converter not defined')

        super().__init__(model=model, target_scaler=target_scaler, 
             atom_graph_converter=atom_graph_converter,
             motif_graph_converter=motif_graph_converter)


    def recompile(self):
        opt_params = {'lr': self.lr}
        if self.optimizer_kwargs is not None:
            opt_params.update(self.optimizer_kwargs)
        self.model.compile(Adam(**opt_params), self.loss, metrics=self.metrics)
    

def megnet_block(nfeat_edge=None, nfeat_global=None, nfeat_node=None, nblocks=3,
                      n1=64, n2=32, n3=16, nvocal=95, embedding_dim=16, nbvocal=None,
                      bond_embedding_dim=None, ngvocal=None, global_embedding_dim=None,
                      npass=3, ntarget=1, act=softplus2,
                      l2_coef=None, dropout=None, dropout_on_predict=False):
    """Make a MEGNet block

    Args:
        nfeat_edge: (int) number of bond features
        nfeat_global: (int) number of state features
        nfeat_node: (int) number of atom features
        nblocks: (int) number of MEGNetLayer blocks
        n1: (int) number of hidden units in layer 1 in MEGNetLayer
        n2: (int) number of hidden units in layer 2 in MEGNetLayer
        n3: (int) number of hidden units in layer 3 in MEGNetLayer
        nvocal: (int) number of total element
        embedding_dim: (int) number of embedding dimension
        nbvocal: (int) number of bond types if bond attributes are types
        bond_embedding_dim: (int) number of bond embedding dimension
        ngvocal: (int) number of global types if global attributes are types
        global_embedding_dim: (int) number of global embedding dimension
        npass: (int) number of recurrent steps in Set2Set layer
        ntarget: (int) number of output targets
        act: (object) activation function
        l2_coef: (float or None) l2 regularization parameter
        is_classification: (bool) whether it is a classification task
        dropout: (float) dropout rate
        dropout_on_predict (bool): Whether to use dropout during prediction and training
    Returns:
        (Model) Keras model, ready to run
    """

    # Get the setting for the training kwarg of Dropout
    dropout_training = True if dropout_on_predict else None

    # Create the input blocks
    int32 = 'int32'
    if nfeat_node is None:
        x1 = Input(shape=(None,), dtype=int32)  # only z as feature
        x1_ = Embedding(nvocal, embedding_dim)(x1)
    else:
        x1 = Input(shape=(None, nfeat_node))
        x1_ = x1
    if nfeat_edge is None:
        x2 = Input(shape=(None,), dtype=int32)
        x2_ = Embedding(nbvocal, bond_embedding_dim)(x2)
    else:
        x2 = Input(shape=(None, nfeat_edge))
        x2_ = x2
    if nfeat_global is None:
        x3 = Input(shape=(None,), dtype=int32)
        x3_ = Embedding(ngvocal, global_embedding_dim)(x3)
    else:
        x3 = Input(shape=(None, nfeat_global))
        x3_ = x3
    x4 = Input(shape=(None,), dtype=int32)
    x5 = Input(shape=(None,), dtype=int32)
    x6 = Input(shape=(None,), dtype=int32)
    x7 = Input(shape=(None,), dtype=int32)
    if l2_coef is not None:
        reg = l2(l2_coef)
    else:
        reg = None

    # two feedforward layers
    def ff(x, n_hiddens=[n1, n2]):
        out = x
        for i in n_hiddens:
            out = Dense(i, activation=act, kernel_regularizer=reg)(out)
        return out

    # a block corresponds to two feedforward layers + one MEGNetLayer layer
    # Note the first block does not contain the feedforward layer since
    # it will be explicitly added before the block
    def one_block(a, b, c, has_ff=True):
        if has_ff:
            x1_ = ff(a)
            x2_ = ff(b)
            x3_ = ff(c)
        else:
            x1_ = a
            x2_ = b
            x3_ = c
        out = MEGNetLayer(
            [n1, n1, n2], [n1, n1, n2], [n1, n1, n2],
            pool_method='mean', activation=act, kernel_regularizer=reg)(
            [x1_, x2_, x3_, x4, x5, x6, x7])

        x1_temp = out[0]
        x2_temp = out[1]
        x3_temp = out[2]
        if dropout:
            x1_temp = Dropout(dropout)(x1_temp, training=dropout_training)
            x2_temp = Dropout(dropout)(x2_temp, training=dropout_training)
            x3_temp = Dropout(dropout)(x3_temp, training=dropout_training)
        return x1_temp, x2_temp, x3_temp

    x1_ = ff(x1_)
    x2_ = ff(x2_)
    x3_ = ff(x3_)
    for i in range(nblocks):
        if i == 0:
            has_ff = False
        else:
            has_ff = True
        x1_1 = x1_
        x2_1 = x2_
        x3_1 = x3_
        x1_1, x2_1, x3_1 = one_block(x1_1, x2_1, x3_1, has_ff)
        # skip connection
        x1_ = Add()([x1_, x1_1])
        x2_ = Add()([x2_, x2_1])
        x3_ = Add()([x3_, x3_1])
    # set2set for both the atom and bond
    node_vec = Set2Set(T=npass, n_hidden=n3, kernel_regularizer=reg)([x1_, x6])
    edge_vec = Set2Set(T=npass, n_hidden=n3, kernel_regularizer=reg)([x2_, x7])
    # concatenate atom, bond, and global
    final_vec = Concatenate(axis=-1)([node_vec, edge_vec, x3_])
    if dropout:
        final_vec = Dropout(dropout)(final_vec, training=dropout_training)
    # final dense layers
    final_vec = Dense(n2, activation=act, kernel_regularizer=reg)(final_vec)
    final_vec = Dense(n3, activation=act, kernel_regularizer=reg)(final_vec)
    
    out = Dense(ntarget, activation=act)(final_vec)
    
    inputs = [x1, x2, x3, x4, x5, x6, x7]
    
    return inputs, out
    '''
    model = Model(inputs=[x1, x2, x3, x4, x5, x6, x7], outputs=out)
    return model
    '''