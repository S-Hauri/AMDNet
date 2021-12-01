import warnings
import networkx as nx
import matplotlib.pyplot as plt
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.core.structure import Structure
warnings.filterwarnings('ignore')


struc = Structure.from_file('../../Desktop/Project/data/dataset-3402/mp-10000.cif')
strategy = CrystalNN()
sg = StructureGraph.with_local_env_strategy(structure=struc, strategy=strategy)
G = sg.graph
nx.draw(G, with_labels=True)
plt.show()
