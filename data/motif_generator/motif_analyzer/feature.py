import warnings
from collections import Counter

import numpy as np
from scipy.spatial.distance import euclidean
from pymatgen.core import Structure
from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import \
    SimplestChemenvStrategy
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import \
    LocalGeometryFinder
from pymatgen.analysis.chemenv.coordination_environments.structure_environments import \
    LightStructureEnvironments
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core.composition import Composition
from matminer.featurizers.structure import SiteStatsFingerprint
from robocrys.condense.fingerprint import get_structure_fingerprint, get_site_fingerprints
from matminer.featurizers.site import CrystalNNFingerprint
from motif_analyzer.site import SiteAnalyzer

warnings.filterwarnings('ignore')

# these are the environment list available in Chem. Mater., 2017, 29 (19), pp 8346-8360
# (Geoffroy Hautier's paper)

MOST_OCCURING_ENVS = {'S:1': 'Single_neighbor', 'L:2': 'Linear', 'A:2': 'Angular', 'TL:3': 'Trigonal_plane',

                      'TY:3': 'Trigonal_non-coplanar', 'TS:3': 'T-shaped', 'T:4': 'Tetrahedron', 'S:4': 'Square_plane',
                      'SY:4': 'Square_non-coplanar', 'SS:4': 'See-Saw', 'PP:5': 'Pentagonal_plane',

                      'S:5': 'Square_pyramid', 'T:5': 'Trigonal_bipyramid', 'O:6': 'Octahedral',
                      'T:6': 'Trigonal_prism',
                      'PP:6': 'Pentagonal_pyramid', 'PB:7': 'Pentagonal_bipyramid', 'ST:7': 'Square_faced_capped_TP',

                      'ET:7': 'End_trigonal_faced_capped_TP', 'FO:7': 'Faced_capped_octahedron', 'C:8': 'Cube',
                      'SA:8': 'Square_antiprism', 'SBT:8': 'Square-face_bicapped_TP',

                      'TBT:8': 'Triangular-face_bicapped_TP', 'DD:8': 'Dodecahedron_WTF',
                      'DDPN:8': 'Dodcahedron_WTF_p2345', 'HB:8': 'Hexagonal_bipyramid', 'BO_1:8': 'Bicapped_octahedron',

                      'BO_2:8': 'Bicapped_oct_OAC', 'BO_3:8': 'Bicapped_oct_OEC', 'TC:9': 'Triangular_cupola',
                      'TT_1:9': 'Tricapped_TP_TSF', 'TT_2:9': 'T_TP_TSF', 'TT_3:9': 'T_TP_OSF',
                      'HD:9': 'Heptagonal_dipyramid', 'TI:9': 'TI9', 'SMA:9': 'SMA9',

                      'SS:9': 'SS9', 'TO_1:9': 'TO19', 'TO_2:9': 'TO29', 'TO_3:9': 'TO3_9', 'PP:10': 'Pentagonal_prism',
                      'PA:10': 'Pentagonal_antiprism', 'SBSA:10': 'S-fBSA', 'MI:10': 'MI', 'S:10': 'S10',
                      'H:10': 'Hexadec',

                      'BS_1:10': 'BCSP_of', 'BS_2:10': 'BCSP_af', 'TBSA:10': 'TBSA',

                      'PCPA:11': 'PCPA', 'H:11': 'HDech', 'SH:11': 'SPHend', 'CO:11': 'Cs-oct', 'DI:11': 'Dimmi_icso',
                      'I:12': 'ICOSh', 'PBP:12': 'PBP12',

                      'TT:12': 'TT', 'C:12': 'Cuboctahedral', 'AC:12': 'ANTICUBOOCT', 'SC:12': 'SQU_cupola',
                      'S:12': 'Sphenemogena', 'HP:12': 'Hexagonal_prism', 'HA:12': 'Hexagonal_anti_prism',

                      'SH:13': 'SH13'}

MOTIF_TYPE_NUMBERING = {name: i for i, name in enumerate(MOST_OCCURING_ENVS.values(), 1)}

STRATEGY = SimplestChemenvStrategy(distance_cutoff=1.4, angle_cutoff=0.3, additional_condition=3)


class MotifFeature:

    def __init__(self, filename):
        self.comp = filename.split('.')[0].split('_')[0]
        self.structure = Structure.from_file(filename)
        # Only oxygen is considered as anion in this script
        self.anion = list(self.structure[-1].species_string)
        self.final_elem_list = []
        for i in range(len(self.structure)):
            specie = self.structure[i].species_string
            if specie not in self.anion:
                self.final_elem_list.append(specie)

        lgf = LocalGeometryFinder()
        lgf.setup_structure(structure=self.structure)
        self.structure_environments = lgf.compute_structure_environments(
            maximum_distance_factor=1.41, only_cations=True, excluded_atoms=['O'])
        self.light_structure_environments = LightStructureEnvironments.from_structure_environments(
            strategy=STRATEGY, structure_environments=self.structure_environments)

        self.get_motif_type()

    def get_composition_from_structure(self):
        return self.structure.composition

    def get_types_of_species(self):
        return self.structure.types_of_specie

    def get_site_finger_print(self):

        each_motif_site_print = [list(site_print.values()) for site_print
                                 in get_site_fingerprints(self.structure)[: len(self.final_elem_list)]]
        # average_finger_print = [i for i in map(np.average, zip(*each_motif_site_print))]
        return each_motif_site_print,  # average_finger_print

    def get_structure_finger_print(self):
        return get_structure_fingerprint(self.structure)

    def get_motif_type(self):
        """
        Get motif_analyzer type list based on Chem. Mater., 2017, 29 (19), pp 8346-8360 (Geoffroy Hautier's paper)
        """
        coordination_env_list = self.light_structure_environments.coordination_environments[:self.structure.num_sites]
        motif_type_list = []
        for env in coordination_env_list:
            if env is None:
                continue
            result = min(env, key=lambda x: x['csm'])
            ce_symbol = result['ce_symbol']
            # if ce_symbol in MOST_OCCURING_ENVS.keys():
            #     motif_type = MOST_OCCURING_ENVS[ce_symbol]
            #     motif_type_list.append(motif_type)
            motif_type = MOST_OCCURING_ENVS.get(ce_symbol)
            if motif_type:
                motif_type_list.append(motif_type)

        self.motif_types = motif_type_list

    @staticmethod
    def vectorize_composition(composition):
        """
        Convert motifs in one hot representation according to the atomic number
        Example: H2O will be [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,......., 0]

        The length of the vector will be maximum atomic number of the element present in the dataset
        """
        vectorize_composition = np.zeros(95)
        comp = Composition(composition)
        for element in comp:
            vectorize_composition[element.Z - 1] = 1

        # number_of_atoms_in_composition = comp.num_atoms
        # average_electronegativity = comp.average_electroneg
        # number_of_different_elements_in_comp = comp.to_data_dict["nelements"]
        # atomic_weights_in_comp = [e.Z for e in comp.elements]
        return vectorize_composition
        # , number_of_atoms_in_composition, average_electronegativity,
        # number_of_different_elements_in_comp, atomic_weights_in_comp

    def get_composition(self):
        """
        Changes composition to one hot representation on the basis of atomic number
        Example: MnO6 will get vector with 1 at 8 and 25 and all other 0 with total lenght 95
        described in vectorize composition
        """
        structure_environments = self.structure_environments
        new_list = self.motif_types
        STRATEGY.set_structure_environments(structure_environments)

        oxygen_sites, neighbor_finding, central_atom1 = [], [], []
        for site in self.structure[: len(new_list)]:
            central_atom1.append(site.species_string)

            surrounding_oxygen = STRATEGY.get_site_neighbors(site)
            surrounding_atoms = [i.species_string for i in surrounding_oxygen]
            neighbor_finding.append(surrounding_atoms)

        neighbor_finding = [sorted(i) for i in neighbor_finding]
        final_key_list = [str(i) + ''.join('%s%d' % t for t in Counter(j).items()) for i, j in
                          zip(central_atom1, neighbor_finding)]

        composition_vector_one_hot_dict = {}
        # considering all compositions with labelled numbers
        for i, comp in enumerate(final_key_list):
            composition_vector_one_hot_dict[comp + "_" + str(i)] = self.vectorize_composition(comp)

        return composition_vector_one_hot_dict

    def get_connection_type_and_edge_feature(self):
        """
        Neighboring connection of motifs
        """
        connection_graph = {}
        bonded_structure = CrystalNN().get_bonded_structure(self.structure)
        sag = SiteAnalyzer(bonded_structure).get_next_nearest_neighbors
        edge_features = []
        for i, ele in enumerate(self.final_elem_list):
            connection_list = []
            nnns = sag(i)

            for nnn in nnns:
                connection = nnn['connectivity']
                if connection == 'corner':
                    connection_list.append(1)
                elif connection == 'edge':
                    connection_list.append(2)
                else:
                    connection_list.append(3)

                x = nnn['site_coords']
                y = nnn['nn_site_coords']
                z = nnn['nnn_site_coords']
                ang1 = nnn['angles']
                distance_between_site = [[euclidean(x, j), euclidean(j, z)]
                                         for j in y]
                distance_between_site_nnn_site = euclidean(x, z)
                angle_distance = ['  '.join(map(str, i)) for i in zip(ang1, distance_between_site)]
                edge_features.append([ele + str(i), nnn['element'], nnn['connectivity'], angle_distance,
                                      distance_between_site_nnn_site])

            connection_graph[ele + str(i)] = connection_list

        return connection_graph, edge_features

    def get_motif_type_and_edge_feature_GH_approach(self):
        """
        motif_analyzer type and final dictionary with various properties
        """
        connection_type, edge_features = self.get_connection_type_and_edge_feature()
        comp_motif = {'motif_types': self.motif_types,
                      'structure_finger_print': self.get_structure_finger_print(),
                      'site_finger_print': self.get_site_finger_print(),
                      'compositions': self.get_composition(),
                      'connection_type': connection_type,
                      'edge_features': edge_features}

        return comp_motif


if __name__ == '__main__':
    def vectorize(motif):
        vector = np.zeros(len(MOTIF_TYPE_NUMBERING) + 1)
        for keys in MOTIF_TYPE_NUMBERING.keys():
            if keys == motif:
                vector[MOTIF_TYPE_NUMBERING[str(keys)] - 1] = 1
        return vector


    v = vectorize('Octahedral')
