from motif_analyzer.feature import MotifFeature

from os import listdir
from os.path import isfile, join, basename


new_name = '../new_motif_graph.pkl'
mpr_key = 'wxhS4ntBiXkzN6My'


cif_path = 'cif'
node_path = 'node_features'
site_path = 'site_features'
node_header = 'motif_center    next_motif_site    sharing    angles_with_corresponding_distance    distance_motif_site1_next_site'
site_header = 'site   finger_print'

cif_files = [join(cif_path, f) for f in listdir(cif_path) if isfile(join(cif_path, f))]
cif_files = [f for f in cif_files if f.endswith('.cif')]

mid_list = []
for cif_file in cif_files: 
    featurizer = MotifFeature(cif_file)
    fea = featurizer.get_motif_type_and_edge_feature_GH_approach()
    
    comp = basename(cif_file).split('_')[0]
    mid = basename(cif_file).split('_')[1]
    name = mid
    mid_list.append(mid)
    
    with open(join(node_path, name + '_site_data.txt'), 'w+') as f:
        print(node_header, file = f)
        print('', file=f) # line break
        for edge in fea['edge_features']:
            print(*edge, file=f, sep='\t')
    
    with open(join(site_path, name + '_site_print.txt'), 'w+') as f:
        print(site_header, file=f)
        for i, site in enumerate(fea['site_finger_print'][0]):
            print(i, '\t', site, file=f)
            



from motifdata import MotifReader
import pickle as pkl
import tqdm
import re
import pandas as pd




motifgraph_processor = MotifReader(mpr_key, mid_list)

new_data = []
fail_count = 0
regex = re.compile('.[a-z]+-\d+') # some letters followed by a '-' followed by some numbers
for mid in tqdm.tqdm( motifgraph_processor.common_mids ):
    try:
        mat_id = mid
        # structure = motifgraph_processor.mpr.get_structure_by_material_id(mat_id)
        row = motifgraph_processor.collect_neighbors(mid)
        
        new_data.append(row)
    except:
        fail_count += 1

if fail_count > 0:
    print('Could not create features for %i items'%fail_count)

pkl.dump(pd.DataFrame(new_data), open(new_name, 'wb'))






