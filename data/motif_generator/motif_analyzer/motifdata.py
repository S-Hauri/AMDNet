# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from pymatgen.ext.matproj import MPRester


class MotifReader:
  def __init__(self, mpr_key, mid_list):
    self.common_mids = mid_list
    
    self.mpr = MPRester(mpr_key)
    
    self.site_folder = 'site_features/'
    self.edge_folder = 'node_features/'
        
    
  @property
  def shapes(self):
    data = self.data_sets[0]
    n_embeddings = data['_numbers'].shape[1]
    n_edge_dim = data['_offset'].shape[1]
    shapes = {
        'aid': (None,),
        'seg_m': (None,),
        'idx_ik': (None,),
        'seg_i': (None,),
        'idx_j': (None,),
        'idx_jk': (None,),
        'seg_j': (None,),
        'ratio_j': (None,),
        'numbers': (None, n_embeddings),
        'offset': (None, n_edge_dim), # distance
        'formation_energy_per_atom': (None, 1),
        'band_gap': (None, 1),       
        'positions': (None, 3),
#        'cells': (None, 3, 3)
    }
    return shapes

  @property
  def dtypes(self):
    data = self[1]
    types = {prop: d.dtype for prop, d in data.items()}
    return types

  @property
  def names(self):
    data = self[1]
    return list(data.keys())

  def __len__(self):
    return len(self.common_mids)

    
  def collect_neighbors(self, mid):
    site_name = self.site_folder + mid +'_site_print.txt'
    edge_name = self.edge_folder + mid + '_site_data.txt'
    try:
        site_features = open(site_name, 'r').readlines()[1:]    
        edge_features = open(edge_name, 'r').readlines()[2:]
    except:
        print(f'no site_print or site_data for {mid}')
        return []
    
    if not site_features:
        print(f'no site_print or site_data for {mid}')
        return []
    
    site_features_df = pd.DataFrame([site.split('\t') for site in site_features], \
                                     columns=['mid', 'fts'])
    
    if len(edge_features) == 0:
      edge_features_df = pd.DataFrame({'src':site_features_df.mid.values, \
                                       'tgt':site_features_df.mid.values, \
                                       'type':['']*len(site_features_df), \
                                       'edge_dist11':[0]*len(site_features_df), \
                                       'edge_dist12':[0]*len(site_features_df), \
                                       'edge_dist21':[0]*len(site_features_df), \
                                       'edge_dist22':[0]*len(site_features_df), \
                                       'edge_dist31':[0]*len(site_features_df), \
                                       'edge_dist32':[0]*len(site_features_df), \
                                       'dist':[0]*len(site_features_df)})
    else:
      bad_char = '\'\"[],'
      data_list = []
      for edge in edge_features:
          for b in bad_char:
              edge = edge.replace(b, '')
          line = edge.split()
          edge_rep = [0.]*10
          edge_rep[0] = line[0]
          edge_rep[1] = line[1]
          edge_rep[2] = line[2]
          edge_rep[3] = line[-1]
          # first distance pair
          edge_rep[4] = line[4]
          edge_rep[5] = line[5]          
          if line[2] == 'edge':
              edge_rep[6] = line[7]
              edge_rep[7] = line[8]  
          elif line[2] == 'face':
              edge_rep[6] = line[7]
              edge_rep[7] = line[8]  
              edge_rep[8] = line[10]
              edge_rep[9] = line[11]
              
          data_list.append(edge_rep)
      edge_features_df = pd.DataFrame(data_list)    
      edge_features_df.columns = ['src', 'tgt', 'type', 'dist',
                                  'edge_dist11','edge_dist12', 
                                  'edge_dist21','edge_dist22',
                                  'edge_dist31','edge_dist32']

    motifs = edge_features_df.src.unique()
    motif_to_id = {motif:aid for aid, motif in enumerate(motifs)}
    
    edge_features_df = edge_features_df[(edge_features_df.src.isin(motifs)) & \
                                        (edge_features_df.tgt.isin(motifs))]
    edge_features_df['src_mid'] = [motif_to_id[mot] for mot in edge_features_df.src]
    edge_features_df['tgt_mid'] = [motif_to_id[mot] for mot in edge_features_df.tgt]
    edge_features_df = edge_features_df.sort_values(by=['src_mid', 'tgt_mid'], ascending=[True, True])
    
    type_dict = {'corner': np.array([1,0,0,0]), 'edge': np.array([0,1,0,0]), 
                 'face': np.array([0,0,1,0]), '': np.array([0,0,0,1])}
    
    idx_ik = []
    seg_i = []
    idx_j = []
    seg_j = []
    idx_jk = []
    ratio_j = []
    c_sites = 0
    edge_rep = []
    numbers = [] # the site features
    for i in range(len(site_features_df)):
      ind = edge_features_df[edge_features_df.src_mid == i].tgt_mid.values
      if len(ind) == 0:
        ind = np.array([i])
        dist = np.array([0])
        edge_type = np.array([0, 0, 0, 0])
      else:
        dist = edge_features_df[edge_features_df.src_mid == i].dist.values.astype(np.float32)
        dist = dist.reshape((-1, 1))
        type_str = edge_features_df[edge_features_df.src_mid == i].type.values
        edge_type = [type_dict[t] for t in type_str]
        edge_type = np.array(edge_type)
        
      dist = np.atleast_2d(dist)
      #edge_type = np.atleast_2d(edge_type)
      edge_feat = dist # np.concatenate((dist, edge_type), axis=1)
      uind = np.unique(ind)
      number = np.array(site_features_df.iloc[i]['fts'][1:-2].replace('[', '').split(', ')).astype(np.float32)
      
      idx_ik.append([i] * len(ind))
      seg_i.append([i] * len(uind))
      idx_j.append(uind)
      idx_jk.append(ind)
      edge_rep.append(edge_feat)
      numbers.append(number)
      if len(ind) > 0:
        tmp = np.nonzero(np.diff(np.hstack((-1, ind, np.Inf))))[0]
        rep = np.diff(tmp)
        ratio_j.append(rep / np.sum(rep))
        seg_ij = np.repeat(np.arange(len(uind)), rep) + c_sites
        seg_j.append(seg_ij)
        c_sites = seg_ij[-1] + 1
      else:
        raise Exception(f'isolated atoms in {mid}')

    seg_i = np.hstack(seg_i)
    if len(seg_j) > 0:
      seg_j = np.hstack(seg_j)
    else:
      seg_j = np.array([])
    idx_ik = np.hstack(idx_ik)
    idx_j = np.hstack(idx_j)
    idx_jk = np.hstack(idx_jk)
    ratio_j = np.hstack(ratio_j)
    edge_rep = np.vstack(edge_rep)
    numbers = np.vstack(numbers) # The actual motif feature
    
    bg, fg = self.get_bandgap_fenergy(mid)
    row = {'_idx_ik': idx_ik, 
           '_seg_i': seg_i,
           '_idx_j': idx_j,
           '_idx_jk': idx_jk,
           '_seg_j': seg_j, 
           '_offset': edge_rep,
           '_ratio_j': ratio_j,
           '_numbers': numbers, 
           '_formation_energy_per_atom': np.array([[fg]]),
           '_band_gap': np.array([[bg]])}
    
    return row
    
  def __getitem__(self, idx):
    if type(idx) is int:
      idx = [idx]

    data = {
        'aid': [],
        'seg_m': [],
        'idx_ik': [],
        'seg_i': [],
        'idx_j': [],
        'idx_jk': [],
        'seg_j': [],
        'ratio_j': [],
        'numbers': [],
        'offset': [],
        'formation_energy_per_atom': [],
        'band_gap': [],
        'positions': [np.array([[1, 2, 3]], dtype=np.float32)],
#        'cells': [np.array([])]
    }


    c_atoms = 0
    c_site_segs = 0
    for k, i in enumerate(idx):
      row = self.data_sets[i]
      num_atoms = len(row['_numbers'])
      
      if len(row['_seg_j']) > 0:
        upd_site_segs = row['_seg_j'][-1] + 1
      else:
        upd_site_segs = 0

      data['aid'].append(np.array([i]))
      data['seg_m'].append(np.array([k] * num_atoms).astype(np.int32))
      data['idx_ik'].append((row['_idx_ik'] + c_atoms).astype(np.int32))
      data['seg_i'].append((row['_seg_i'] + c_atoms).astype(np.int32))
      data['idx_j'].append((row['_idx_j'] + c_atoms).astype(np.int32))
      data['idx_jk'].append((row['_idx_jk'] + c_atoms).astype(np.int32))
      data['seg_j'].append((row['_seg_j'] + c_site_segs).astype(np.int32))
      data['offset'].append(row['_offset'].astype(np.float32))
      data['ratio_j'].append(row['_ratio_j'].astype(np.float32))
      data['numbers'].append(row['_numbers'].astype(np.float32))
      data['formation_energy_per_atom'].append(row['_formation_energy_per_atom'].astype(np.float32))
      data['band_gap'].append(row['_band_gap'].astype(np.float32))
#       data['positions'].append(at.get_positions().astype(np.float32))
#       data['cells'].append(at.cell[np.newaxis].astype(np.float32))
      c_atoms += num_atoms
      c_site_segs += upd_site_segs

#      for prop in self.kvp.keys():
#        data[prop].append(np.array([[row[prop]]], dtype=np.float32))
#
#      for prop in self.data_props.keys():
#        data[prop].append(row.data[prop].astype(np.float32))
  
#    for p, b in data.items():
#      print(p, type(b), len(b))
    data = {p: np.concatenate(b, axis=0) for p, b in data.items()}
  
    return data

  def get_property(self, pname, idx):
    if type(idx) is int:
      idx = np.array([idx])
    idx = [self.common_mids[i] + '.cif' for i in idx]
    
#    property = [self.targets[self.targets.mid == i][pname].item() for i in idx]
    property = self.targets.ix[idx][pname].values
    return property

#    def get_atomic_numbers(self, idx):
#        if type(idx) is int:
#            idx = np.array([idx])
#
#        idx = idx + 1
#
#        with connect(self.asedb) as conn:
#            numbers = [conn.get(int(i))['numbers'] for i in idx]
#        return numbers

  def get_number_of_atoms(self, idx):
    if type(idx) is int:
      idx = np.array([idx])
      
#    n_atoms = [len(self.collect_neighbors(self.common_mids[i])['_numbers'])\
#                 for i in idx]
    n_atoms = [len(self.data_sets[i]['_numbers']) for i in idx]
    n_atoms = np.array(n_atoms)
    return n_atoms

  def get_edge_dim_and_info(self, idx):
      edge_dim = self.data_sets[idx[0]]['_offset'].shape[-1]
      all_edges_list = []
      for i in idx:
          edge_feats = self.data_sets[i]['_offset']
          for j in range(edge_feats.shape[0]):
              all_edges_list.append(edge_feats[j])
      edge_feat = np.array(all_edges_list)
      edge_min = np.min(edge_feat, axis=0)
      edge_max = np.max(edge_feat, axis=0)
      
      return edge_dim, edge_min, edge_max

  def get_bandgap_fenergy(self, mid):
    try:
        new_id = self.mpr.get_materials_id_from_task_id(mid)
        dat = self.mpr.get_data(new_id)[0]
        band_gap = dat['band_gap']
        f_energy = dat['formation_energy_per_atom']
    except:
        print(f'Material {mid}: band gap and formation energy not found')
        band_gap = None
        f_energy = None
    return band_gap, f_energy
    
    
    