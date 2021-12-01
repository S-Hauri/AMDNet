import functools
import csv
from pymatgen.core.structure import Structure
# from pathlib import Path
import os


class MotifData:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        # with open(Path(self.dataset_dir, 'mp_ids.csv')) as f:
        with open(os.path.join(self.dataset_dir, 'mp_ids.csv')) as f:
            reader = csv.reader(f)
            self.mp_ids = [row for row in reader]

    def __len__(self):
        return len(self.mp_ids)

    @functools.lru_cache(maxsize=None)
    def __getitem__(self, index):
        cif_id = self.mp_ids[index]
        # filename = str(Path(self.dataset_dir, cif_id))
        filename = os.path.join(self.dataset_dir, cif_id)
        struc = Structure.from_file(filename)
        return struc
