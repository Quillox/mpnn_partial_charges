import os
import sys
import time
from functools import partial
from pathlib import Path
# import multiprocessing
import torch.multiprocessing
# import torch.multiprocessing as multiprocessing

import torch 
import dgl 
from dgl.data import DGLDataset

import rdkit
from rdkit import Chem
# from rdkit.Chem import PropertyMol

from param_ml.utils.utils import mol2_blocks_from_mol2_file, mol_graph_and_features, mol_from_mol2_block
from param_ml.utils.utils import count_and_tell, wrapper_mol_from_mol2_block
from param_ml.utils.featurizers import ElementAtomFeaturizer

class MolDataset(DGLDataset):
    '''Dataset of molecules that are read from a mol2 file.
    TODO: extend to prepare directly from SQL database.

    Parameters
    ----------

    name : str
    '''
    def __init__(self, name, 
                 mol_file,
                 raw_dir,
                 save_dir,
                 num_processes = None,
                 mol_selection = None,
                 sanitize=True,
                 remove_hs=False,
                 atom_featurizer=None,
                 bond_featurizer=None,
                 force_reload=False,
                 simple_read=False,
                 in_memory=True,
                 ignore_lps=False,
                 verbose=False):
        
        if not all(map(os.path.exists, [raw_dir, save_dir, mol_file])):
            raise FileNotFoundError(f'Input file or directories not existing.')


        self._mol_file = mol_file
        if not num_processes:
            num_processes = os.cpu_count()
        self._num_processes = num_processes
        self._sanitize = sanitize 
        self._remove_hs = remove_hs 
        self._atom_featurizer = atom_featurizer
        self._bond_featurizer = bond_featurizer
        if isinstance(mol_selection, int):
            self._mol_selection = list(range(mol_selection))
        self._mol_selection = mol_selection
        self._simple_read = simple_read
        self._in_memory = in_memory
        self._ignore_lps = ignore_lps

        self._parameters = {
            "mol_file": self._mol_file,
            "sanitize": self._sanitize,
            "remove_hs": self._remove_hs,
            "mol_selection": self._mol_selection,
            "ignore_lps": self._ignore_lps
            # "atom_featurizer": self._atom_featurizer # This we cannot save!
        }
        
        super().__init__(name=name,
                       url=None,
                       raw_dir=raw_dir,
                       save_dir=save_dir,
                       force_reload=force_reload,
                       verbose=verbose)

    def has_cache(self):
        existing = all(map(os.path.exists, [self.graph_file, self.info_file]))
        if existing:
            # Check that parameters are the same:
            print(f"Checking if the given parameters and the stored ones are equal...")
            saved_parameters = dgl.data.utils.load_info(self.info_file)['parameters']
            if saved_parameters != self.parameters:
                print("Given parameters:")
                print(self.parameters)
                print("Saved parameters:")
                print(saved_parameters)
                print('Some differences between parameters have been identified')
                # check if it only the filename:
                for k, v in self.parameters.items():
                    if k != "mol_file":
                        if v != saved_parameters[k]:
                            raise ValueError
                    else:
                        if v != saved_parameters[k]:
                            # check if basename is the same:
                            if os.path.basename(v) != os.path.basename(saved_parameters[k]):
                                raise ValueError
                            else:
                                print(f'Path differing, but assuming files named {os.path.basename(v)} are the same.')
                # print('\t-> Overwriting the given parameters with the stored ones')
                # self._parameters = saved_parameters
        return existing

    def download(self):
        pass

    def process(self):
        if self._num_processes > 1:
            Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

        print(f"--- Processing molecules with {self._num_processes} processes ---")
        mols = []
        coords = []
        lps = []
        if self._in_memory:
            blocks = mol2_blocks_from_mol2_file(self._mol_file) 
            for block in blocks:
               mol, conf_coords, lps_dic = mol_from_mol2_block(block, sanitize=self._sanitize, 
                                                      removeHs=self._remove_hs, simple_read=self._simple_read)
               mols.append(mol)
               coords.append(conf_coords)
               lps.append(lps_dic)
        else:
            # need to build an index and then we can read in parallel
            print("--- Building molecule index ---")
            start_mol, end_mol, nmol = count_and_tell(self._mol_file)
            print("--- Reading molecule blocks ---")
            if self._num_processes == 1:
                for imol in range(nmol):
                    mol, conf_coords, lps_dic = wrapper_mol_from_mol2_block(start_mol[imol], end_mol[imol], fname=self._mol_file,
                                                                            sanitize=self._sanitize, removeHs=self._remove_hs,
                                                                            simple_read=self._simple_read)
                    mols.append(mol)
                    coords.append(conf_coords)
                    lps.append(lps_dic)

            else:
                partial_wrapper_mol_from_mol2_block = partial(wrapper_mol_from_mol2_block, fname=self._mol_file,
                                                            sanitize=self._sanitize, removeHs=self._remove_hs, 
                                                            simple_read=self._simple_read)
                with torch.multiprocessing.Pool(processes=self._num_processes) as pool:
                    tmp = pool.starmap(partial_wrapper_mol_from_mol2_block, zip(start_mol, end_mol))
                mols, coords, lps = [list(x) for x in zip(*tmp)]
        
        if self._mol_selection: # if it is not None
            print("--- Subsetting molecule list ---")
            mols = [mols[i] for i in self._mol_selection]
            coords = [coords[i] for i in self._mol_selection]
            lps = [lps[i] for i in self._mol_selection]

        print("--- Removing invalid molecules ---")
        nmols = len(mols)
        valid_idxs = []
        for imol, mol in enumerate(mols):
            if mol is not None:
                valid_idxs.append(imol)
        ninvalid = nmols - len(valid_idxs)
        if self._verbose:
            print(f"Found {len(valid_idxs)} ({ninvalid}) valid (invalid) molecules out of {nmols} total selected molecules")
        valid_mols = [mols[i] for i in valid_idxs]
        if not self._ignore_lps:
            valid_lps = [lps[i] for i in valid_idxs]
        else:
            # dirty trick to exclude lps from the graph
            valid_lps = [None for i in valid_idxs]

        print("--- Creating graphs ---")
        tic = time.perf_counter()
        self.graphs = []
        self.smiles = []
        self.net_charges = []

        if self._num_processes == 1:
            # for imol, mol in enumerate(mols):
            for imol, mol in enumerate(valid_mols):
                g, ss, ncharge = mol_graph_and_features(mol, valid_lps[imol], bond_mode="covalent", 
                                               atom_featurizer=self._atom_featurizer, 
                                               bond_featurizer=self._bond_featurizer)
                self.graphs.append(g)
                self.smiles.append(ss)
                self.net_charges.append(ncharge)
        else: 
            torch.multiprocessing.set_sharing_strategy('file_system') # Suggested here: https://github.com/pytorch/pytorch/issues/11201
            partial_mol_graph_and_features = partial(mol_graph_and_features, bond_mode="covalent", 
                                               atom_featurizer=self._atom_featurizer, 
                                               bond_featurizer=self._bond_featurizer)
            with torch.multiprocessing.Pool(processes=self._num_processes) as pool:
                tmp = pool.starmap(partial_mol_graph_and_features, zip(valid_mols, valid_lps))
            self.graphs, self.smiles, self.net_charges = [list(x) for x in zip(*tmp)]

        self.net_charges = torch.tensor(self.net_charges, dtype=torch.float32)
        print(f"Graphs and featurization done in {time.perf_counter() - tic:.3}")
        self._process_time = time.perf_counter() - tic


    def save(self):
        dgl.save_graphs(self.graph_file, self.graphs, {'net_charge': self.net_charges})
        dgl.data.utils.save_info(self.info_file, {'smiles': self.smiles, 'parameters': self._parameters})

    def load(self):
        self.graphs, label_dict = dgl.load_graphs(self.graph_file)
        self.net_charges = label_dict['net_charge']
        info_dic = dgl.data.utils.load_info(self.info_file)
        self.smiles = info_dic['smiles']
        self._parameters = info_dic['parameters']
        self.mol_names = info_dic['mol_names']
    
    def get_graph_and_mol_name(self, idx):
        return self.graphs[idx], self.mol_names[idx]

    def __getitem__(self, idx):
        return self.graphs[idx], self.net_charges[idx]
    
    def __len__(self):
        return len(self.graphs)
    
    def __repr__(self):
        return f'Dataset("{self.name}", num_graphs={len(self.graphs)},\n' + \
               f'        raw_dir="{self.raw_dir}"\n' +  \
               f'        graph_file="{self.graph_file}")'
        
    @property
    def graph_file(self):
        '''File containing the DGLGraph objects'''
        return(os.path.join(self.save_dir, f'{self.name}_graph.bin'))

    @property
    def info_file(self):
        '''File containing the DGLGraph additional information and parameters'''
        return(self.graph_file.replace('.bin', '_parameters.pkl'))

    @property
    def parameters(self):
        return(self._parameters)

if __name__ == '__main__':
    from dgllife.utils import BaseBondFeaturizer
    from dgllife.utils import bond_type_one_hot

    proj_dir = Path(__file__).resolve().parents[2]
    data_dir = os.path.join(proj_dir, "data")

    # featurizers:
    atom_featurizer = ElementAtomFeaturizer(allowable_set=["C", "N", "O", "S", "F", "P", "Cl", "Br", "I", "H"],
                                            encode_unknown=False)
    bond_featurizer = BaseBondFeaturizer({'e': bond_type_one_hot})

    mol_dataset = MolDataset(name="small",
                             mol_file=os.path.join(data_dir, "raw/small.mol2"),
                             raw_dir=os.path.join(data_dir, "raw"),
                             save_dir=os.path.join(data_dir, "processed"),
                             atom_featurizer=atom_featurizer,
                             bond_featurizer=bond_featurizer,
                             num_processes=2,
                             in_memory=True,
                             force_reload=True,
                             simple_read=False,
                             ignore_lps=True,
                             verbose=True)

    print(mol_dataset)
    print(f"Number of molecules: {len(mol_dataset)}")
    print(mol_dataset[:2])
    g, charge = mol_dataset[0]
    print("Node data:")
    print(g.ndata)
    print("Edge data:")
    print(g.edata)
    print("Graph data:")
    print(charge)

