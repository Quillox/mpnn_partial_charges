import os
import copy
import numpy as np
import torch 
import dgl

from dgllife.utils import BaseBondFeaturizer, bond_type_one_hot

from param_ml.utils.featurizers import ElementAtomFeaturizer
import rdkit
from rdkit import Chem


tripos_keys = [ "@<TRIPOS>ALT_TYPE",
                "@<TRIPOS>ANCHOR_ATOM",
                "@<TRIPOS>ASSOCIATED_ANNOTATION",
                "@<TRIPOS>ATOM",
                "@<TRIPOS>BOND",
                "@<TRIPOS>CENTER_OF_MASS",
                "@<TRIPOS>CENTROID",
                "@<TRIPOS>COMMENT",
                "@<TRIPOS>CRYSIN",
                "@<TRIPOS>DICT",
                "@<TRIPOS>DATA_FILE",
                "@<TRIPOS>EXTENSION_POINT",
                "@<TRIPOS>FF_PBC",
                "@<TRIPOS>FFCON_ANGLE",
                "@<TRIPOS>FFCON_DIST",
                "@<TRIPOS>FFCON_MULTI",
                "@<TRIPOS>FFCON_RANGE",
                "@<TRIPOS>FFCON_TORSION",
                "@<TRIPOS>LINE",
                "@<TRIPOS>LSPLANE",
                "@<TRIPOS>MOLECULE",
                "@<TRIPOS>NORMAL",
                "@<TRIPOS>QSAR_ALIGN_RULE",
                "@<TRIPOS>RING_CLOSURE",
                "@<TRIPOS>ROTATABLE_BOND",
                "@<TRIPOS>SEARCH_DIST",
                "@<TRIPOS>SEARCH_OPTIONS",
                "@<TRIPOS>SET",
                "@<TRIPOS>SUBSTRUCTURE",
                "@<TRIPOS>U_FEAT",
                "@<TRIPOS>UNITY_ATOM_ATTR",
                "@<TRIPOS>UNITY_BOND_ATTR"]

def safe_cast(value, target_type, default=None):
    try:
        return target_type(value)
    except (ValueError, TypeError):
        return default

def mol2_blocks_from_mol2_file(mol2_fn):
    '''Reads mol2 file and split it in molecule blocks
    '''
    with open(mol2_fn, "r") as f:
        mol2 = f.read()
    blocks = mol2.split("@<TRIPOS>MOLECULE")[1:]
    blocks = ["@<TRIPOS>MOLECULE" + x for x in blocks if (x != "")]
    return blocks

def mol2_block_from_file(start_p, end_p, fname):
    '''Returns block between two pointers
    '''
    f_handle = open(fname, 'r')
    f_handle.seek(start_p)
    block = f_handle.read(end_p - start_p)
    f_handle.close()
    return block

def wrapper_mol_from_mol2_block(start_p, end_p, fname, **kwargs):
    block = mol2_block_from_file(start_p, end_p, fname)
    mol, conf_coords, lps_dic = mol_from_mol2_block(block, **kwargs)
    return mol, conf_coords, lps_dic

def count_and_tell(in_fn):
    """Counts the molecules and tell the mol start and end positions in the file.
    Taken from SEED script shuffle_library_withDictionaries.py
    """
    start_mol = [] # Note that this is faster than start_mol = list()
    nmol = 0
    pattern = "@<TRIPOS>MOLECULE"

    f = open(in_fn, 'r')
    line = f.readline()
    prev_ptr = 0
    curr_ptr = f.tell()

    while line:
        if line.strip() == pattern:
            nmol += 1
            start_mol.append(prev_ptr)
        line = f.readline()
        prev_ptr = curr_ptr
        curr_ptr = f.tell()
    f.close()
    # print(nmol)

    end_mol = start_mol[1:]
    end_mol.append(os.path.getsize(in_fn))
    return start_mol, end_mol, nmol


def mol_from_mol2_block(block, sanitize=True, removeHs=False, simple_read=True):
    '''Create an RDKit molecule from a mol2 block.
    This function is needed for reading the alternative type information and dealing with LPs
    '''
    lps = {}
    if simple_read:
        mol = Chem.MolFromMol2Block(block, sanitize, removeHs)
        if mol is not None:    
            conf = mol.GetConformer()
            conf_coords = conf.GetPositions()
        else:
            print("Molecule is None!")
            conf_coords = None
        return mol, conf_coords, None
    else:
        # Extended read mode: to deal with LP particles and extra atom type information
        tripos_sections = read_tripos_sections(block)
        mol_name = tripos_sections["@<TRIPOS>MOLECULE"][0].strip()
        # print(tripos_sections)

        # Checking atoms for LPs:
        lp_idxs = [] # ATOM index of the LP
        lp_coords = []
        lp_charges = []
        # lp_atom_lines = [] # ATOM line for the LP
        lp_x = [] # main atom to which LP is connected
        new_atom_section = []
        new_bond_section = []
        for ll, line in enumerate(tripos_sections["@<TRIPOS>ATOM"]):
            ls = line.strip().split()
            if ls[1].startswith("LP"):
                # print("lone pair!")
                lp_idxs.append(int(ls[0]))
                lp_coords.append(np.array([float(x) for x in ls[2:5]], dtype=np.float32))
                lp_charges.append(float(ls[8]))
                # lp_atom_lines.append(line)
            else:
                new_atom_section.append(line)
        
        for ll, line in enumerate(tripos_sections["@<TRIPOS>BOND"]):
            ls = [int(x) for x in line.split()[:3]]
            if ls[1] in lp_idxs:
                lp_x.append(ls[2])
            elif ls[2] in lp_idxs:
                lp_x.append(ls[1])
            else:
                new_bond_section.append(line)
        
        nlps = len(lp_idxs)
        if nlps > 0:
            # if there are any lone pair particles:
            lps["index"] = lp_idxs
            lps["coords"] = lp_coords
            lps["charge"] =  lp_charges 
            lps["donor"] = lp_x 

            # new_molecule_section:
            new_molecule_section = copy.deepcopy(tripos_sections["@<TRIPOS>MOLECULE"])
            ls = [int(x) for x in new_molecule_section[1].strip().split()]
            ls[0] -= nlps 
            ls[1] -= nlps 
            new_molecule_section[1] = ' '.join([str(x) for x in ls])

            new_block = "\n".join(["@<TRIPOS>MOLECULE"] + new_molecule_section +
                                ["@<TRIPOS>ATOM"] + new_atom_section + 
                                ["@<TRIPOS>BOND"] + new_bond_section) + "\n" # very important to add final "\n"
        else:
            # no lone pair particles:
            lps = None 
            new_block = block
        
        # Now we can create the RDKit molecule from the block:
        mol = Chem.MolFromMol2Block(new_block, sanitize, removeHs)
        if mol is not None:
            conf = mol.GetConformer()
            conf_coords = conf.GetPositions()
        else:
            print(f"Molecule {mol_name} is None!")
            conf_coords = None
            lps = None

        return mol, conf_coords, lps



def mol_graph_and_features(mol, lps=None, bond_mode="covalent", atom_featurizer=None, 
                           bond_featurizer=None):
    '''Construct graph from RDKit molecule
    '''
    # print(mol)
    nnodes = mol.GetNumAtoms()
    nedges = mol.GetNumBonds()
    
    if lps:
        nlps = len(lps["index"])
    else:
        nlps = 0

    src, dst, charges = ([] for _ in range(3))
    bond_types = []

    if bond_mode == "covalent":
        for bond in mol.GetBonds():
            src.append(bond.GetBeginAtomIdx())
            dst.append(bond.GetEndAtomIdx())
        if lps:
            for ilp in range(nlps):
                src.append(lps["index"][ilp]-1) # -1 because of 0-indexing
                dst.append(lps["donor"][ilp]-1) # -1 because of 0-indexing
    else:
        raise NotImplementedError("Only covalent bonds are currently implemented.")
    
    # charges = list(map(lambda x: float(x.GetProp("_TriposPartialCharge")) , mol.GetAtoms()))
    charges = [float(x.GetProp("_TriposPartialCharge")) for x in mol.GetAtoms()]
    if lps:
        for lpc in lps["charge"]:
            charges.append(float(lpc))
    
    graph = dgl.graph((torch.tensor(src), torch.tensor(dst)), num_nodes=nnodes+nlps)

    ### Featurization ###
    if atom_featurizer is None:
        atom_featurizer = ElementAtomFeaturizer()
    if bond_featurizer is None:
        bond_featurizer = BaseBondFeaturizer({'e': bond_type_one_hot})
    
    # atom features:
    feats = atom_featurizer(mol)['h'].type(torch.int32)
    # graph.ndata.update({'feats': atom_featurizer(mol)['h'].type(torch.int32)}) # EQUIVALENT
    # Extend featurization with dummy one-hot 1-hot for the LP:
    if lps:
        lps_feats = torch.zeros((nlps, feats.size(dim=1)), dtype=torch.int32)
        feats = torch.vstack([feats, lps_feats]) 
    
    is_lp = torch.tensor([0] * nnodes + [1] * nlps, dtype=torch.int32)
    feats = torch.hstack([feats, is_lp.reshape(-1,1)])

    # bond features:
    efeats = bond_featurizer(mol)['e'].type(torch.int32)
    efeats = efeats[::2,:]
    if lps:
        lps_efeats = torch.zeros((nlps, efeats.size(dim=1)), dtype=torch.int32)
        efeats = torch.vstack([efeats, lps_efeats])
    is_lpedge = torch.tensor([0] * nedges + [1] * nlps, dtype=torch.int32)
    efeats = torch.hstack([efeats, is_lpedge.reshape(-1,1)])

    graph.edata.update({'e': efeats})
    graph.ndata['feats'] = feats
    graph.ndata['charges'] = torch.tensor(charges, dtype=torch.float32)
    net_charge = torch.sum(graph.ndata['charges'])

    # Make undirected graph (and copy features!):
    graph = dgl.add_reverse_edges(graph, copy_ndata=True, copy_edata=True) 

    smiles_string = Chem.MolToSmiles(mol)
    mol_name = mol.GetProp("_Name")
    
    # print(lps)
    # print(graph.ndata)
    # Final checks:
    if graph.edata['e'].size(dim=0) != (nedges+nlps)*2:
        raise ValueError('Wrong number of edge features!')
    if graph.ndata['feats'].size(dim=0) != nnodes+nlps:
        raise ValueError('Wrong number of node features!')
    if graph.ndata['charges'].size(dim=0) != nnodes+nlps:
        raise ValueError('Wrong number of charges!')
    return graph, smiles_string, net_charge

def read_tripos_sections(mol2_string):
    """Read the tripos sections.

    Parameters
    ----------
    mol2_string : str
        The mol2 file contents.

    Returns
    -------
    dict[str, list]
        The mol2 file contents separated into sections.
    """

    tripos_sections = dict.fromkeys(tripos_keys, [])
    lines = mol2_string.strip().split("\n")
    lines = [x.strip() for x in lines]
    section_indexes = [
        i for i, x in enumerate(lines) if x in tripos_keys]
    sections = [lines[i:j]
                for i, j in zip([0]+section_indexes, section_indexes+[None])]
    sections = [x for x in sections if x]
    for section in sections:
        tripos_sections[section[0]] = section[1:]

    if ("@<TRIPOS>ALT_TYPE" in tripos_sections):
        nats = len(tripos_sections["@<TRIPOS>ATOM"])
        for k in range(1,len(tripos_sections["@<TRIPOS>ALT_TYPE"]),2):
            tokened = tripos_sections["@<TRIPOS>ALT_TYPE"][k].split()
            if (len(tokened) >= 3):
                effstr = tokened[0]
                keepit = False
                for i in range(0,len(tokened)):
                    if (i%2 == 1):
                        atix = safe_cast(tokened[i],int,0)
                        if ((atix >= 1) and (atix <= nats)):
                            keepit = True
                            effstr += " "+tokened[i]
                        else:
                            keepit = False
                    else:
                        if keepit:
                            effstr += " "+tokened[i]
                if (len(effstr) > len(tokened[0])):
                    tripos_sections["@<TRIPOS>ALT_TYPE"][k] = effstr

    return tripos_sections

if __name__ == "__main__":
    pass
