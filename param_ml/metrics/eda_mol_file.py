"""Script to get some statistics on the molecules in a mol file."""
__author__ = "David Parker"

import argparse
import traceback
from rdkit.Chem.rdmolfiles import MolFromMol2Block
from rdkit.Chem import Descriptors
from rdkit.Chem import GetPeriodicTable
import pandas as pd
import matplotlib.pyplot as plt


def get_cli_args():
    """Parse command line arguments"""
    try:
        parser = argparse.ArgumentParser(description="""Script to get some statistics on the molecules in a mol file.""")
        parser.add_argument("-i", "--input", help="Input mol2 file")
    except:
        print("An exception occurred with argument parsing. Check your provided options.")
        traceback.print_exc()
    return parser.parse_args()


def get_mols_from_mol2_file(mol2_file):
    """Get molecules from a mol2 file"""
    mols = []
    with open(mol2_file, 'r') as f:
        mol2_block = ''
        for line in f:
            if line.startswith('@<TRIPOS>MOLECULE'):
                if mol2_block:
                    mol = MolFromMol2Block(mol2_block, sanitize=True, removeHs=False)
                    if mol:
                        mols.append(mol)
                mol2_block = ''
            mol2_block += line
    return mols


class MolStats:
    """Class to hold statistics on a molecule"""
    def __init__(self, mol):
        self.mol = mol
        self.num_atoms = mol.GetNumAtoms()
        self.mol_weight = Descriptors.MolWt(mol)
        self.num_heavy_atoms = Descriptors.HeavyAtomCount(mol)
        self.num_h_acceptors = Descriptors.NumHAcceptors(mol)
        self.num_h_donors = Descriptors.NumHDonors(mol)
        self.num_rotatable_bonds = Descriptors.NumRotatableBonds(mol)

        self.atom_features = self.get_atomic_features()
    
    def get_atomic_features(self):
        """Get the following features for each atom in the molecule:
        - atomic number
        - electronegativity
        - atomic size
        - Degree: The degree of an atom is defined to be its number of directly-bonded neighbors. The degree is independent of bond orders, but is dependent on whether or not Hs are explicit in the graph.
        - hybridization
        - aromatic nature
        - chiral
        - axial
        """
        # Dictoionary to hold the features for each atom
        atom_features = {}
        for atom in self.mol.GetAtoms():
            atom_features[atom.GetIdx()] = {
                'atomic_symbol': GetPeriodicTable().GetElementSymbol(atom.GetAtomicNum()),
                # 'electronegativity': atom.GetProp('EN'),
                'atomic_size': GetPeriodicTable().GetRvdw(atom.GetAtomicNum()),
                'degree': atom.GetDegree(),
                'total_valence': atom.GetTotalValence(),
                'num_h': atom.GetTotalNumHs(),
                'hybridization': atom.GetHybridization(),
                'aromatic': atom.GetIsAromatic(),
                'chiral': atom.GetChiralTag(),
                'owning_mol': atom.GetOwningMol().GetProp('_Name')
                # 'axial': atom.GetProp('AX'),

            }
        return atom_features
    
    def __str__(self):
        return f"""Molecule with {self.num_atoms} atoms"""


class MolStatsPlotter:
    """Class to plot the statistics of a list of molecules, of the the MolStats class"""
    def __init__(self, mol_stats: list):
        self.mol_stats = mol_stats
        self.mol_df= self.get_mol_df()
        self.atom_df = self.get_atom_df()
        self.num_mols = len(mol_stats)
        self.num_atoms = sum([mol_stat.num_atoms for mol_stat in mol_stats])

    def get_mol_df(self):
        mol_features = []
        for mol_stat in self.mol_stats:
            mol_features.append({
                'num_atoms': mol_stat.num_atoms,
                'mol_weight': mol_stat.mol_weight,
                'num_heavy_atoms': mol_stat.num_heavy_atoms,
                'num_h_acceptors': mol_stat.num_h_acceptors,
                'num_h_donors': mol_stat.num_h_donors,
                'num_rotatable_bonds': mol_stat.num_rotatable_bonds
            })
        return pd.DataFrame(mol_features)
    
    def get_atom_df(self):
        atom_features = []
        for mol_stat in self.mol_stats:
            for atom_idx, atom_feature in mol_stat.atom_features.items():
                # atom_feature['atom_idx'] = atom_idx
                atom_features.append(atom_feature)
        atom_df = pd.DataFrame(atom_features)
        return atom_df
    
    def get_mol_stats_plot(self):
        """Plot the histograms of each molecule feature"""
        fig, axs = plt.subplots(3, 2)

        self.mol_df['num_atoms'].hist(
            ax=axs[0, 0],
            log=True
            )
        axs[0, 0].set_title('Number of Atoms')

        self.mol_df['mol_weight'].hist(
            ax=axs[0, 1],
            log=True
        )
        axs[0, 1].set_title('Molecular Weight')

        self.mol_df['num_heavy_atoms'].hist(
            ax=axs[1, 0],
            log=True
        )
        axs[1, 0].set_title('Number of Heavy Atoms')

        self.mol_df['num_h_acceptors'].value_counts().plot(
            kind='bar',
            ax=axs[1, 1],
            log=True,
            grid=True,
            rot=0,
            xticks=sorted(self.mol_df['num_h_acceptors'].unique())[::5]
        )
        axs[1, 1].set_title('Number of H-Bond Acceptors')

        self.mol_df['num_h_donors'].value_counts().plot(
            kind='bar',
            ax=axs[2, 0],
            log=True,
            grid=True,
            rot=0,
            xticks=sorted(self.mol_df['num_h_donors'].unique())[::5]
        )
        axs[2, 0].set_title('Number of H-Bond Donors')

        self.mol_df['num_rotatable_bonds'].value_counts().plot(
            kind='bar',
            ax=axs[2, 1],
            log=True,
            grid=True,
            rot=0,
            xticks=sorted(self.mol_df['num_rotatable_bonds'].unique())[::5]
        )
        axs[2, 1].set_title('Number of Rotatable Bonds')

        plt.subplots_adjust(hspace=0.5, wspace=0.5)
        fig.suptitle(f'Molecule Statistics for {self.num_mols} Molecules')

        return fig
    
    def get_atom_stats_plot(self):
        """Plot the histograms of each atom feature"""
        fig, axs = plt.subplots(4, 2)
        
        self.atom_df['atomic_size'].hist(
            ax=axs[0, 0],
            log=True
            )
        axs[0, 0].set_title('Atomic Size')
        
        self.atom_df['degree'].hist(
            ax=axs[0, 1],
            log=True
        )
        axs[0, 1].set_title('Degree')
        
        self.atom_df['total_valence'].hist(
            ax=axs[1, 0],
            log=True
        )
        axs[1, 0].set_title('Total Valence')
        
        self.atom_df['num_h'].value_counts().plot(
            ax=axs[1, 1],
            kind='bar',
            grid=True,
            logy=True,
            ylim=(0, self.num_atoms),
            rot=0
        )
        axs[1, 1].set_title('Num H')
        
        self.atom_df['hybridization'].hist(
            ax=axs[2, 0],
            log=True
        )
        axs[2, 0].set_title('Hybridization')
        
        self.atom_df['chiral'].value_counts().plot(
            ax=axs[2, 1],
            kind='bar',
            grid=True,
            logy=True,
            rot=0
        )
        axs[2, 1].set_title('Chiral')
        
        self.atom_df['atomic_symbol'].value_counts().plot(
            ax=axs[3, 0],
            kind='bar',
            logy=True,
            grid=True,
            rot=0
        )
        axs[3, 0].set_title('Atomic Symbol')

        self.atom_df['aromatic'].value_counts().plot(
            ax=axs[3, 1],
            kind='bar',
            grid=True,
            logy=True,
            ylim=(0, self.num_atoms),
            rot=0
        )
        axs[3, 1].set_title('Aromatic')

        plt.subplots_adjust(hspace=0.5, wspace=0.5)
        fig.suptitle(f'Atom Statistics for {self.num_mols} molecules with {self.num_atoms} atoms total')

        return fig

    def show_plots(self):
        mol_stats_plot = self.get_mol_stats_plot()
        atom_stats_plot = self.get_atom_stats_plot()
        plt.show()

def main():
    args = get_cli_args()
    mols = get_mols_from_mol2_file(args.input)

    mol_stats = [MolStats(mol) for mol in mols]
    mol_stats_plotter = MolStatsPlotter(mol_stats)

    mol_stats_plotter.show_plots()



if __name__ == '__main__':
    main()