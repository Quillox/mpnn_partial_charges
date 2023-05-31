"""Script to evaluate the performance of a model on a dataset. """
__author__ = "David Parker"

import argparse
import traceback
import os
from dgl.dataloading import GraphDataLoader
import torch
import pandas as pd
import matplotlib.pyplot as plt


from param_ml.models.mpnn import VanillaMPNN
from param_ml.utils.datasets import MolDataset
from param_ml.utils.featurizers import ElementAtomFeaturizer

# This must be the same as the one used to train the model
ALLOWABLE_SET = ["C", "N", "O", "S", "F", "P", "Cl", "Br", "I", "H"]


def get_cli_args():
    """Parse command line arguments"""
    try:
        parser = argparse.ArgumentParser(description="""Script to evaluate the performance of a model on a dataset.""")
        parser.add_argument("-c", "--checkpoint", help="Path to the checkpoint of the model to evaluate.", required=True)
        parser.add_argument("-d", "--data", help="Path to the directory with the data to evaluate the model on.", required=True)
        parser.add_argument("-n", "--molfile", help="Name of the molfile to use.", required=True)
        parser.add_argument("-nm", "--n_mols", help="Number of molecules to use from the dataset.", default=100, type=int)
    except:
        print("An exception occurred with argument parsing. Check your provided options.")
        traceback.print_exc()
    return parser.parse_args()


def get_dataset(data_dir, molfile_name, n_mols):
    """Get the dataset to evaluate the model on."""

    raw_dir = os.path.join(data_dir, "raw")
    processed_dir = os.path.join(data_dir, "processed")
    molfile = os.path.join(raw_dir, molfile_name)

    atom_featurizer = ElementAtomFeaturizer(allowable_set=ALLOWABLE_SET,
                                            encode_unknown=False)
    
    mol_dataset = MolDataset(name="dataset",
                             mol_file=molfile,
                             raw_dir=raw_dir,
                             save_dir=processed_dir,
                             atom_featurizer=atom_featurizer,
                             mol_selection = list(range(n_mols)), 
                             in_memory=False,
                            #  num_processes = os.cpu_count(),
                             num_processes=1,
                             sanitize=True,
                             remove_hs=False,
                             bond_featurizer=None,
                             force_reload=True,
                             simple_read=False,
                             verbose=False)
    
    return mol_dataset


def get_model(checkpoint_path, mol_dataset, device):
    """Get the model to evaluate."""
    model = VanillaMPNN(node_in_feats=mol_dataset[0][0].ndata['feats'].shape[1],
                                edge_in_feats=mol_dataset[0][0].edata['e'].shape[1],
                                node_hidden_feats=[64, 128],
                                edge_hidden_feats=[32, 64, 128],
                                ntasks=1,
                                do_batchnorm=True,
                                readout_type='node',
                                bias=True
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def evaluate_model(model, dataloader, device):
    """Evaluate the model on a dataset."""
    model.eval()
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for data, _ in dataloader:
            batched_graph = data.to(device)
            labels = batched_graph.ndata['charges'] # note: already on device

            pred = model(batched_graph, 
                        batched_graph.ndata['feats'].to(dtype=torch.float32), 
                        batched_graph.edata['e'].to(dtype=torch.float32))
            true_labels.extend(labels.tolist())
            pred_labels.extend(pred.flatten().tolist())
    
    return true_labels, pred_labels


def one_hot_to_element(one_hot, allowable_set):
    """Convert a one-hot encoded vector to the corresponding element name."""
    allowable_set=ALLOWABLE_SET
    idx = one_hot.index(1)
    if idx == len(allowable_set):
        element = "X"
    else:
        element = allowable_set[idx]
    return element


def make_node_mapping(dataset, true_labels, pred_labels):
    """Make a mapping to be able to index the evaluation labels with the node id."""
    node_map = {}
    i = 0
    graph_id = 0
    for graph, label in dataset:
        for node_id in graph.nodes().tolist():
            node_map[i] = {'feats':graph.ndata['feats'][node_id], 'charge':float(graph.ndata['charges'][node_id])}
            node_map[i]['graph_id'] = graph_id
            i += 1
        graph_id += 1
    for node in range(len(true_labels)):
        assert(node_map[node]["charge"] == true_labels[node])
        node_map[node]["pred"] = pred_labels[node]

    # decode the one-hot encoded element features
    for node in node_map:
        node_map[node]["element"] = one_hot_to_element(node_map[node]["feats"].tolist(), ALLOWABLE_SET)

    return node_map


def make_df_from_node_mapping(node_map):
    """Make a dataframe from the node mapping."""
    df = pd.DataFrame.from_dict(node_map, orient='index')
    df = df.drop(columns=['feats'])
    df = df.rename(columns={'charge':'true', 'pred':'pred', 'element':'element'})
    df['abs_error'] = abs(df['true'] - df['pred'])
    return df

def main():
    args = get_cli_args()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    run_on_gpu = True if device == 'cuda:0' else False

    mol_dataset = get_dataset(args.data, args.molfile, args.n_mols)
    model = get_model(args.checkpoint, mol_dataset, device)

    dataloader = GraphDataLoader(
        mol_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False
    )

    true_labels, pred_labels = evaluate_model(model, dataloader, device)

    node_map = make_node_mapping(mol_dataset, true_labels, pred_labels)
    df = make_df_from_node_mapping(node_map)

    print(df.groupby('element')['abs_error'].describe())
    df.boxplot(column='abs_error', by='element', figsize=(12,8))
    num_atoms = len(df)
    plt.title(f"Absolute error of predicted partial charges for {num_atoms} atoms")
    # Add information about the model
    if "\\" in args.checkpoint:
        model_name = args.checkpoint.split("\\")[-2].split(".")[0]
    elif "/" in args.checkpoint:
        model_name = args.checkpoint.split("/")[-2].split(".")[0]
    else:
        model_name = args.checkpoint.split(".")[0]
    plt.suptitle(f"Model: {model_name}")
    plt.ylabel("absolute error")
    plt.show()


if __name__ == '__main__':
    main()