"""Setup script for param_ml package. Use to train a model on a mol file."""
__author__ = "David Parker and Cassiano Langini"

import os
import pathlib
import argparse
import traceback

import numpy as np
import torch
import torch.nn.functional as F

import dgl
from dgl.data.utils import split_dataset
from dgl.dataloading import GraphDataLoader
from param_ml.models.mpnn import VanillaMPNN

from param_ml.utils.datasets import MolDataset
from param_ml.utils.featurizers import ElementAtomFeaturizer
from param_ml.utils.training import eval_one_epoch, train_one_epoch, compute_extra_metrics
# reporting:
from param_ml.utils.reporting import tensorboard_setup, tensorboard_add_scalars_epoch
from param_ml.utils.reporting import save_checkpoint, load_checkpoint, save_datapoint, load_datapoint

def get_cli_args():
    """Parse command line arguments"""
    try:
        parser = argparse.ArgumentParser(description="""Setup script for param_ml package. Use to train a model on a mol file.""")
        parser.add_argument("-i", "--input", help="Input dataset name to train on.", required=True)
        parser.add_argument("-n", "--mol_file", help="Input mol2 file", default=None)
        parser.add_argument("-d", "--rundir", help="Base directory (data locations are inferred from this).", default=None)
        parser.add_argument("-m", "--model", help="Model to train", choices=["vanilla_mpnn"], default="vanilla_mpnn")
        parser.add_argument("-ne", "--nepochs", type=int, default=3, help="Total number of epochs")
        parser.add_argument("-nm", "--n_mols", type=int, default=1000, help="Number of molecules to train on")
        parser.add_argument("-b", "--batch-size", type=int, default=4, help="Batch size")
        parser.add_argument("-s", '--seed', type=int, default=42, help="Base seed for random number generators")
        parser.add_argument("-r", "--restart", default=None, type=str, help="Checkpoint to restart training from")
        parser.add_argument("-nc", "--n_cpus", type=int, default=1, help="Number of cpus to use for data loading")
    except:
        print("An exception occurred with argument parsing. Check your provided options.")
        traceback.print_exc()
    args = parser.parse_args()
    # taking care of defaults
    if args.mol_file is None:
        args.mol_file = args.input+".mol2"
    if args.rundir is None:
        args.rundir = pathlib.Path(__file__).resolve().parents[1]
    args.raw = os.path.join(args.rundir, "data", "raw")
    args.processed = os.path.join(args.rundir, "data", "processed")

    return args

def mol_collate_fn(data):
    graphs, labels = zip(*data)
    batched_graph = dgl.batch(graphs)
    labels = torch.stack(labels, dim=0)
    return batched_graph, labels

def main():
    args = get_cli_args()
    print("Input arguments: ")
    print(args)
    # set up
    random_seed = args.seed
    batch_size = args.batch_size
    n_epochs = args.nepochs
    best_valid_loss = np.inf
    checkpoint_dir = os.path.join(args.rundir, "checkpoints", args.input)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    # Set device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    run_on_gpu = True if device == 'cuda:0' else False
    if run_on_gpu:
        print("Running on GPU")
    else:
        print("Running on CPU")

    # load dataset:
    atom_featurizer = ElementAtomFeaturizer(allowable_set=["C", "N", "O", "S", "F", "P", "Cl", "Br", "I", "H"],
                                            encode_unknown=False)
    
    mol_dataset = MolDataset(name=args.input,
                             mol_file=os.path.join(args.raw, args.mol_file),
                             raw_dir=args.raw,
                             save_dir=args.processed,
                             atom_featurizer=atom_featurizer,
                             mol_selection = list(range(args.n_mols)), 
                             in_memory=False,
                             num_processes = args.n_cpus,
                             sanitize=True,
                             remove_hs=False,
                             bond_featurizer=None,
                             force_reload=True,
                             simple_read=False,
                             verbose=False)
    print("Loaded dataset: ")
    print(mol_dataset)
    print(f"Number of molecules: {len(mol_dataset)}")

    with open(os.path.join(args.rundir, "benchmarks", args.input+"_dataset_process_time.csv"), "w") as f:
        # Format: dataset_name, process_time, n_mols, n_cpus
        f.write("dataset_name,process_time,n_mols,n_cpus\n")
        f.write(f"{args.input}, {mol_dataset._process_time}, {len(mol_dataset)}, {mol_dataset._num_processes}\n")


    # TODO dont need test set
    train, valid = split_dataset(mol_dataset, frac_list=[0.9, 0.1], shuffle=True, 
                                       random_state=random_seed)
    
    train_dataloader = GraphDataLoader(train,
                                       #sampler=train.indices,
                                       batch_size=batch_size,
                                       drop_last=False, #drop last incomplete batch
                                       shuffle=True, #randomly shuffle indices at each epoch,
                                        collate_fn=mol_collate_fn
    )
    valid_dataloader = GraphDataLoader(valid,
                                       batch_size=batch_size,
                                       drop_last=False,
                                       shuffle=False,
                                       collate_fn=mol_collate_fn
    )

    # load model
    if args.model == "vanilla_mpnn":
        model = VanillaMPNN(node_in_feats=mol_dataset[0][0].ndata['feats'].shape[1],
                                       edge_in_feats=mol_dataset[0][0].edata['e'].shape[1],
                                       node_hidden_feats=[64, 128],
                                       edge_hidden_feats=[32, 64, 128],
                                       ntasks=1,
                                       do_batchnorm=True,
                                       readout_type='node',
                                       bias=True
        ).to(device)
    else:
        raise NotImplementedError(f"Model {args.model} not implemented.")

    loss_fn = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # extra metrics
    extra_metrics = {'mse': F.mse_loss,
                     'mae': F.l1_loss}
    # Create a SummaryWriter object to write logs to TensorBoard
    config = {'batch_size': batch_size}
    writer = tensorboard_setup(dataset_name=args.input,
                               model=model, config=config)
    
    train_times = []
    train_losses = []
    valid_times = []
    valid_losses = []
    # valid_rmses = []

    # Training loop
    for epoch in range(n_epochs):
        loss_epoch, time_epoch, train_labels = train_one_epoch(train_dataloader, model, loss_fn, optimizer, device)
        train_losses.append(loss_epoch)
        train_times.append(time_epoch)

        valid_loss, valid_time, valid_labels = eval_one_epoch(valid_dataloader, model, loss_fn, device)
        valid_losses.append(valid_loss)
        valid_times.append(valid_time)

        # calculate extra metrics:
        train_metrics = compute_extra_metrics(torch.tensor(train_labels[1]), 
                                              torch.tensor(train_labels[0]),
                                              extra_metrics)
        valid_metrics = compute_extra_metrics(torch.tensor(valid_labels[1]),
                                              torch.tensor(valid_labels[0]),
                                              extra_metrics)
        all_metrics = {
            'mse': (train_metrics['mse'], valid_metrics['mse']), # dummy, should be equal to loss
            'rmse': (torch.sqrt(train_metrics['mse']), torch.sqrt(valid_metrics['mse'])),
            'mae': (train_metrics['mae'], valid_metrics['mae']),
        }
        # Log to tensorboard
        tensorboard_add_scalars_epoch(writer=writer, epoch=epoch, 
                                  train_loss=loss_epoch, valid_loss=valid_loss,
                                  train_time=time_epoch, valid_time=valid_time,
                                  additional_metrics=all_metrics)
        # Save checkpoint
        if valid_loss < best_valid_loss or epoch % 5 == 0:
            if valid_loss < best_valid_loss:
                # best checkpoint
                print("Best validation loss so far!")
                best_valid_loss = valid_loss
                save_checkpoint(model, optimizer, epoch, loss_epoch, valid_loss,
                                checkpoint_dir, remove_previous=True, name=f"best_{args.model}",
                                log_dir=writer.get_logdir())
                # also save datapoint:
                save_datapoint(epoch, train_labels, valid_labels, data_dir=checkpoint_dir,
                               remove_previous=True, name=f"best_{args.model}",
                               log_dir=writer.get_logdir())
            else:
                # periodic checkpoint
                save_checkpoint(model, optimizer, epoch, loss_epoch, valid_loss,
                                checkpoint_dir, remove_previous=True, 
                                log_dir=writer.get_logdir())


        print(f'Epoch {epoch+1:d}/{n_epochs:d}:') 
        print(f"Epoch training time: {time_epoch:.5} seconds")
        print(f'Epoch average loss: {loss_epoch:.4f}')
        print(f"Validation time: {valid_time:.5} seconds")
        print(f'Validation average loss: {valid_loss:.4f}')

    writer.close()


if __name__ == '__main__':
    main()