# functions and utilities for checkpointing and logging
import os 
from datetime import datetime
import pickle
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

# TODO: probably would have been better to include a checkpoint instance into
# datapoint rather than use inheritance
def save_checkpoint(model, optimizer, epoch, train_loss, valid_loss, checkpoint_dir, 
                    remove_previous=False, name=None, log_dir=None):
    checkpoint = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'epoch': epoch,
                  'train_loss': train_loss,
                  'valid_loss': valid_loss,
                  'log_dir': log_dir} # logging directory
    # Set checkpoint name:
    if name is None:
            name = f"epoch-{epoch}_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    cptname = os.path.join(checkpoint_dir, 'cpt_' + name + '.tar')

    if remove_previous:
        for f in os.listdir(checkpoint_dir):
            if f.startswith('cpt_epoch-') or f == ('cpt_' + name + '.tar'):
                os.remove(os.path.join(checkpoint_dir, f))
    torch.save(checkpoint, cptname)

def load_checkpoint(filename, model, optimizer):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    train_loss = checkpoint['train_loss']
    valid_loss = checkpoint['valid_loss']
    log_dir = checkpoint['log_dir']
    return model, optimizer, epoch, train_loss, valid_loss, log_dir

def save_datapoint(epoch, train_data, valid_data, data_dir, 
                   remove_previous=False, name=None, log_dir=None):
    datapoint = {
        'epoch': epoch,
        'train_data': train_data,
        'valid_data': valid_data,
        'log_dir': log_dir
    }

    if name is None:
        name = f"epoch-{epoch}_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dataname = os.path.join(data_dir, 'data_' + name + '.pkl')
    
    if remove_previous:
        for f in os.listdir(data_dir):
            if f.startswith('data_epoch-') or f == ('data_' + name + '.pkl'):
                os.remove(os.path.join(data_dir, f))

    with open(dataname, 'wb') as handle:
        pickle.dump(datapoint, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_datapoint(filename):
    with open(filename, 'rb') as handle:
        datapoint = pickle.load(handle)
    return datapoint #datapoint['epoch'], datapoint['train_data'], datapoint['valid_data']
    
# Tensorboard logging functions
def tensorboard_setup(dataset_name, model, config=None):
    '''
    Setting up a tensorboard log directory and adding the main model 
    and run configurations to it.
    '''
    
    file_name = model.__class__.__name__ + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    project_dir = Path(__file__).resolve().parents[2]
    logdir = os.path.join(project_dir, 'runs', dataset_name, file_name)
    writer = SummaryWriter(logdir) #, flush_secs=120)

    print(f"Tensorboard logdir is {logdir}")
    
    # writer.add_text("Parameters", f"Ncores: {config['ncores']}")
    # writer.add_hparams({"batch": config['batch_size'],
    #                     "model": model.__class__.__name__})
    
    if config is not None:
        for k,v in config.items():
            writer.add_text("Parameters", f"{k}: {v}")
    writer.add_text("Parameters", f"Model: {model.__class__.__name__}")
    writer.add_text("Parameters", f"Dataset: {dataset_name}")
    writer.flush()

    return writer

def tensorboard_add_scalars_epoch(writer, epoch, train_loss, valid_loss, train_time, valid_time, 
                                  additional_metrics=None,
                                  profiler=None):
    '''Adding scalars to tensorboard'''
    writer.add_scalars(main_tag="Metrics/Loss",
                       tag_scalar_dict={'train': train_loss, 'valid': valid_loss},
                       global_step=epoch)
    writer.add_scalars(main_tag="Time",
                       tag_scalar_dict={'train': train_time, 'valid': valid_time},
                       global_step=epoch)
    if additional_metrics is not None:
        for met, val in additional_metrics.items():
            writer.add_scalars(main_tag="Metrics/" + met,
                               tag_scalar_dict={'train': val[0], 'valid': val[1]},
                               global_step=epoch)
    if profiler is not None:
        # Log the profiling information to TensorBoard
        writer.add_scalar('CPU time (ms)', profiler.self_cpu_time_total / 1000, epoch)
        writer.add_scalar('CUDA time (ms)', profiler.self_cuda_time_total / 1000, epoch)
        writer.add_scalar('CPU memory usage (bytes)', profiler.cpu_memory_usage, epoch)
        writer.add_scalar('CUDA memory usage (bytes)', profiler.cuda_memory_usage, epoch)

    writer.flush()
