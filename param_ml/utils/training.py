import time
import torch 
import dgl

def compute_extra_metrics(pred, target, metrics_func):
    ''' Compute extra metrics:
    Parameters
    ----------
    pred : torch.Tensor
        Predicted values
    target : torch.Tensor
        Target values
    metrics : dict of callable
        Dictionary of metrics to compute    
    '''
    extra_metrics = {k: v(pred, target) for k,v in metrics_func.items()}
    return extra_metrics



def train_one_epoch(dataloader, model, loss_fn, optimizer, device):
    model.train()
    
    tic = time.perf_counter()
    loss_epoch = 0
    nnodes_epoch = 0
    
    true_labels = []
    pred_labels = []

    for data, _ in dataloader:
        batched_graph = data.to(device)
        labels = batched_graph.ndata['charges'] # note: already on device
        
        # TODO: consider converting to float32 directly in the dataset class
        pred = model(batched_graph, 
                     batched_graph.ndata['feats'].to(dtype=torch.float32), 
                     batched_graph.edata['e'].to(dtype=torch.float32))
        
        # print(pred.shape, labels.shape)
        loss = loss_fn(pred, labels.view(-1, 1)) # + PENALTY * model.regularization_loss()
        # model.regularization_loss() ONLY uses the weights of the model and auxiliary functions, not the predicted values

        # save labels:
        true_labels.extend(labels.tolist())
        pred_labels.extend(pred.flatten().tolist())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()
        nnodes_epoch += batched_graph.number_of_nodes()

    loss_epoch /= nnodes_epoch
    labels_out = [true_labels, pred_labels]
    toc = time.perf_counter()

    return loss_epoch, toc-tic, labels_out

def eval_one_epoch(dataloader, model, loss_fn, device):
    model.eval()
    
    tic = time.perf_counter()
    loss_epoch = 0
    nnodes_epoch = 0

    true_labels = []
    pred_labels = []
    
    with torch.no_grad():
        for data, _ in dataloader:
            batched_graph = data.to(device)
            labels = batched_graph.ndata['charges'] # note: already on device

            pred = model(batched_graph, 
                        batched_graph.ndata['feats'].to(dtype=torch.float32), 
                        batched_graph.edata['e'].to(dtype=torch.float32))
            loss = loss_fn(pred, labels.view(-1, 1))

            # save labels:
            true_labels.extend(labels.tolist())
            pred_labels.extend(pred.flatten().tolist())

            loss_epoch += loss.item()
            nnodes_epoch += batched_graph.number_of_nodes()
    loss_epoch /= nnodes_epoch
    labels_out = [true_labels, pred_labels]
    toc = time.perf_counter()

    return loss_epoch, toc-tic, labels_out
