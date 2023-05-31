import torch
from torch import nn
import torch.nn.functional as F

from param_ml.models.layers import LinearBn
from dgl.nn.pytorch import NNConv, Set2Set


class VanillaMPNN(nn.Module):
    """"More or less integrating
    MPNNGNN and MPNNPredictor class from dgllifescience
    Also compare to: https://www.kaggle.com/c/champs-scalar-coupling/discussion/93972
    """
    def __init__(self, 
                 node_in_feats=32, 
                 edge_in_feats=16, 
                 node_hidden_feats=[64,128],
                 edge_hidden_feats=[32, 64, 128],
                 ntasks=1, 
                 num_step_message_passing=6,
                 do_batchnorm=True,
                 readout_type='node', # whether have graph- or node-based readout
                 bias=True):

        super(VanillaMPNN, self).__init__()

        self._num_step_message_passing = num_step_message_passing
        self._num_step_set2set = 6
        self._num_layer_set2set = 3
        self._readout_type = readout_type

        # extend list to include input/output layer sizes
        if isinstance(node_hidden_feats, int):
            node_hidden_feats = [node_in_feats, node_hidden_feats]
        else:
            node_hidden_feats = [node_in_feats] + node_hidden_feats
        if isinstance(edge_hidden_feats, int):
            edge_hidden_feats = [edge_in_feats, edge_hidden_feats, node_hidden_feats[-1]*node_hidden_feats[-1]]
        else:
            edge_hidden_feats = [edge_in_feats] + edge_hidden_feats + [node_hidden_feats[-1]*node_hidden_feats[-1]]

        n_node_layers = len(node_hidden_feats)-1
        self.preprocess = nn.ModuleList()
        for i in range(n_node_layers):
            self.preprocess.append(
                LinearBn(node_hidden_feats[i], node_hidden_feats[i+1], bias=bias, do_batchnorm=do_batchnorm, activation=None)
            )
            self.preprocess.append(nn.ReLU())
        self.preprocess = nn.Sequential(*self.preprocess) # create a sequential model from the list
        
        n_edge_layers = len(edge_hidden_feats)-1
        self.edge_function = nn.ModuleList()
        for i in range(n_edge_layers):
            self.edge_function.append(
                LinearBn(edge_hidden_feats[i], edge_hidden_feats[i+1], bias=bias, do_batchnorm=do_batchnorm, activation=None)
            )
            if i < n_edge_layers - 1:
                self.edge_function.append(nn.ReLU())
        self.edge_function = nn.Sequential(*self.edge_function) # create a sequential model from the list

        self.gnn_layer = NNConv(
            in_feats=node_hidden_feats[-1],
            out_feats=node_hidden_feats[-1],
            edge_func=self.edge_function,
            aggregator_type='mean',
            bias=True
        )

        self.gru = nn.GRU(node_hidden_feats[-1], node_hidden_feats[-1])

        if self._readout_type == 'graph':
            self.graph_readout = Set2Set(input_dim=node_hidden_feats[-1],
                                n_iters=self._num_step_set2set,
                                n_layers=self._num_layer_set2set
                                )
            self.predict = nn.Sequential(
                LinearBn(2 * node_hidden_feats[-1], node_hidden_feats[-1], 
                        bias=bias, do_batchnorm=do_batchnorm, activation=None), # 2 * -> see set2set definition
                nn.ReLU(),
                nn.Linear(node_hidden_feats[-1], ntasks)
            )
        elif self._readout_type == 'node':
            self.predict = nn.Sequential(
                LinearBn(node_hidden_feats[-1], node_hidden_feats[-1], 
                        bias=bias, do_batchnorm=do_batchnorm, activation=None),
                nn.ReLU(),
                nn.Linear(node_hidden_feats[-1], ntasks)
            )
        else:
            raise NotImplementedError(f"Readout type {self._readout_type} not implemented")

    # TODO: implement reset_parameters
    # def reset_parameters(self):
    #     self.preprocess[0].reset_parameters()
    #     for ll in self.edge_function:
    #         if isinstance(ll, nn.Linear):
    #             ll.reset_parameters()
    #     self.gnn_layer.reset_parameters()
    #     self.gru.reset_parameters()
    #     self.readout.reset_parameters()
    #     for ll in self.predict:
    #         if isinstance(ll, nn.Linear):
    #             ll.reset_parameters()

    def forward(self, graph, node_feats, edge_feats):

        node_feats = self.preprocess(node_feats)
        hidden_feats = node_feats.unsqueeze(0)

        for _ in range(self._num_step_message_passing):
            # Message:
            messages = F.relu(self.gnn_layer(graph, node_feats, edge_feats))
            node_feats, hidden_feats = self.gru(messages.unsqueeze(0), hidden_feats)
            node_feats = node_feats.squeeze(0)
        
        if self._readout_type == 'graph':
            readout = self.graph_readout(graph, node_feats)
        elif self._readout_type == 'node':
            readout = node_feats 
         
        return self.predict(readout)

if __name__ == "__main__":
    import dgl 

    g = dgl.graph(([0, 1, 2, 3], [1, 2, 3, 4]))
    g.ndata['h'] = torch.ones(5, 32)
    g.edata['e'] = torch.ones(4, 16)
    print("Graph: ", g)

    print("--- Node-based task: ---")
    model_node = VanillaMPNN(readout_type='node')
    print(model_node)
    
    model_node.eval()
    x = model_node(g, g.ndata['h'], g.edata['e'])
    print(x.shape)
    print(x)

    print("--- Graph-based task: ---")
    model_graph = VanillaMPNN(readout_type='graph')
    print(model_graph)
    
    model_graph.eval()
    x = model_graph(g, g.ndata['h'], g.edata['e'])
    print(x.shape)
    print(x)
