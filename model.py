import torch
import torch.nn as nn
from torch_scatter import scatter
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class GNNLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, attn_dim, n_rel, act=lambda x:x, add_manual_edges=False):
        super(GNNLayer, self).__init__()
        self.n_rel = n_rel
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attn_dim = attn_dim
        self.act = act
        n_embed = 2*n_rel+3 if add_manual_edges else 2*n_rel+1
        self.rela_embed = nn.Embedding(n_embed, in_dim)
        self.Ws_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wr_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wqr_attn = nn.Linear(in_dim, attn_dim)
        self.w_alpha  = nn.Linear(attn_dim, 1)
        self.W_h = nn.Linear(in_dim, out_dim, bias=False)
    
    def forward(self, q_sub, q_rel, r_idx, hidden, edges, n_node, shortcut=False):
        # edges: [h, r, t]
        sub = edges[:,0]
        rel = edges[:,1]
        obj = edges[:,2]
        
        # Mathematically equivalent projection optimizations to save massive GPU memory
        ws_proj = self.Ws_attn(hidden)[sub]
        wr_weight = self.Wr_attn(self.rela_embed.weight)
        wr_proj = F.embedding(rel, wr_weight)
        wqr_proj = self.Wqr_attn(self.rela_embed(q_rel))[r_idx]
        
        # message aggregation
        alpha = torch.sigmoid(self.w_alpha(nn.ReLU()(ws_proj + wr_proj + wqr_proj)))
        
        hs = hidden[sub]
        hr = self.rela_embed(rel)
        message = hs * hr
        message = alpha * message        
        message_agg = scatter(message, index=obj, dim=0, dim_size=n_node, reduce='sum') #ori
        
        # get new hidden representations
        hidden_new = self.act(self.W_h(message_agg))
        
        if shortcut: hidden_new = hidden_new + hidden
        
        return hidden_new

class PropagationCell(nn.Module):
    def __init__(self, gnn_layer, gate, dropout, shortcut):
        super(PropagationCell, self).__init__()
        self.gnn_layer = gnn_layer
        self.gate = gate
        self.dropout = dropout
        self.shortcut = shortcut

    def forward(self, hidden, h0, q_sub, q_rel, edge_batch_idxs, batch_sampled_edges, n_node):
        hidden = self.gnn_layer(q_sub, q_rel, edge_batch_idxs, hidden, batch_sampled_edges, n_node, shortcut=self.shortcut)
        act_signal = (hidden.sum(-1) == 0).detach().int()
        hidden = self.dropout(hidden)
        hidden, h0 = self.gate(hidden.unsqueeze(0), h0)
        hidden = hidden.squeeze(0)
        hidden = hidden * (1-act_signal).unsqueeze(-1)
        h0 = h0 * (1-act_signal).unsqueeze(-1).unsqueeze(0)
        return hidden, h0

class GNN_auto(torch.nn.Module):
    def __init__(self, params, loader):
        super(GNN_auto, self).__init__()
        self.params = params
        self.n_layer = params.n_layer
        self.hidden_dim = params.hidden_dim
        self.attn_dim = params.attn_dim
        self.n_rel = params.n_rel
        self.n_ent = params.n_ent
        self.loader = loader
        acts = {'relu': nn.ReLU(), 'tanh': torch.tanh, 'idd': lambda x:x}
        act = acts[params.act]

        add_manual_edges = getattr(params, 'add_manual_edges', False)
        self.gnn_layers = []
        for i in range(self.n_layer):
            self.gnn_layers.append(GNNLayer(self.hidden_dim, self.hidden_dim, self.attn_dim, self.n_rel, act=act, add_manual_edges=add_manual_edges))
        self.gnn_layers = nn.ModuleList(self.gnn_layers)
        self.dropout = nn.Dropout(params.dropout)
        self.gate = nn.GRU(self.hidden_dim, self.hidden_dim)
        
        self.cells = nn.ModuleList([
            PropagationCell(self.gnn_layers[i], self.gate, self.dropout, params.shortcut)
            for i in range(self.n_layer)
        ])
        
        n_query_embed = 2*self.n_rel+3 if add_manual_edges else 2*self.n_rel+1
        if self.params.initializer == 'relation': self.query_rela_embed = nn.Embedding(n_query_embed, self.hidden_dim)
        if self.params.readout == 'linear':
            if self.params.concatHidden:
                self.W_final = nn.Linear(self.hidden_dim * (self.n_layer+1), 1, bias=False)
            else:
                self.W_final = nn.Linear(self.hidden_dim, 1, bias=False)
        
    def forward(self, q_sub, q_rel, subgraph_data, mode='train'):
        ''' forward with extra propagation '''
        n = len(q_sub)
        batch_idxs, abs_idxs, query_sub_idxs, edge_batch_idxs, batch_sampled_edges = subgraph_data
        n_node = len(batch_idxs)
        h0 = torch.zeros((1, n_node, self.hidden_dim)).cuda()
        hidden = torch.zeros(n_node, self.hidden_dim).cuda()
        
        # initialize the hidden
        if self.params.initializer == 'binary':
            hidden[query_sub_idxs, :] = 1
        elif self.params.initializer == 'relation':
            hidden[query_sub_idxs, :] = self.query_rela_embed(q_rel)
        
        # store hidden at each layer or not
        if self.params.concatHidden: hidden_list = [hidden]
        
        # propagation
        for i in range(self.n_layer):
            if self.training:
                # Use gradient checkpointing to save VRAM during training
                hidden, h0 = checkpoint(
                    self.cells[i],
                    hidden, h0, q_sub, q_rel, edge_batch_idxs, batch_sampled_edges, n_node,
                    use_reentrant=False
                )
            else:
                hidden, h0 = self.cells[i](
                    hidden, h0, q_sub, q_rel, edge_batch_idxs, batch_sampled_edges, n_node
                )
            
            if self.params.concatHidden: hidden_list.append(hidden)

        # readout
        if self.params.readout == 'linear':
            if self.params.concatHidden: hidden = torch.cat(hidden_list, dim=-1)
            scores = self.W_final(hidden).squeeze(-1)        
        elif self.params.readout == 'multiply':
            if self.params.concatHidden: hidden = torch.cat(hidden_list, dim=-1)
            scores = torch.sum(hidden * hidden[query_sub_idxs][batch_idxs], dim=-1)
        
        # re-indexing
        scores_all = torch.zeros((n, self.loader.n_ent)).cuda()
        scores_all[batch_idxs, abs_idxs] = scores.float()

        return scores_all