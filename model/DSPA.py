import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse.linalg import eigs
from DSPACell import DSPACell

Tensor = torch.Tensor


def gumbel_softmax(logits: Tensor, tau: float = 1, hard: bool = False, eps: float = 1e-10, dim: int = -1) -> Tensor:
    r"""
    Samples from the Gumbel-Softmax distribution (`Link 1`_  `Link 2`_) and optionally discretizes.

    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd
      dim (int): A dimension along which softmax will be computed. Default: -1.

    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.
      If ``hard=True``, the returned samples will be one-hot, otherwise they will
      be probability distributions that sum to 1 across `dim`.

    .. note::
      This function is here for legacy reasons, may be removed from nn.Functional in the future.

    .. note::
      The main trick for `hard` is to do  `y_hard - y_soft.detach() + y_soft`

      It achieves two things:
      - makes the output value exactly one-hot
      (since we add then subtract y_soft value)
      - makes the gradient equal to y_soft gradient
      (since we strip all other gradients)

    Examples::
        >>> logits = torch.randn(20, 32)
        >>> # Sample soft categorical using reparametrization trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=False)
        >>> # Sample hard categorical using "Straight-through" trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=True)

    .. _Link 1:
        https://arxiv.org/abs/1611.00712
    .. _Link 2:
        https://arxiv.org/abs/1611.01144
    """
    if eps != 1e-10:
        warnings.warn("`eps` parameter is deprecated and has no effect.")

    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    # y_soft = gumbels.softmax(dim)
    y_soft = gumbels

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret
#Data Smoothing
# def ph():#03
#         num_node = 358
#         embed_dim = 10
#         A = nn.Parameter(torch.randn(num_node, embed_dim), requires_grad=True)
#         B = torch.zeros((358, 10))
#         for i in range(1, 356):
#             for j in range(1, 8):
#                 B[i, j] = (4 * A[i, j] + 1 * (A[i - 1, j] + A[i + 1, j] + A[i, j - 1] + A[i, j + 1]) + (1 / 4) * (
#                         A[i - 1, j - 1] + A[i + 1, j - 1] + A[i - 1, j + 1] + A[i + 1, j + 1])) / (21 / 4)
#         i = 1
#         for j in range(1, 8):
#             B[i, j] = (A[i, j - 1] + 2 * A[i, j] + A[i, j + 1]) / 3
#         j = 0
#         for i in range(1, 356):
#             B[i, j] = (A[i - 1, j] + 2 * A[i, j] + A[i + 1, j]) / 3
#         i = 356
#         for j in range(1, 8):
#             B[i, j] = (A[i, j - 1] + 2 * A[i, j] + A[i, j + 1]) / 3
#         j = 9
#         for i in range(1, 356):
#             B[i, j] = (A[i - 1, j] + 2 * A[i, j] + A[i + 1, j]) / 3
#         B[0, 0] = B[0, 9] = B[357, 0] = B[357, 9] = (A[0, 0] + A[0, 9] + A[357, 0] + A[357, 9]) / 4
#         return B
# def ph():#pemsd4
#         num_node=307
#         embed_dim=10
#         A= nn.Parameter(torch.randn(num_node, embed_dim), requires_grad=True)
#         B=torch.zeros((307,10))
#         for i in range(1,305):
#             for j in range(1,8):
#                 B[i,j]=(4*A[i,j]+1*(A[i-1,j]+A[i+1,j]+A[i,j-1]+A[i,j+1])+1/4*(A[i-1,j-1]+A[i+1,j-1]+A[i-1,j+1]+A[i+1,j+1]))/(21/4)
#         i=1
#         for j in range(1,8):
#             B[i,j] = (A[i,j-1]+2*A[i,j]+A[i,j+1])/3
#         j=0
#         for i in range(1,305):
#             B[i, j] = (A[i-1, j ] + 2 * A[i, j] + A[i+1, j ]) / 3
#         i=305
#         for j in range(1,8):
#             B[i, j] = (A[i, j - 1] + 2 * A[i, j] + A[i, j + 1]) / 3
#         j=9
#         for i in range(1,305):
#             B[i, j] = (A[i-1, j ] + 2 * A[i, j] + A[i+1, j ]) / 3
#         B[0,0]=B[0,9]=B[306,0]=B[306,9]=(A[0,0]+A[0,9]+A[306,0]+A[306,9])/4
#         return B


# def ph(): # pemsd7
#         num_node = 883
#         embed_dim = 10
#         A = nn.Parameter(torch.randn(num_node, embed_dim), requires_grad=True)
#         B = torch.zeros((883, 10))
#         for i in range(1, 881):
#             for j in range(1, 8):
#                 B[i, j] = (4 * A[i, j] + 1 * (A[i - 1, j] + A[i + 1, j] + A[i, j - 1] + A[i, j + 1]) + (1 / 4) * (
#                         A[i - 1, j - 1] + A[i + 1, j - 1] + A[i - 1, j + 1] + A[i + 1, j + 1])) / (21 / 4)
#         i = 1
#         for j in range(1, 8):
#             B[i, j] = (A[i, j - 1] + 2 * A[i, j] + A[i, j + 1]) / 3
#         j = 0
#         for i in range(1, 881):
#             B[i, j] = (A[i - 1, j] + 2 * A[i, j] + A[i + 1, j]) / 3
#         i = 881
#         for j in range(1, 8):
#             B[i, j] = (A[i, j - 1] + 2 * A[i, j] + A[i, j + 1]) / 3
#         j = 9
#         for i in range(1, 881):
#             B[i, j] = (A[i - 1, j] + 2 * A[i, j] + A[i + 1, j]) / 3
#         B[0, 0] = B[0, 9] = B[882, 0] = B[882, 9] = (A[0, 0] + A[0, 9] + A[882, 0] + A[882, 9]) / 4
#         return B

def ph():  # pemsd8
        num_node = 170
        embed_dim = 8
        A = nn.Parameter(torch.randn(num_node, embed_dim), requires_grad=True)
        B = torch.zeros((170, 8))
        for i in range(1, 168):
            for j in range(1, 6):
                B[i, j] = (4 * A[i, j] + 1 * (A[i - 1, j] + A[i + 1, j] + A[i, j - 1] + A[i, j + 1]) + (1 / 4) * (
                        A[i - 1, j - 1] + A[i + 1, j - 1] + A[i - 1, j + 1] + A[i + 1, j + 1])) / (21 / 4)

        i = 1
        for j in range(1, 6):
            B[i, j] = (A[i, j - 1] + 2 * A[i, j] + A[i, j + 1]) / 3
        j = 0
        for i in range(1, 168):
            B[i, j] = (A[i - 1, j] + 2 * A[i, j] + A[i + 1, j]) / 3
        i = 168
        for j in range(1, 6):
            B[i, j] = (A[i, j - 1] + 2 * A[i, j] + A[i, j + 1]) / 3
        j = 7
        for i in range(1, 168):
            B[i, j] = (A[i - 1, j] + 2 * A[i, j] + A[i + 1, j]) / 3
        B[0, 0] = B[0, 7] = B[169, 0] = B[169, 7] = (A[0, 0] + A[0, 7] + A[169, 0] + A[169, 7]) / 4
        return B
class AVWDCRNN(nn.Module):
    def __init__(self, cheb_polynomials, L_tilde, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
        super(AVWDCRNN, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(DSPACell(cheb_polynomials, L_tilde, node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(DSPACell(cheb_polynomials, L_tilde, node_num, dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, x, init_state, node_embeddings, learned_tilde):
        #shape of x: (B, T, N, D)
        #shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings, learned_tilde)
                #print(state.shape)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        #current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        #output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        #print(current_inputs.shape)
        return current_inputs, output_hidden


    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)      #(num_layers, B, N, hidden_dim)

class SAG(nn.Module):
    def __init__(self, args,cheb_polynomials, L_tilde):
        super(SAG, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.num_layers = args.num_layers
        self.scaling_factor = args.scaling_factor

        self.default_graph = args.default_graph
        self.node_embeddings = nn.Parameter(ph(), requires_grad=True)
        self.encoder = AVWDCRNN(cheb_polynomials, L_tilde, args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
                                args.embed_dim, args.num_layers)

        #predictor
        self.end_conv = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)
        self.start_conv = nn.Conv2d(args.horizon * self.output_dim, 1, kernel_size=1, bias=True)

        self.adj = None
        self.tilde = None

    def scaled_laplacian(self, node_embeddings, is_eval=False):
        # Normalized graph Laplacian function.
        # :param W: np.ndarray, [n_route, n_route], weighted adjacency matrix of G.
        # :return: np.matrix, [n_route, n_route].
        # learned graph
        node_num = self.num_node
        learned_graph = torch.mm(node_embeddings, node_embeddings.transpose(0, 1))
        norm = torch.norm(node_embeddings, p=2, dim=1, keepdim=True)
        norm = torch.mm(norm, norm.transpose(0, 1))
        learned_graph = learned_graph / norm
        learned_graph = (learned_graph + 1) / 2.
        # learned_graph = F.sigmoid(learned_graph)
        learned_graph = torch.stack([learned_graph, 1-learned_graph], dim=-1)

        # make the adj sparse
        if is_eval:
            adj = gumbel_softmax(learned_graph, tau=1, hard=True)
        else:
            adj = gumbel_softmax(learned_graph, tau=1, hard=True)
        adj = adj[:, :, 0].clone().reshape(node_num, -1)
        # mask = torch.eye(self.num_nodes, self.num_nodes).to(device).byte()
        mask = torch.eye(node_num, node_num).bool().cuda()
        adj.masked_fill_(mask, 0)

        # d ->  diagonal degree matrix
        W = adj
        n = W.shape[0]
        d = torch.sum(W, axis=1)
        ## L -> graph Laplacian
        L = -W
        L[range(len(L)), range(len(L))] = d
        #for i in range(n):
        #    for j in range(n):
        #        if (d[i] > 0) and (d[j] > 0):
        #            L[i, j] = L[i, j] / torch.sqrt(d[i] * d[j])
        ## lambda_max \approx 2.0, the largest eigenvalues of L.
        try:
            # e, _ = torch.eig(L)
            # lambda_max = e[:, 0].max().detach()
            # import pdb; pdb.set_trace()
            # e = torch.linalg.eigvalsh(L)
            # lambda_max = e.max()
            lambda_max = (L.max() - L.min())
        except Exception as e:
            print("eig error!!: {}".format(e))
            lambda_max = 1.0

        # pesudo laplacian matrix, lambda_max = eigs(L.cpu().detach().numpy(), k=1, which='LR')[0][0].real
        tilde = (2 * L / lambda_max - torch.eye(n).cuda())
        self.adj = adj
        self.tilde = tilde
        return adj, tilde

    def forward(self, source,targets, teacher_forcing_ratio=0.5):
        source1=self.scaling_factor * source
        #source: B, T_1, N, D
        #target: B, T_2, N, D
        #supports = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec1.transpose(0,1))), dim=1)
        if self.train:
            adj, learned_tilde = self.scaled_laplacian(self.node_embeddings, is_eval=False)
        else:
            adj, learned_tilde = self.scaled_laplacian(self.node_embeddings, is_eval=True)

        init_state = self.encoder.init_hidden(source.shape[0])
        output, _ = self.encoder(source, init_state, self.node_embeddings, learned_tilde)      #B, T, N, hidden
        output = output[:, -1:, :, :] +self.start_conv(source1)                              #B, 1, N, hidden

        #CNN based predictor
        output = self.end_conv((output))                         #B, T*C, N, 1
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node)
        output = output.permute(0, 1, 3, 2)                             #B, T, N, C

        return output
