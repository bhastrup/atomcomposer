import itertools
from typing import Tuple, List

import numpy as np
import torch
from torch import nn


def unpad_and_cat(stacked_seq: torch.Tensor, seq_len: torch.Tensor):
    """
    Unpad and concatenate by removing batch dimension

    Args:
        stacked_seq: (batch_size, max_length, *) Tensor
        seq_len: (batch_size) Tensor with length of each sequence

    Returns:
        (prod(seq_len), *) Tensor

    """
    unstacked = stacked_seq.unbind(0)
    unpadded = [torch.narrow(t, 0, 0, length) for (t, length) in zip(unstacked, seq_len.unbind(0))]
    return torch.cat(unpadded, dim=0)


def calc_distance(
    positions: torch.Tensor,
    cells: torch.Tensor,
    edges: torch.Tensor,
    edges_displacement: torch.Tensor,
    splits: torch.Tensor,
    return_diff=False,
):
    """
    Calculate distance of edges

    Args:
        positions: Tensor of shape (num_nodes, 3) with xyz coordinates inside cell
        cells: Tensor of shape (num_splits, 3, 3) with one unit cell for each split
        edges: Tensor of shape (num_edges, 2)
        edges_displacement: Tensor of shape (num_edges, 3) with the offset (in number of cell vectors) of the sending node
        splits: 1-dimensional tensor with the number of edges for each separate graph
        return_diff: If non-zero return the also the vector corresponding to edges
    """
    unitcell_repeat = torch.repeat_interleave(cells, splits, dim=0)  # num_edges, 3, 3
    displacement = torch.matmul(torch.unsqueeze(edges_displacement, 1), unitcell_repeat)  # num_edges, 1, 3
    displacement = torch.squeeze(displacement, dim=1)
    neigh_pos = positions[edges[:, 0]]  # num_edges, 3
    neigh_abs_pos = neigh_pos + displacement  # num_edges, 3
    this_pos = positions[edges[:, 1]]  # num_edges, 3
    diff = this_pos - neigh_abs_pos  # num_edges, 3
    dist = torch.sqrt(torch.sum(torch.square(diff), dim=1, keepdim=True))  # num_edges, 1

    if return_diff:
        return dist, diff
    else:
        return dist


def sinc_expansion(input_x: torch.Tensor, expand_params: List[Tuple]):
    """
    Expand each feature in a sinc-like basis function expansion.
    Based on [1].
    sin(n*pi*f/rcut)/f

    [1] arXiv:2003.03123 - Directional Message Passing for Molecular Graphs

    Args:
        input_x: (num_edges, num_features) tensor
        expand_params: list of None or (n, cutoff) tuples

    Return:
        (num_edges, n1+n2+...) tensor
    """
    feat_list = torch.unbind(input_x, dim=1)
    expanded_list = []
    for step_tuple, feat in itertools.zip_longest(expand_params, feat_list):
        assert feat is not None, "Too many expansion parameters given"
        if step_tuple:
            n, cutoff = step_tuple
            feat_expanded = torch.unsqueeze(feat, dim=1)
            n_range = torch.arange(n, device=input_x.device, dtype=input_x.dtype) + 1
            # multiplication by pi n_range / cutoff is done in original painn for some reason
            out = torch.sinc(n_range / cutoff * feat_expanded) * np.pi * n_range / cutoff
            expanded_list.append(out)
        else:
            expanded_list.append(torch.unsqueeze(feat, 1))
    return torch.cat(expanded_list, dim=1)


def cosine_cutoff(distance: torch.Tensor, cutoff: float):
    """
    Calculate cutoff value based on distance.
    This uses the cosine Behler-Parinello cutoff function:

    f(d) = 0.5*(cos(pi*d/d_cut)+1) for d < d_cut and 0 otherwise
    """

    return torch.where(
        distance < cutoff,
        0.5 * (torch.cos(np.pi * distance / cutoff) + 1),
        torch.tensor(0.0, device=distance.device, dtype=distance.dtype),
    )


class PaiNNInteraction(nn.Module):
    """Interaction network"""

    def __init__(self, node_size, edge_size, cutoff):
        """
        Args:
            node_size (int): Size of node state
            edge_size (int): Size of edge state
            cutoff (float): Cutoff distance
        """
        super().__init__()

        self.filter_layer = nn.Linear(edge_size, 3 * node_size)

        self.cutoff = cutoff

        self.scalar_message_mlp = nn.Sequential(
            nn.Linear(node_size, node_size),
            nn.SiLU(),
            nn.Linear(node_size, 3 * node_size),
        )

    def forward(
        self,
        node_state_scalar,
        node_state_vector,
        edge_state,
        edge_vector,
        edge_distance,
        edges,
    ):
        """
        Args:
            node_state_scalar (tensor): Node states (num_nodes, node_size)
            node_state_vector (tensor): Node states (num_nodes, 3, node_size)
            edge_state (tensor): Edge states (num_edges, edge_size)
            edge_vector (tensor): Edge vector difference between nodes (num_edges, 3)
            edge_distance (tensor): l2-norm of edge_vector (num_edges, 1)
            edges (tensor): Directed edges with node indices (num_edges, 2)

        Returns:
            Tuple of 2 tensors:
                updated_node_state_scalar (num_nodes, node_size)
                updated_node_state_vector (num_nodes, 3, node_size)
        """
        # Compute all messages
        edge_vector_normalised = edge_vector / torch.maximum(
            torch.linalg.norm(edge_vector, dim=1, keepdim=True), torch.tensor(1e-10)
        )  # num_edges, 3

        filter_weight = self.filter_layer(edge_state)  # num_edges, 3*node_size
        filter_weight = filter_weight * cosine_cutoff(edge_distance, self.cutoff)

        scalar_output = self.scalar_message_mlp(node_state_scalar)  # num_nodes, 3*node_size
        scalar_output = scalar_output[edges[:, 0]]  # num_edges, 3*node_size
        filter_output = filter_weight * scalar_output  # num_edges, 3*node_size

        gate_state_vector, gate_edge_vector, messages_scalar = torch.split(
            filter_output, node_state_scalar.shape[1], dim=1
        )

        gate_state_vector = torch.unsqueeze(gate_state_vector, 1)  # num_edges, 1, node_size
        gate_edge_vector = torch.unsqueeze(gate_edge_vector, 1)  # num_edges, 1, node_size

        # Only include sender in messages
        messages_state_vector = node_state_vector[edges[:, 0]] * gate_state_vector + gate_edge_vector * torch.unsqueeze(
            edge_vector_normalised, 2
        )

        # Sum messages
        message_sum_scalar = torch.zeros_like(node_state_scalar)
        message_sum_scalar.index_add_(0, edges[:, 1], messages_scalar)
        message_sum_vector = torch.zeros_like(node_state_vector)
        message_sum_vector.index_add_(0, edges[:, 1], messages_state_vector)

        # State transition
        new_state_scalar = node_state_scalar + message_sum_scalar
        new_state_vector = node_state_vector + message_sum_vector

        return new_state_scalar, new_state_vector


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.
        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.
        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class PaiNNUpdate(nn.Module):
    """PaiNN style update network. Models the interaction between scalar and vectorial part"""

    def __init__(self, node_size, rms_norm: bool = False):
        super().__init__()

        self.rms_norm = rms_norm
        if rms_norm:
            norm_eps: float = 1e-5
            self.scalar_norm = RMSNorm(node_size, eps=norm_eps)
            self.vector_norm = RMSNorm(node_size, eps=norm_eps)

        self.linearU = nn.Linear(node_size, node_size, bias=False)
        self.linearV = nn.Linear(node_size, node_size, bias=False)
        self.combined_mlp = nn.Sequential(
            nn.Linear(2 * node_size, node_size),
            nn.SiLU(),
            nn.Linear(node_size, 3 * node_size),
        )

    def forward(self, node_state_scalar, node_state_vector):
        """
        Args:
            node_state_scalar (tensor): Node states (num_nodes, node_size)
            node_state_vector (tensor): Node states (num_nodes, 3, node_size)

        Returns:
            Tuple of 2 tensors:
                updated_node_state_scalar (num_nodes, node_size)
                updated_node_state_vector (num_nodes, 3, node_size)
        """

        if hasattr(self, "scalar_norm") and hasattr(self, "vector_norm"):
            if self.rms_norm:
                node_state_vector = self.vector_norm(node_state_vector)
                node_state_scalar = self.scalar_norm(node_state_scalar)

        Uv = self.linearU(node_state_vector)  # num_nodes, 3, node_size
        Vv = self.linearV(node_state_vector)  # num_nodes, 3, node_size

        Vv_norm = torch.linalg.norm(Vv, dim=1, keepdim=False)  # num_nodes, node_size

        mlp_input = torch.cat((node_state_scalar, Vv_norm), dim=1)  # num_nodes, node_size*2
        mlp_output = self.combined_mlp(mlp_input)

        a_ss, a_sv, a_vv = torch.split(mlp_output, node_state_scalar.shape[1], dim=1)  # num_nodes, node_size

        inner_prod = torch.sum(Uv * Vv, dim=1)  # num_nodes, node_size

        delta_v = torch.unsqueeze(a_vv, 1) * Uv  # num_nodes, 3, node_size

        delta_s = a_ss + a_sv * inner_prod  # num_nodes, node_size

        return node_state_scalar + delta_s, node_state_vector + delta_v
