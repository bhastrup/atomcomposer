from typing import Tuple

import torch
import torch.distributions
# import torch_scatter


def to_one_hot(indices: torch.Tensor, num_classes: int, device=None) -> torch.Tensor:
    """
    Generates one-hot encoding with <num_classes> classes from <indices>

    :param indices: (N x 1) tensor
    :param num_classes: number of classes
    :param device: torch device
    :return: (N x num_classes) tensor
    """
    shape = (*indices.shape[:-1], num_classes)
    oh = torch.zeros(shape, device=device).view(-1, num_classes)

    # scatter_ is the in-place version of scatter
    oh.scatter_(1, indices.view(-1, 1), 1)

    return oh.view(*shape)


def init_layer(layer: torch.nn.Linear, w_scale=1.0) -> torch.nn.Linear:
    torch.nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)  # type: ignore
    torch.nn.init.constant_(layer.bias.data, 0)
    return layer


class MLP(torch.nn.Module):
    def __init__(self, input_dim: int, output_dims: Tuple[int, ...] = (64, 64), gate=torch.nn.functional.relu):
        super().__init__()
        dims = (input_dim,) + output_dims
        self.layers = torch.nn.ModuleList(
            [init_layer(torch.nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])]
        )
        self.gate = gate
        self.output_dim = dims[-1]

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.gate(layer(x))
        x = self.layers[-1](x)
        return x


# def masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
#     return torch_scatter.composite.scatter_softmax(src=logits, index=mask.to(torch.long), dim=-1) * mask


def masked_softmax(
    vector: torch.Tensor,
    mask: torch.Tensor,
    dim: int = -1,
    memory_efficient: bool = False,
    mask_fill_value: float = -1e32,
) -> torch.Tensor:
    """
    ``torch.nn.functional.softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular softmax.
    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    If ``memory_efficient`` is set to true, we will simply use a very large negative number for those
    masked positions so that the probabilities of those positions would be approximately 0.
    This is not accurate in math, but works for most cases and consumes less memory.
    In the case that the input vector is completely masked and ``memory_efficient`` is false, this function
    returns an array of ``0.0``. This behavior may cause ``NaN`` if this is used as the last layer of
    a model that uses categorical cross-entropy loss. Instead, if ``memory_efficient`` is true, this function
    will treat every element as equal, and do softmax over equal numbers.
    """
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            # To limit numerical errors from large vector elements outside the mask, we zero these out.
            result = torch.nn.functional.softmax(vector * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            masked_vector = vector.masked_fill((1 - mask).byte(), mask_fill_value)
            result = torch.nn.functional.softmax(masked_vector, dim=dim)
    return result
