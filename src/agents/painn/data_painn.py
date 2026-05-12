from typing import List
import sys
import warnings
import torch
import numpy as np
import scipy.spatial
import ase.db

try:
    import asap3
except ModuleNotFoundError:
    warnings.warn("Failed to import asap3 module for fast neighborlist")


def _cell_heights(cell_object):
    volume = cell_object.volume
    crossproducts = np.cross(cell_object[[1, 2, 0]], cell_object[[2, 0, 1]])
    crosslengths = np.sqrt(np.sum(np.square(crossproducts), axis=1))
    heights = volume / crosslengths
    return heights


class AseNeigborListWrapper:
    """
    Wrapper around ASE neighborlist to have the same interface as asap3 neighborlist

    """

    def __init__(self, cutoff, atoms):
        self.neighborlist = ase.neighborlist.NewPrimitiveNeighborList(
            cutoff, skin=0.0, self_interaction=False, bothways=True
        )
        self.neighborlist.build(atoms.get_pbc(), atoms.get_cell(), atoms.get_positions())
        self.cutoff = cutoff
        self.atoms_positions = atoms.get_positions()
        self.atoms_cell = atoms.get_cell()

    def get_neighbors(self, i, cutoff):
        assert cutoff == self.cutoff, "Cutoff must be the same as used to initialise the neighborlist"

        indices, offsets = self.neighborlist.get_neighbors(i)

        rel_positions = self.atoms_positions[indices] + offsets @ self.atoms_cell - self.atoms_positions[i][None]

        dist2 = np.sum(np.square(rel_positions), axis=1)

        return indices, rel_positions, dist2


class TransformAtomsObjectsToGraphXyz:
    """
    Transform Atoms() to graph while keeping the xyz positions of the vertices

    """

    def __init__(self, cutoff=5.0):
        self.cutoff = cutoff

    def __call__(self, atoms):

        if np.any(atoms.get_pbc()):
            atoms.wrap()  # Make sure all atoms are inside unit cell
            edges, edges_displacement = self.get_edges_neighborlist(atoms)
        else:
            edges, edges_displacement = self.get_edges_simple(atoms)

        default_type = torch.get_default_dtype()

        # pylint: disable=E1102
        graph_data = {
            "nodes": torch.tensor(atoms.get_atomic_numbers()),
            "nodes_xyz": torch.tensor(atoms.get_positions(), dtype=default_type),
            "num_nodes": torch.tensor(len(atoms.get_atomic_numbers())),
            "edges": torch.tensor(edges),
            "edges_displacement": torch.tensor(edges_displacement, dtype=default_type),
            "cell": torch.tensor(np.array(atoms.get_cell()), dtype=default_type),
            "num_edges": torch.tensor(edges.shape[0]),
        }

        return graph_data

    def get_edges_simple(self, atoms):
        # Compute distance matrix
        pos = atoms.get_positions()
        dist_mat = scipy.spatial.distance_matrix(pos, pos)

        # Build array with edges and edge features (distances)
        valid_indices_bool = dist_mat < self.cutoff
        np.fill_diagonal(valid_indices_bool, False)  # Remove self-loops
        edges = np.argwhere(valid_indices_bool)  # num_edges x 2
        edges_displacement = np.zeros((edges.shape[0], 3))

        return edges, edges_displacement

    def get_edges_neighborlist(self, atoms):
        edges = []
        edges_displacement = []
        atom_positions = atoms.get_positions()
        inv_cell_T = np.linalg.inv(atoms.get_cell().complete().T)

        # Compute neighborlist
        if (
            np.any(atoms.get_cell().lengths() <= 0.0001)
            or (np.any(atoms.get_pbc()) and np.any(_cell_heights(atoms.get_cell()) < self.cutoff))
            or ("asap3" not in sys.modules)
        ):
            neighborlist = AseNeigborListWrapper(self.cutoff, atoms)
        else:
            neighborlist = asap3.FullNeighborList(self.cutoff, atoms)

        for i in range(len(atoms)):
            neigh_idx, neigh_vec, _ = neighborlist.get_neighbors(i, self.cutoff)

            self_index = np.ones_like(neigh_idx) * i
            this_edges = np.stack((neigh_idx, self_index), axis=1)

            neigh_pos = atom_positions[neigh_idx]
            this_pos = atom_positions[i]
            neigh_origin = neigh_vec + this_pos - neigh_pos
            neigh_origin_scaled = np.round(inv_cell_T.dot(neigh_origin.T).T)

            ############
            # assert np.allclose(neigh_pos + (neigh_origin_scaled @ atoms.get_cell()) - this_pos, neigh_vec)
            ############

            edges.append(this_edges)
            edges_displacement.append(neigh_origin_scaled)

        return np.concatenate(edges), np.concatenate(edges_displacement)


def pad_and_stack(tensors: List[torch.Tensor]):
    """Pad list of tensors if tensors are arrays and stack if they are scalars"""
    if tensors[0].shape:
        return torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=0)
    return torch.stack(tensors)


def collate_atomsdata(graphs: List[dict], pin_memory=True):
    # Convert from "list of dicts" to "dict of lists"
    dict_of_lists = {k: [dic[k] for dic in graphs] for k in graphs[0]}
    # Convert each list of tensors to single tensor with pad and stack
    if pin_memory:
        pin = lambda x: x.pin_memory()  # noqa: E731
    else:
        pin = lambda x: x  # noqa: E731

    collated = {k: pin(pad_and_stack(dict_of_lists[k])) for k in dict_of_lists}
    return collated
