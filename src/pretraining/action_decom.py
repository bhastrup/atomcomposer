from collections import deque

import numpy as np
from numpy.linalg import norm
import torch

import ase
from ase import neighborlist

from src.tools import util
from src.tools.util import to_numpy
from src.rl.envs.environment import tmqmEnv, HeavyFirst
from src.rl.reward import InteractionReward
from src.rl.spaces import FormulaType

from src.rl.buffer_container import PPOBufferContainer
from src.rl.env_container import SimpleEnvContainer


def decompose_pos(
    atoms, pos: np.ndarray, decom_method: str, cutoff: float, shuffle: bool, mega_shuffle: bool, hydrogen_delay: bool
) -> np.ndarray:
    """Decompose the position array using breath-first search starting from the transition metal atom"""

    if decom_method == "dfs":
        return dfs_sort_nodes(atoms, pos, cutoff, shuffle, mega_shuffle, hydrogen_delay=hydrogen_delay)
    elif decom_method == "bfs":
        return bfs_sort_nodes(atoms, pos, cutoff, shuffle, mega_shuffle, hydrogen_delay=hydrogen_delay)
    else:
        raise ValueError(f"Invalid decomposition method: {decom_method}")


def get_connectivity_matrix(atoms, positions):
    atoms_object = ase.Atoms(atoms, positions)
    cutOff = neighborlist.natural_cutoffs(atoms_object)
    neighborList = neighborlist.NeighborList(cutOff, self_interaction=False, bothways=True)
    neighborList.update(atoms_object)
    connectivity = neighborList.get_connectivity_matrix().keys()

    return connectivity


def expand_cutoff(distances_mutual, cutoff):
    all_connected = False
    while not all_connected:
        connected = distances_mutual[0] < cutoff
        while not all_connected:
            old_connected = connected.copy()
            connected = np.logical_or(old_connected, np.any(distances_mutual[connected] < cutoff, axis=0))
            if sum(connected) == len(connected):
                all_connected = True
                # print(f'Cutoff large enough: {cutoff:.2f} Å')
                break
            if np.all(old_connected == connected):
                # print(f'Further expansion not possible. Increasing cutoff to {cutoff:.2f} Å')
                cutoff += 0.1
                break
    return cutoff


def dfs_sort_nodes(atoms, positions, cutoff=None, shuffle=False, mega_shuffle=False, hydrogen_delay=True):
    """
    The function first calculates the distances of all atoms from the origin and finds the index of
    the atom closest to the origin. It then initializes a stack for the DFS traversal, a set to keep
    track of visited nodes, and an empty list to store the sorted node indices.
    The DFS traversal starts from the atom closest to the origin, and the function iteratively visits
    unvisited neighbors of each node and adds them to the stack until all nodes have been visited.
    The function returns the list of sorted node indices.

    Note: The function assumes that the atoms in the input array are connected, i.e., there is a path
    between any two atoms in the molecule. If the molecule has disconnected components, the function
    will only sort the atoms in the component that contains the atom closest to the origin.
    """

    if hydrogen_delay:
        atoms = np.array(atoms)
        hydrogen_indices = np.where(atoms == 1)[0]
        atoms = list(np.delete(atoms, hydrogen_indices, axis=0))
        positions = np.delete(positions, hydrogen_indices, axis=0)

    connectivity = get_connectivity_matrix(atoms, positions)

    # Calculate the distances of all atoms from the origin
    distances_origo = np.sqrt(np.sum(positions**2, axis=1))

    # Calculate all mutual distances
    distances_mutual = np.linalg.norm(positions[:, None] - positions[None, :], axis=2)

    # Choose reasonable cutoff
    min_cutoff = np.min(distances_mutual + np.eye(positions.shape[0]) * (10**6), axis=0).max()
    cutoff = max(min_cutoff, cutoff + 0.01) if cutoff is not None else min_cutoff
    cutoff = expand_cutoff(distances_mutual, cutoff)  # There has to be a connected path from the origin to all atoms

    # Find the index of the atom closest to the origin
    start_index = np.argmin(distances_origo)

    # Initialize a stack for the DFS traversal
    stack = [start_index]

    # Initialize a set to keep track of visited nodes
    visited = set()

    # Initialize an empty list to store the sorted node indices
    sorted_indices = []

    # Perform DFS until all nodes have been visited
    while stack:
        if mega_shuffle:
            stack = np.random.permutation(stack)

        # Pop the last node from the stack
        current_index = stack.pop()

        # Add the current node to the sorted list
        if current_index not in visited:
            sorted_indices.append(current_index)

        # Add the current node to the visited set
        visited.add(current_index)

        # Find the neighbors of the current node that are within the cutoff distance.
        neighbors = np.where(distances_mutual[current_index] < cutoff)[0]
        neighbors = neighbors[distances_mutual[current_index, neighbors].argsort()][::-1]
        if shuffle:
            neighbors = neighbors[np.random.permutation(len(neighbors))]

        # Add unvisited neighbors to the stack
        for neighbor_index in neighbors:
            if neighbor_index not in visited and (neighbor_index, current_index) in connectivity:
                stack.append(neighbor_index)

    if len(sorted_indices) != len(positions):
        print(
            "Not all nodes have been visited exactly once. len(sorted_indices): ",
            len(sorted_indices),
            "len(positions): ",
            len(positions),
        )
        return None

    # append the hydrogen indices to the sorted indices (shuffle randomly)
    sorted_indices = sorted_indices + (list(np.random.permutation(hydrogen_indices)) if hydrogen_delay else [])

    return sorted_indices


def bfs_sort_nodes(atoms, positions, cutoff=None, shuffle=False, mega_shuffle=False, hydrogen_delay=True):

    if hydrogen_delay:
        atoms = np.array(atoms)
        hydrogen_indices = np.where(atoms == 1)[0]
        atoms = list(np.delete(atoms, hydrogen_indices, axis=0))
        positions = np.delete(positions, hydrogen_indices, axis=0)

    connectivity = get_connectivity_matrix(atoms, positions)

    # Calculate the distances of all atoms from the origin
    distances = np.sqrt(np.sum(positions**2, axis=1))

    # Calculate all mutual distances
    distances_mutual = np.linalg.norm(positions[:, None] - positions[None, :], axis=2)

    # Choose reasonable cutoff
    min_cutoff = np.min(distances_mutual + np.eye(positions.shape[0]) * (10**6), axis=0).max()
    cutoff = max(min_cutoff, cutoff + 0.01) if cutoff is not None else min_cutoff
    cutoff = expand_cutoff(distances_mutual, cutoff)  # There has to be a connected path from the origin to all atoms

    # Find the index of the atom closest to the origin
    start_index = np.argmin(distances)

    # Initialize a queue for the BFS traversal
    queue = deque([start_index])

    # Initialize a set to keep track of visited nodes
    visited = set()

    # Initialize an empty list to store the sorted node indices
    sorted_indices = []

    # Perform BFS until all nodes have been visited
    while queue:
        if mega_shuffle:
            queue = deque(np.random.permutation(list(queue)))

        # Pop the first node from the queue
        current_index = queue.popleft()

        # Add the current node to the visited set
        visited.add(current_index)

        # Add the current node to the sorted list
        sorted_indices.append(current_index)

        # Find the neighbors of the current node
        neighbors = np.where(distances_mutual[current_index] < cutoff)[0]
        neighbors = neighbors[distances_mutual[current_index, neighbors].argsort()]
        if shuffle:
            neighbors = neighbors[np.random.permutation(len(neighbors))]

        # Add unvisited neighbors to the queue
        for neighbor_index in neighbors:
            if (
                neighbor_index not in visited
                and neighbor_index not in queue
                and (neighbor_index, current_index) in connectivity
            ):
                queue.append(neighbor_index)

    if len(sorted_indices) != len(positions):
        print(
            "Not all nodes have been visited exactly once. len(sorted_indices): ",
            len(sorted_indices),
            "len(positions): ",
            len(positions),
        )
        return None

    # append the hydrogen indices to the sorted indices (shuffle randomly)
    sorted_indices = sorted_indices + (list(np.random.permutation(hydrogen_indices)) if hydrogen_delay else [])

    return np.array(sorted_indices)


# TODO: Move to internal agent folder
def pos_seq_to_actions(pos, atomic_numbers, zs, no_hydro_focus=True):
    """Breaks down the molecule into a trajectory of expert actions,
    based on the MolGym internal action policy.

    actions:
        stop: 0
        focus: 1
        element: 2
        distance: 3
        angle: 4
        dihedral: 5
        kappa: 6
    """

    # TODO: Remove loop and make it vectorized
    from src.agents.zmat import position_atom_helper

    pos = torch.tensor(pos, dtype=torch.double)
    atomic_numbers = torch.tensor(atomic_numbers, dtype=torch.long)

    mutual_distances = torch.linalg.norm(pos[:, None] - pos[None], axis=2)
    mutual_distances += torch.eye(len(pos)) * 1e4

    all_actions = torch.zeros(size=(pos.shape[0] - 1, 7), dtype=torch.float64)

    for i in range(1, pos.shape[0]):
        # Stop is always 0
        stop = torch.zeros(1, dtype=torch.float)

        # Focus is the nearest neighbor on the canvas
        focus = (mutual_distances[i, :i] + (1e4 * (atomic_numbers[:i] == 1) if no_hydro_focus else 0)).argmin()

        # Element is the element of the atom
        element = zs.index(atomic_numbers[i])

        # Distance is the distance to the nearest neighbor
        distance = torch.sqrt(torch.sum((pos[focus] - pos[i]) ** 2))

        if i == 1:
            angle = torch.zeros(1, dtype=torch.float)
            dihedral = torch.zeros(1, dtype=torch.float)
            kappa = torch.zeros(1, dtype=torch.float)
        elif i == 2:
            # Angle is the angle between new_to_focus and new_to_N1
            first_neighbor = 1 - focus
            focus_to_new = pos[i] - pos[focus]
            focus_to_N1 = pos[first_neighbor] - pos[focus]

            angle = torch.arccos(
                torch.clamp(
                    torch.dot(focus_to_new, focus_to_N1)
                    / (torch.linalg.norm(focus_to_new) * torch.linalg.norm(focus_to_N1)),
                    -1,
                    1,
                )
            ).unsqueeze(-1)

            # Dihedral is undefined in this case
            dihedral = torch.zeros(1, dtype=torch.float)
            kappa = torch.zeros(1, dtype=torch.float)
        else:
            first_neighbor, second_neighbor, third_neighbor = mutual_distances[focus, :i].argsort()[:3]

            # if torch.abs(d1 - d2) < 1e-7:
            #     # Can lead to disagreements with the zmat.position_atom_helper and thus to wrong actions
            #     print(f'NUMERICAL HACK: Perturbing position of FIRST NEIGHBOR of focus (pushing is closer to focus).')
            #     perturb_direction = (pos[focus] - pos[first_neighbor]) / torch.linalg.norm(pos[focus] - pos[first_neighbor])
            #     pos[first_neighbor] += 1e-4 * perturb_direction
            #     mutual_distances = torch.linalg.norm(pos[:, None] - pos[None], axis=2)
            #     mutual_distances += torch.eye(len(pos)) * 1e4

            # if torch.abs(d2 - d3) < 1e-7:
            #     print(f'first_neighbor, second_neighbor, third_neighbor: {first_neighbor}, {second_neighbor}, {third_neighbor}')
            #     # Can lead to disagreements with the zmat.position_atom_helper and thus to wrong actions
            #     print(f'NUMERICAL HACK: PERTURBING POSITION OF SECOND NEIGHBOR OF FOCUS (pulling is closer to focus, but no closer than first neighbor).')
            #     perturb_direction = (pos[focus] - pos[second_neighbor]) / torch.linalg.norm(pos[focus] - pos[second_neighbor])
            #     perturb_dist = min((mutual_distances[focus, second_neighbor] - mutual_distances[focus, first_neighbor]) / 2, 1e-6)
            #     pos[second_neighbor] += perturb_dist * perturb_direction

            focus_to_new = pos[i] - pos[focus]
            focus_to_N1 = pos[first_neighbor] - pos[focus]

            angle = torch.arccos(
                torch.clamp(
                    torch.dot(focus_to_new, focus_to_N1)
                    / (torch.linalg.norm(focus_to_new) * torch.linalg.norm(focus_to_N1)),
                    -1,
                    1,
                )
            ).unsqueeze(-1)

            # division by zero, do the same as we do below for kappa
            # question is wether angle should be 0 or pi
            if torch.isnan(angle):
                # print all tensors in angle
                print("angle is nan")
                print("focus_to_new", focus_to_new)
                print("focus_to_N1", focus_to_N1)
                print("torch.linalg.norm(focus_to_new)", torch.linalg.norm(focus_to_new))
                print("torch.linalg.norm(focus_to_N1)", torch.linalg.norm(focus_to_N1))
                print("torch.dot(focus_to_new, focus_to_N1)", torch.dot(focus_to_new, focus_to_N1))
                print(
                    "torch.dot(focus_to_new, focus_to_N1) / (torch.linalg.norm(focus_to_new) * torch.linalg.norm(focus_to_N1))",
                    torch.dot(focus_to_new, focus_to_N1)
                    / (torch.linalg.norm(focus_to_new) * torch.linalg.norm(focus_to_N1)),
                )

                # view ase Atoms
                from ase import Atoms

                atoms = Atoms(positions=pos, numbers=atomic_numbers)
                atoms.view()
                exit()

            # To calculate dihedral, we need to calculate the two normal vectors of the planes (x, x_f, x_N1) and (x_f, x_N1, x_N2)

            # Normal vector of the plane (x, x_f, x_N1)
            focus_to_n1 = pos[first_neighbor] - pos[focus]
            focus_to_new = pos[i] - pos[focus]
            normal1 = torch.cross(focus_to_new, focus_to_n1)

            # Normal vector of the plane (x_f, x_N1, x_N2)
            focus_to_n2 = pos[second_neighbor] - pos[focus]
            normal2 = torch.cross(focus_to_n2, focus_to_n1)  # flipped

            # Dihedral is the angle between the two normal vectors
            dihedral = torch.arccos(
                torch.clamp(
                    torch.dot(normal2, normal1) / (torch.linalg.norm(normal1) * torch.linalg.norm(normal2)), -1, 1
                )
            ).unsqueeze(-1)

            if dihedral != torch.arccos(
                torch.dot(normal2, normal1) / (torch.linalg.norm(normal1) * torch.linalg.norm(normal2))
            ).unsqueeze(-1):
                print("DIHEDRAL WAS CLAMPED!")

            if torch.isnan(dihedral):
                if torch.linalg.norm(normal1) == 0 or torch.linalg.norm(normal2) == 0:
                    print(
                        "normal1 or normal2 is of zero length. Coordinate system undefined. Sampling random dihedral angle between 0 and pi"
                    )
                    if torch.linalg.norm(normal1) == 0:
                        print("normal1 is zero -> focus_to_new, focus_to_n1 are (anti)parallel")
                    elif torch.linalg.norm(normal2) == 0:
                        print("normal2 is zero -> focus_to_n2, focus_to_n1 are (anti)parallel")
                    dihedral = torch.rand(1) * np.pi

                # # print all tensors in dihedral
                # print('dihedral is nan')
                # print('focus_to_new', focus_to_new)
                # print('focus_to_N1', focus_to_N1)
                # print('focus_to_n2', focus_to_n2)
                # print('normal1', normal1)
                # print('normal2', normal2)
                # print('torch.linalg.norm(normal1)', torch.linalg.norm(normal1))
                # print('torch.linalg.norm(normal2)', torch.linalg.norm(normal2))
                # print('torch.dot(normal2, normal1)', torch.dot(normal2, normal1))
                # print('torch.dot(normal2, normal1) / (torch.linalg.norm(normal1) * torch.linalg.norm(normal2))', torch.dot(normal2, normal1) / (torch.linalg.norm(normal1) * torch.linalg.norm(normal2)))

                # from ase import Atoms
                # from ase.visualize import view
                # atoms = Atoms(positions=pos, numbers=atomic_numbers)
                # view(atoms)
                # exit()

            # Kappa can be either 1 or -1. Try both and see which one is closer to the actual position
            trial_poses = torch.tensor(
                np.array(
                    [
                        position_atom_helper(
                            positions=pos[:i].clone().numpy(),
                            focus=focus.clone().numpy(),
                            distance=distance.clone().numpy(),
                            angle=angle.clone().numpy(),
                            dihedral=sign * dihedral.clone().numpy(),
                        )
                        for sign in [1, -1]
                    ]
                )
            )
            kappa = torch.argmin(torch.linalg.norm(trial_poses - pos[i], axis=1))
            # print(f'kappa dists {torch.linalg.norm(trial_poses - pos[i], axis=1)}')
            # if i == 4 and (torch.linalg.norm(mutual_distances[focus, :i].sort().values - torch.tensor([1.0941e+00, 1.0950e+00, 1.0950e+00, 1.0000e+04])) < 0.01):
            #    kappa = torch.argmax(torch.linalg.norm(trial_poses - pos[i], axis=1))
            #    print(f'FLIPPING KAPPA AROUND')

            # print(f'mutual_distances[focus, :i].argsort(): {mutual_distances[focus, :i].sort()}')

        focus = torch.tensor([focus], dtype=torch.float32)
        element = torch.tensor([element], dtype=torch.float32)
        distance = torch.tensor([distance], dtype=torch.float32)
        angle = torch.tensor([angle], dtype=torch.float32)
        dihedral = torch.tensor([dihedral], dtype=torch.float32)
        kappa = torch.tensor([kappa], dtype=torch.float32)
        # for arr in [stop, focus, element, distance, angle, dihedral, kappa]:
        #     print(f'{arr.shape} {arr.dtype} {arr}')

        all_actions[i - 1] = torch.cat([stop, focus, element, distance, angle, dihedral, kappa])

    return all_actions


# TODO: Move to Euclidean agent folder
def pos_seq_to_actions_emma(pos, atomic_numbers, zs, no_hydro_focus=True):
    """Breaks down the molecule into a trajectory of expert actions. Apart from the usual
    categorical subactions, the actions correspond to Euclidean unit vectors and distances.

    For now, IGNORE that the equivariant tensors don't span the 3d space when the molcule is still planar.

    actions:
        stop: 0
        focus: 1
        element: 2
        distance: 3
        direction_x: 4
        direction_y: 5
        direction_z: 6
    """

    action_dim = 7

    pos = torch.tensor(pos, dtype=torch.float32)
    atomic_numbers = torch.tensor(atomic_numbers, dtype=torch.long)

    mutual_distances = torch.linalg.norm(pos[:, None] - pos[None], axis=2)
    mutual_distances += torch.eye(len(pos)) * 1e4

    all_actions = torch.zeros(size=(pos.shape[0] - 1, action_dim), dtype=torch.float32)

    for i in range(1, pos.shape[0]):
        # Stop is always 0
        stop = torch.zeros(1, dtype=torch.float32)

        # Focus is the nearest neighbor on the canvas
        focus = (mutual_distances[i, :i] + (1e4 * (atomic_numbers[:i] == 1) if no_hydro_focus else 0)).argmin()

        # Element is the element of the atom
        element = zs.index(atomic_numbers[i])

        # Distance is the distance to the nearest neighbor
        distance = torch.sqrt(torch.sum((pos[focus] - pos[i]) ** 2))

        # Direction (unit length)
        direction = (pos[i] - pos[focus]) / distance

        # Collect all actions
        all_actions[i - 1] = torch.cat(
            [stop, focus.unsqueeze(-1), torch.tensor([element], dtype=torch.float32), distance.unsqueeze(-1), direction]
        )

    return all_actions


def pos_seq_to_actions_explorer(pos, atomic_numbers, zs, no_hydro_focus=True, num_trials: int = 15):
    """
    Breaks down the molecule into a trajectory of expert actions.
    It picks the focus and element as usual, but for direction it simply
    one of multiple trial poses, i.e. 4 actions including stop.

    actions:
        stop: 0
        focus: 1
        element: 2
        trial_idx: 3 (of trial poses)
    """

    action_dim = 4
    pos = torch.tensor(pos, dtype=torch.float32)
    atomic_numbers = torch.tensor(atomic_numbers, dtype=torch.long)

    mutual_distances = torch.linalg.norm(pos[:, None] - pos[None], axis=2)
    mutual_distances += torch.eye(len(pos)) * 1e4

    all_actions = torch.zeros(size=(pos.shape[0], action_dim), dtype=torch.float32)

    for i in range(0, pos.shape[0]):
        if i == 0:
            all_actions[i] = torch.zeros(size=(1, action_dim), dtype=torch.float32)
            continue

        # Stop is always 0
        stop = torch.zeros(1, dtype=torch.float32)

        # Focus is the nearest neighbor on the canvas
        focus = (mutual_distances[i, :i] + (1e4 * (atomic_numbers[:i] == 1) if no_hydro_focus else 0)).argmin()

        # Element is the element of the atom
        element = zs.index(atomic_numbers[i])

        # Trial idx
        # Setup n_trials positions on the unit sphere around the focus in an equidistant grid
        trial_poses = torch.tensor(util.fibonacci_sphere(num_trials), dtype=torch.float32) + pos[focus]
        # Find trial pose closest to the new position
        trial_idx = torch.argmin(torch.linalg.norm(trial_poses - pos[i], axis=1))
        trial_idx = torch.tensor([trial_idx], dtype=torch.float32)

        # Collect all actions
        all_actions[i] = torch.cat([stop, focus.unsqueeze(-1), torch.tensor([element], dtype=torch.float32), trial_idx])

    return all_actions


def view_decomposed_molecule(pos, symbols, sorted_indices):
    """
    Expands a list of Atoms() objects by adding the individual atoms in the sorted order
    """
    from ase.visualize import view
    # view(ase.Atoms(symbols=symbols, positions=pos))

    traj = []
    atoms_object = ase.Atoms()
    for i in sorted_indices:
        atoms_object.append(ase.Atom(symbol=symbols[i], position=pos[i]))
        traj.append(atoms_object.copy())
    view(traj)


def build_mol_from_actions(actions, pos, formula, model, observation_space, action_space, config, mol_dataset):
    """Use model.to_action_space() and tmqmEnv.step() to build the molecule trajectory, and then view it"""

    print(f"formula: {formula}")

    reward = InteractionReward(n_workers=config["num_envs"])
    RLEnvironment = tmqmEnv if mol_dataset == "TMQM" else HeavyFirst
    env = RLEnvironment(
        reward=reward,
        observation_space=observation_space,
        action_space=action_space,
        formulas=[formula],
        min_atomic_distance=config["min_atomic_distance"],
        max_solo_distance=config["max_solo_distance"],
        min_reward=config["min_reward"],
        worker_id=0,
    )

    # print(f'actions: {actions}')
    from ase.visualize import view

    obs = env.obs_reset
    for t in range(0, pos.shape[0] - 1):
        print(f"internal action: {actions[t, :]}")
        action = model.to_action_space(actions[t, :], obs)
        obs, reward, done, info = env.step(action)

        if done and t < pos.shape[0] - 2:
            print(f"Failed to build molecule at step {t}")
            from ase import Atom

            atoms_object, _ = observation_space.parse(obs)
            atoms_object.append(Atom(action[0], action[1]))
            from ase.visualize import view

            view(atoms_object)
            exit()
            break
        if t == pos.shape[0] - 2:
            atoms_object, _ = observation_space.parse(obs)
            view(atoms_object)
            break


def replay_episode_with_adv(actions, pos, formula, model, observation_space, action_space, config, mol_dataset):
    """Use model.to_action_space() and tmqmEnv.step() to build the molecule trajectory, and then view it"""

    RLEnvironment = tmqmEnv if mol_dataset == "TMQM" else HeavyFirst
    reward = InteractionReward(reward_coefs=config["reward_coefs"])

    env = RLEnvironment(
        reward=reward,
        observation_space=observation_space,
        action_space=action_space,
        formulas=[formula],
        min_atomic_distance=config["min_atomic_distance"],
        max_solo_distance=config["max_solo_distance"],
        min_reward=config["min_reward"],
        worker_id=0,
    )
    envs = SimpleEnvContainer([env])
    buffer_container = PPOBufferContainer(size=envs.get_size(), gamma=1, lam=0.97)

    all_obs = []

    # obs = env.obs_reset
    observations = [e.obs_reset for e in envs.environments]
    for t in range(0, pos.shape[0] - 1):
        predictions = model.step(observations)
        # action = [model.to_action_space(actions[t, :], obs) for obs in observations]
        action = [model.to_action_space(actions[t, :], observations[0])]
        next_observations, rewards, terminals, _ = envs.step(action)

        buffer_container.store(
            observations=observations,
            actions=to_numpy(predictions["a"]),
            rewards=rewards,
            next_observations=next_observations,
            terminals=terminals,
            values=to_numpy(predictions["v"]),
            logps=to_numpy(predictions["logp"]),
        )

        observations = envs.reset_if_terminal(next_observations, terminals)

        if terminals[0] and t < pos.shape[0] - 2:
            print(
                f"Episode terminated with {t + 2} atoms, but it should have terminated with t={pos.shape[0]}. Consider lowering min_reward?"
            )
            return None, None, None, None

        # all_obs.append(obs)
        # action = model.to_action_space(actions[t, :], obs)
        # obs, reward, done, _ = env.step(action)
        # all_rews.append(reward)
        # all_terminals.append(done)

        # if done and t < pos.shape[0]-2:
        #     print(f'Episode terminated with {t+2} atoms, but it should have terminated with t={pos.shape[0]}. Consider lowering min_reward?')
        #     # from ase import Atom
        #     # atoms_object, _ = observation_space.parse(obs)
        #     # atoms_object.append(Atom(action[0], action[1]))
        #     # from ase.visualize import view
        #     # view(atoms_object)
        #     # exit()

        #     return None, None, None, None

    buffer_container = buffer_container.merge()
    data_dict = buffer_container.get_data_unnormalized()

    # for key in data_dict.keys():
    #     print(f'{key}: {data_dict[key]}')

    all_obs = data_dict["obs"]
    all_rets = data_dict["ret"]
    all_advs = data_dict["adv"]

    # Return
    # all_rets = util.discount_cumsum(np.array(all_rews), discount=1.0)

    # notice that the last obs of the full molecule is excluded, because it is not used for training
    return all_obs, to_numpy(actions), all_rets, all_advs  # np.array(all_rews)  # , np.array(all_terminals)


def replay_episode(actions, pos, formula, model, observation_space, action_space, config, mol_dataset):
    """Use model.to_action_space() and tmqmEnv.step() to build the molecule trajectory, and then view it"""

    RLEnvironment = tmqmEnv if mol_dataset == "TMQM" else HeavyFirst
    reward = InteractionReward(reward_coefs=config["reward_coefs"])

    env = RLEnvironment(
        reward=reward,
        observation_space=observation_space,
        action_space=action_space,
        formulas=[formula],
        min_atomic_distance=config["min_atomic_distance"],
        max_solo_distance=config["max_solo_distance"],
        min_reward=config["min_reward"],
        worker_id=0,
    )

    all_obs = []
    all_rews = []
    all_terminals = []

    obs = env.obs_reset
    for t in range(0, pos.shape[0] - 1):
        all_obs.append(obs)

        action = model.to_action_space(actions[t, :], obs)
        obs, reward, done, _ = env.step(action)

        all_rews.append(reward)
        all_terminals.append(done)

        if done and t < pos.shape[0] - 2:
            print(
                f"Episode terminated with {t + 2} atoms, but it should have terminated with t={pos.shape[0]}. Consider lowering min_reward?"
            )
            # from ase import Atom
            # atoms_object, _ = observation_space.parse(obs)
            # atoms_object.append(Atom(action[0], action[1]))
            # from ase.visualize import view
            # view(atoms_object)
            # exit()

            return None, None, None

    all_rets = util.discount_cumsum(np.array(all_rews), discount=1.0)
    # notice that the last obs of the full molecule is excluded, because it is not used for training
    return all_obs, to_numpy(actions), all_rets  # np.array(all_rews)  # , np.array(all_terminals)


def recenter(
    pos: np.ndarray,
    elements: np.ndarray,
    formula: FormulaType = None,
    mol_dataset: str = "TMQM",
    heavy_first: bool = True,
) -> np.ndarray:

    if mol_dataset == "TMQM":
        # Center the molecule around the transition metal atom
        center_element = [z for (z, _) in formula if z in tmqmEnv.transition_metal_numbers][0]
    else:
        if heavy_first:
            # Sample randomly between those atoms with the highest atomic number
            max_z = max(elements)
            candidates = [i for i, z in enumerate(elements) if z == max_z]
        else:
            # Any element which is not hydrogen
            candidates = [i for i, z in enumerate(elements) if z != 1]

    # Choose uniformly between the candidates
    center_element = np.random.choice(candidates)
    pos = pos - pos[center_element]

    return pos


def rotate_to_axis(positions: np.ndarray, atom_index: int, axis: str = "x") -> np.ndarray:
    """Make sure chosen axis matches the one of determine_coordinates() function in ActorCritic agent module"""

    if positions.shape[1] != 3:
        raise ValueError("Positions array must have shape (n, 3)")
    if not (0 <= atom_index < len(positions)):
        raise ValueError("Atom index is out of bounds")
    if axis not in ["x", "y", "z"]:
        raise ValueError("Axis must be 'x', 'y', or 'z'")

    # Axis vectors
    axis_vectors = {"x": np.array([1, 0, 0]), "y": np.array([0, 1, 0]), "z": np.array([0, 0, 1])}
    target_axis = axis_vectors[axis]

    # Vector from the origin to the target atom
    atom_vector = positions[atom_index]

    # Compute the rotation axis (cross product with target axis vector)
    rotation_axis = np.cross(atom_vector, target_axis)

    if norm(rotation_axis) == 0:  # Check if the vectors are parallel or anti-parallel
        if np.allclose(atom_vector, target_axis):
            return positions  # No rotation needed
        else:  # Anti-parallel case
            # Choose an arbitrary perpendicular axis for rotation
            perpendicular_axis = np.cross(target_axis, np.array([1, 0, 0]))
            if norm(perpendicular_axis) == 0:
                perpendicular_axis = np.cross(target_axis, np.array([0, 1, 0]))
            rotation_axis = perpendicular_axis / norm(perpendicular_axis)
            angle = np.pi  # 180-degree rotation
    else:
        rotation_axis = rotation_axis / norm(rotation_axis)  # Normalize the axis
        # Angle between atom_vector and target axis
        angle = np.arccos(np.dot(atom_vector, target_axis) / norm(atom_vector))

    # Rodrigues' rotation formula components
    K = np.array(
        [
            [0, -rotation_axis[2], rotation_axis[1]],
            [rotation_axis[2], 0, -rotation_axis[0]],
            [-rotation_axis[1], rotation_axis[0], 0],
        ]
    )
    I = np.eye(3)  # noqa: E741

    # Rodrigues' rotation formula
    rotation_matrix = I + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)

    # Apply rotation to all positions
    rotated_positions = np.dot(positions, rotation_matrix.T)

    return rotated_positions


def gaussian_perturbation(pos: np.ndarray, sigma: float = 0.05) -> np.ndarray:
    """Perturb the positions of the atoms by a Gaussian noise"""
    if sigma == 0.0 or sigma is None:
        return pos
    perturbation = np.random.normal(loc=0, scale=sigma, size=pos.shape)
    perturbed_pos = pos + perturbation
    return perturbed_pos
