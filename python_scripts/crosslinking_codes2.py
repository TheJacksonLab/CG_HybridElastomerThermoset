import numpy as np
import random
# from MDAnalysis.lib.pkdtree import PeriodicKDTree
import copy
from tqdm import tqdm
import extract_local_str as els
import my_common as mc
import os
from scipy.spatial import KDTree, cKDTree

def update_lammps_data(lmp_data, bond_info, lmp_command):
    lmp_tmp = els.lammps(
        natoms=lmp_data.natoms,
        nbonds=len(bond_info),
        natom_types=lmp_data.natom_types,
        nbond_types=lmp_data.nbond_types,
        x=lmp_data.x, y=lmp_data.y, z=lmp_data.z,
        mass=lmp_data.mass,
        atom_info=lmp_data.atom_info,
        bond_info=bond_info
    )
    els.write_lammps_full('tmp.dat', lmp_tmp)
    os.system(lmp_command)
    os.system('cp anneal.dat tmp.dat')

def save_snapshot_if_needed(current_bonds, target_total_bonds, lmp_data, bond_info, last_checkpoint_reached):
    """
    Saves snapshots at 10% crosslinking increments

    Parameters:
    -----------
    current_bonds : int
        Current number of bonds formed during crosslinking
    target_total_bonds : float
        Target total number of bonds to form
    lmp_data : object
        LAMMPS data object
    bond_info : numpy.ndarray
        Bond information array
    last_checkpoint_reached : int
        Last checkpoint reached (0-10)

    Returns:
    --------
    int
        Updated last_checkpoint_reached value
    """
    # Check if we've reached a new checkpoint (every 10% of total target)
    current_checkpoint = int(current_bonds / target_total_bonds * 10)

    if current_checkpoint > last_checkpoint_reached:
        # We've reached a new checkpoint
        last_checkpoint_reached = current_checkpoint
        percent_str = f"{current_checkpoint*10}"
        snapshot_filename = f"crosslink_{percent_str}.dat"

        # Create a temporary lammps object with current data
        lmp_snapshot = els.lammps(
            natoms=lmp_data.natoms,
            nbonds=len(bond_info),
            natom_types=3,
            nbond_types=3,
            x=lmp_data.x, y=lmp_data.y, z=lmp_data.z,
            mass=lmp_data.mass,
            atom_info=lmp_data.atom_info,
            bond_info=bond_info
        )
        els.write_lammps_full(snapshot_filename, lmp_snapshot)
        print(f"Saved snapshot at {current_checkpoint*10}% crosslinking: {snapshot_filename}", flush=True)

    return last_checkpoint_reached

# Scripts to carry out crosslinking of the three hybrid monomer systems

# SYS 1: Individual hybrid monomers and crosslinkers in simulation box

def crosslink_sys1(relaxed_file, output_file, lmp_command, percentage_ck_A, percentage_ck_B,
              functionality_A, functionality_B, functionality_C, ck_dis_A, ck_dis_B,
              probability_AC, probability_B, save_snapshots=False):
    """
    Crosslinking function with probability-based bond formation

    Parameters:
    -----------
    relaxed_file : str
        Input relaxed data file
    output_file : str
        Output crosslinked data file
    lmp_command : str
        Command to run LAMMPS
    percentage_ck_A : float
        Target percentage for A-A, A-C, C-C crosslinking
    percentage_ck_B : float
        Target percentage for B-B crosslinking
    functionality_A : int
        Maximum number of bonds for A-type beads
    functionality_B : int
        Maximum number of bonds for B-type beads
    functionality_C : int
        Maximum number of bonds for C-type beads
    ck_dis_A : float
        Cutoff distance for A-A, A-C, C-C crosslinking
    ck_dis_B : float
        Cutoff distance for B-B crosslinking
    probability_AC : float
        Probability (0-1) of forming a bond between A and C atoms when all other criteria are met
    probability_B : float
        Probability (0-1) of forming a bond between B atoms when all other criteria are met
    save_snapshots : bool
        Whether to save snapshots at different crosslinking percentages (default: False)
    """

    lmp_data = els.read_lammps_full(relaxed_file)
    lmp_data.atom_info = lmp_data.atom_info[np.argsort(lmp_data.atom_info[:, 0])]
    atom_positions = lmp_data.atom_info[:, 4:7]  # Assuming positions are columns 4-7
    atom_types = lmp_data.atom_info[:, 2]  # Assuming atom types are in column 2

    bond_info = lmp_data.bond_info
    bond_index = bond_info[:,2:]-1
    box, coors = els.box_coors_from_lmp(lmp_data)

    positions_A = coors[atom_types == 1]
    idx_A = np.squeeze(np.argwhere(atom_types == 1))
    num_A = len(positions_A)

    positions_B = coors[atom_types == 2]
    idx_B = np.squeeze(np.argwhere(atom_types == 2))
    num_B = len(positions_B)

    positions_C = coors[atom_types == 3]
    idx_C = np.squeeze(np.argwhere(atom_types == 3))
    num_C = len(positions_C)

    # Initialize reaction counters for each A, B, and C bead
    reaction_counter_A = np.zeros(len(positions_A))
    for i in range(len(idx_A)):
        reaction_counter_A[i] = np.sum(np.concatenate(bond_index)==idx_A[i])
    capacity_AA_ideal = (functionality_A*num_A + functionality_C*num_C) - np.sum(reaction_counter_A)

    percentage_ck_A_actual = (percentage_ck_A*capacity_AA_ideal - np.sum(bond_info[:,1]==2))/capacity_AA_ideal
    print(f"AA max={capacity_AA_ideal}, already={np.sum(bond_info[:,1]==2)}, target = {percentage_ck_A*capacity_AA_ideal}")
    print(f"AA actual percentage = {percentage_ck_A_actual}")

    reaction_counter_C = np.zeros(len(positions_C))
    for i in range(len(idx_C)):
        reaction_counter_C[i] = np.sum(np.concatenate(bond_index)==idx_C[i])

    reaction_counter_B = np.zeros(len(positions_B))
    for i in range(len(idx_B)):
        reaction_counter_B[i] = np.sum(np.concatenate(bond_index)==idx_B[i])
    capacity_ck_B = functionality_B * num_B - np.sum(reaction_counter_B)

    print(f"BB max={capacity_ck_B}, target = {percentage_ck_B*capacity_ck_B}")

    # Create a unified connected molecules dictionary for A and C types
    connected_molecules_AC = {}
    # Initialize for A beads
    for idx in idx_A:
        connected_molecules_AC[idx] = set()
    # Initialize for C beads
    for idx in idx_C:
        connected_molecules_AC[idx] = set()

    # Initialize for B beads (separate tracking)
    connected_molecules_B = {idx: set() for idx in idx_B}

    # Pre-populate the connected_molecules dictionaries from existing bonds
    for bond in bond_info:
        if bond[1] == 2:  # If it's an A-A, A-C, or C-C bond
            atom1, atom2 = int(bond[2])-1, int(bond[3])-1
            if atom1 in connected_molecules_AC and atom2 in connected_molecules_AC:
                connected_molecules_AC[atom1].add(atom2)
                connected_molecules_AC[atom2].add(atom1)
        elif bond[1] == 3:  # If it's a B-B bond
            atom1, atom2 = int(bond[2])-1, int(bond[3])-1
            if atom1 in connected_molecules_B and atom2 in connected_molecules_B:
                connected_molecules_B[atom1].add(atom2)
                connected_molecules_B[atom2].add(atom1)

    # Combine A and C bead information
    all_AC_positions = np.vstack([positions_A, positions_C])
    all_AC_indices = np.concatenate([idx_A, idx_C])
    all_AC_types = np.array(['A'] * len(positions_A) + ['C'] * len(positions_C))
    all_AC_functionality = np.concatenate([np.ones(len(positions_A)) * functionality_A,
                                         np.ones(len(positions_C)) * functionality_C])
    # Create a mapping between combined array indices and original indices
    combined_to_original = {}
    for i in range(len(positions_A)):
        combined_to_original[i] = ('A', i)
    for i in range(len(positions_C)):
        combined_to_original[i + len(positions_A)] = ('C', i)

    # Create mapping from combined index to actual atom index in the system
    combined_to_atom_idx = {}
    for i in range(len(positions_A)):
        combined_to_atom_idx[i] = idx_A[i]
    for i in range(len(positions_C)):
        combined_to_atom_idx[i + len(positions_A)] = idx_C[i]

    # Identify potential crosslinking pairs
    num_add_ck_A = 0  # Counter for all A-A, A-C, C-A, and C-C bonds
    num_add_ck_B = 0  # Counter for B-B bonds
    lmp_data.nbond_types = 3

    # Counters for tracking probability-based rejections
    rejected_AC_bonds = 0
    rejected_B_bonds = 0

    max_iterations = 1000  # Safety limit to prevent infinite loops
    iteration = 0

    # For snapshot saving
    target_total_bonds = (capacity_AA_ideal/2 * percentage_ck_A_actual) + (capacity_ck_B/2 * percentage_ck_B)
    last_checkpoint_reached = -1

    # Initial snapshot at 0%
    if save_snapshots:
        last_checkpoint_reached = save_snapshot_if_needed(0, target_total_bonds, lmp_data, bond_info, last_checkpoint_reached)

    # Main crosslinking loop
    while ((num_add_ck_A < capacity_AA_ideal/2 * percentage_ck_A_actual) or
           (num_add_ck_B < capacity_ck_B/2 * percentage_ck_B)) and iteration < max_iterations:
        iteration += 1
        bonds_added_this_iteration = 0

        # Initialize lists to store potential bonds
        potential_AC_bonds = []  # Will store tuples of (i, j, distances) for A-C bonds
        potential_B_bonds = []   # Will store tuples of (i, j, distances) for B-B bonds

        # Find all potential A-C bonds
        if num_add_ck_A < capacity_AA_ideal/2 * percentage_ck_A_actual:
            for i in range(len(all_AC_positions)):
                pos_i = all_AC_positions[i]
                type_i, orig_i = combined_to_original[i]
                atom_idx_i = combined_to_atom_idx[i]

                # Check if this bead has available functionality
                if (type_i == 'A' and reaction_counter_A[orig_i] >= functionality_A) or \
                   (type_i == 'C' and reaction_counter_C[orig_i] >= functionality_C):
                    continue

                # Find all potential partners (A or C beads) within distance
                j_list = np.argwhere((np.abs(all_AC_positions[:, 0] - pos_i[0]) < ck_dis_A) &
                                     (np.abs(all_AC_positions[:, 1] - pos_i[1]) < ck_dis_A) &
                                     (np.abs(all_AC_positions[:, 2] - pos_i[2]) < ck_dis_A)).flatten()

                for j in j_list:
                    if i == j:  # Skip self
                        continue

                    type_j, orig_j = combined_to_original[j]
                    atom_idx_j = combined_to_atom_idx[j]
                    pos_j = all_AC_positions[j]

                    # Check if partner has available functionality
                    if (type_j == 'A' and reaction_counter_A[orig_j] >= functionality_A) or \
                       (type_j == 'C' and reaction_counter_C[orig_j] >= functionality_C):
                        continue

                    # Check distance using PBC
                    distance = mc.pbc_distance(pos_i, pos_j, box)
                    if distance >= ck_dis_A:
                        continue

                    # Check if already connected - using the unified dictionary
                    if atom_idx_j in connected_molecules_AC[atom_idx_i]:
                        continue

                    # Add to potential bonds list - store all relevant information
                    potential_AC_bonds.append((
                        i, j, distance, type_i, orig_i, atom_idx_i, type_j, orig_j, atom_idx_j
                    ))

        # Find all potential B-B bonds
        if num_add_ck_B < capacity_ck_B/2 * percentage_ck_B:
            for i, pos_B in enumerate(positions_B):
                if reaction_counter_B[i] < functionality_B:
                    j_list = np.argwhere((np.abs(positions_B[:, 0] - pos_B[0]) < ck_dis_B) &
                                         (np.abs(positions_B[:, 1] - pos_B[1]) < ck_dis_B) &
                                         (np.abs(positions_B[:, 2] - pos_B[2]) < ck_dis_B)).flatten()

                    for j in j_list:
                        if j == i:  # Skip self
                            continue

                        if reaction_counter_B[j] >= functionality_B:
                            continue

                        distance = mc.pbc_distance(pos_B, positions_B[j], box)
                        if distance >= ck_dis_B:
                            continue

                        if idx_B[j] in connected_molecules_B[idx_B[i]]:
                            continue

                        # Add to potential bonds list
                        potential_B_bonds.append((i, j, distance, idx_B[i], idx_B[j]))

        # Shuffle both lists of potential bonds
        np.random.shuffle(potential_AC_bonds)
        np.random.shuffle(potential_B_bonds)

        # Choose randomly between A-C and B-B bonds until no more potential bonds remain
        while potential_AC_bonds or potential_B_bonds:
            # Decide which type of bond to try next (if both are available)
            if potential_AC_bonds and potential_B_bonds:
                # 50/50 chance of picking A-C vs B-B
                if np.random.random() < 0.5:
                    bond_type = 'AC'
                else:
                    bond_type = 'B'
            elif potential_AC_bonds:
                bond_type = 'AC'
            else:
                bond_type = 'B'

            # Process the chosen bond type
            if bond_type == 'AC':
                # Get the next potential A-C bond
                bond_data = potential_AC_bonds.pop(0)
                i, j, distance, type_i, orig_i, atom_idx_i, type_j, orig_j, atom_idx_j = bond_data

                # Apply probability-based bond formation
                if np.random.random() > probability_AC:
                    rejected_AC_bonds += 1
                    continue

                # Form bond and update counters
                bond_info = np.vstack([bond_info, [len(bond_info) + 1, 2, atom_idx_j + 1, atom_idx_i + 1]])

                # Update reaction counters
                if type_i == 'A':
                    reaction_counter_A[orig_i] += 1
                else:  # type_i == 'C'
                    reaction_counter_C[orig_i] += 1

                if type_j == 'A':
                    reaction_counter_A[orig_j] += 1
                else:  # type_j == 'C'
                    reaction_counter_C[orig_j] += 1

                # Update connected molecules - using the unified dictionary
                connected_molecules_AC[atom_idx_i].add(atom_idx_j)
                connected_molecules_AC[atom_idx_j].add(atom_idx_i)

                # Increment the counter for all A and C bonds
                num_add_ck_A += 1
                bonds_added_this_iteration += 1

                # Filter out any potential bonds that involve these atoms that now have new connections
                potential_AC_bonds = [
                    bond for bond in potential_AC_bonds
                    if not (bond[5] == atom_idx_i or bond[5] == atom_idx_j or
                            bond[8] == atom_idx_i or bond[8] == atom_idx_j)
                ]

                # Check if we should save a snapshot after adding this bond
                if save_snapshots:
                    total_bonds_formed = num_add_ck_A + num_add_ck_B
                    last_checkpoint_reached = save_snapshot_if_needed(
                        total_bonds_formed, target_total_bonds, lmp_data, bond_info, last_checkpoint_reached
                    )

            else:  # bond_type == 'B'
                # Get the next potential B-B bond
                bond_data = potential_B_bonds.pop(0)
                i, j, distance, idx_B_i, idx_B_j = bond_data

                # Apply probability-based bond formation
                if np.random.random() > probability_B:
                    rejected_B_bonds += 1
                    continue

                # Form B-B bond
                bond_info = np.vstack([bond_info, [len(bond_info) + 1, 3, idx_B_j + 1, idx_B_i + 1]])
                reaction_counter_B[i] += 1
                reaction_counter_B[j] += 1
                num_add_ck_B += 1
                bonds_added_this_iteration += 1

                # Update connected molecules for B
                connected_molecules_B[idx_B_i].add(idx_B_j)
                connected_molecules_B[idx_B_j].add(idx_B_i)

                # Filter out any potential bonds that involve these atoms
                potential_B_bonds = [
                    bond for bond in potential_B_bonds
                    if not (bond[3] == idx_B_i or bond[3] == idx_B_j or
                            bond[4] == idx_B_i or bond[4] == idx_B_j)
                ]

                # Check if we should save a snapshot after adding this bond
                if save_snapshots:
                    total_bonds_formed = num_add_ck_A + num_add_ck_B
                    last_checkpoint_reached = save_snapshot_if_needed(
                        total_bonds_formed, target_total_bonds, lmp_data, bond_info, last_checkpoint_reached
                    )

        # Calculate current crosslinking percentages
        ac_percent = num_add_ck_A/(capacity_AA_ideal/2) * 100
        b_percent = num_add_ck_B/(capacity_ck_B/2) * 100

        # Calculate total crosslinking percentage based on both types
        # Weighted average based on capacities
        total_capacity = capacity_AA_ideal/2 + capacity_ck_B/2
        total_bonds_formed = num_add_ck_A + num_add_ck_B
        total_percent = total_bonds_formed / total_capacity * 100
        relative_percent = total_bonds_formed / target_total_bonds * 100

        print(f'Iteration {iteration}:', flush=True)
        print(f'  - AA crosslinking: {ac_percent:.2f}% of target {percentage_ck_A_actual*100:.2f}%', flush=True)
        print(f'  - BB crosslinking: {b_percent:.2f}% of target {percentage_ck_B*100:.2f}%', flush=True)
        print(f'  - Total crosslinking: {total_percent:.2f}%', flush=True)
        print(f'  - Relative to target: {relative_percent:.2f}%', flush=True)
        print(f'  - Bonds added this iteration: {bonds_added_this_iteration}', flush=True)
        print(f'  - Probability rejections: AC={rejected_AC_bonds}, B={rejected_B_bonds}', flush=True)

        # If we still haven't reached the target,
        # run annealing to try to reposition atoms
        if ((num_add_ck_A < capacity_AA_ideal/2 * percentage_ck_A_actual) or
                                               (num_add_ck_B < capacity_ck_B/2 * percentage_ck_B)):

            update_lammps_data(lmp_data, bond_info, lmp_command)

            # Read updated positions after annealing
            lmp_data = els.read_lammps_full('anneal.dat')
            lmp_data.atom_info = lmp_data.atom_info[np.argsort(lmp_data.atom_info[:, 0])]
            atom_types = lmp_data.atom_info[:, 2]
            box, coors = els.box_coors_from_lmp(lmp_data)

            # Update positions
            positions_A = coors[atom_types == 1]
            positions_B = coors[atom_types == 2]
            positions_C = coors[atom_types == 3]

            # Update combined positions
            all_AC_positions = np.vstack([positions_A, positions_C])

    # Print final statistics
    print("\nFinal Crosslinking Statistics:", flush=True)
    print(f"A-A/A-C/C-C Crosslinking: {num_add_ck_A} bonds added ({num_add_ck_A/(capacity_AA_ideal/2)*100:.2f}% of target)", flush=True)
    print(f"B-B Crosslinking: {num_add_ck_B} bonds added ({num_add_ck_B/(capacity_ck_B/2)*100:.2f}% of target)", flush=True)
    print(f"Total Crosslinking: {(num_add_ck_A + num_add_ck_B)/total_capacity*100:.2f}%", flush=True)
    print(f"Probability-based rejections: AC={rejected_AC_bonds}, B={rejected_B_bonds}", flush=True)

    if iteration >= max_iterations:
        print("WARNING: Maximum iterations reached before completing crosslinking", flush=True)

    # Save the final snapshot if it hasn't been saved yet and we're at 100%
    if save_snapshots and last_checkpoint_reached < 10 and total_bonds_formed / target_total_bonds >= 0.95:
        snapshot_filename = f"crosslink_100.dat"
        lmp_snapshot = els.lammps(
            natoms=lmp_data.natoms,
            nbonds=len(bond_info),
            natom_types=3,
            nbond_types=3,
            x=lmp_data.x, y=lmp_data.y, z=lmp_data.z,
            mass=lmp_data.mass,
            atom_info=lmp_data.atom_info,
            bond_info=bond_info
        )
        els.write_lammps_full(snapshot_filename, lmp_snapshot)
        print(f"Saved final 100% snapshot: {snapshot_filename}", flush=True)

    # Write updated structure with crosslinks to a file
    lmp_network = els.lammps(
        natoms=lmp_data.natoms,
        nbonds=len(bond_info),
        natom_types=3,
        nbond_types=3,
        x=lmp_data.x, y=lmp_data.y, z=lmp_data.z,
        mass=lmp_data.mass,
        atom_info=lmp_data.atom_info,
        bond_info=bond_info
    )
    els.write_lammps_full(output_file, lmp_network)

#############################################################################################################################################################################################################################

# SYS 2: Hybrid-monomer chains

def crosslink_sys2(relaxed_file, output_file, lmp_command, percentage_ck_A, percentage_ck_B,
              functionality_A, functionality_B, functionality_C, ck_dis_A, ck_dis_B,
              probability_AC=1.0, probability_B=1.0, save_snapshots=False):
    """
    Crosslinking function with probability-based bond formation for systems with pre-existing chain bonds

    Parameters:
    -----------
    relaxed_file : str
        Input relaxed data file
    output_file : str
        Output crosslinked data file
    lmp_command : str
        Command to run LAMMPS
    percentage_ck_A : float
        Target percentage for A-A, A-C, C-C crosslinking (of available functionality)
    percentage_ck_B : float
        Target percentage for B-B crosslinking (of available functionality)
    functionality_A : int
        Maximum number of bonds for A-type beads
    functionality_B : int
        Maximum number of bonds for B-type beads
    functionality_C : int
        Maximum number of bonds for C-type beads
    ck_dis_A : float
        Cutoff distance for A-A, A-C, C-C crosslinking
    ck_dis_B : float
        Cutoff distance for B-B crosslinking
    probability_AC : float
        Probability (0-1) of forming a bond between A and C atoms when all other criteria are met
    probability_B : float
        Probability (0-1) of forming a bond between B atoms when all other criteria are met
    save_snapshots : bool
        Whether to save snapshots at different crosslinking percentages (default: False)
    """
    import numpy as np

    lmp_data = els.read_lammps_full(relaxed_file)
    lmp_data.atom_info = lmp_data.atom_info[np.argsort(lmp_data.atom_info[:, 0])]
    atom_positions = lmp_data.atom_info[:, 4:7]  # Assuming positions are columns 4-7
    atom_types = lmp_data.atom_info[:, 2]  # Assuming atom types are in column 2

    bond_info = lmp_data.bond_info
    bond_index = bond_info[:,2:]-1
    box, coors = els.box_coors_from_lmp(lmp_data)

    positions_A = coors[atom_types == 1]
    idx_A = np.squeeze(np.argwhere(atom_types == 1))
    num_A = len(positions_A)

    positions_B = coors[atom_types == 2]
    idx_B = np.squeeze(np.argwhere(atom_types == 2))
    num_B = len(positions_B)

    positions_C = coors[atom_types == 3]
    idx_C = np.squeeze(np.argwhere(atom_types == 3))
    num_C = len(positions_C)

    # Initialize reaction counters for each A, B, and C bead
    # For A beads, count ALL existing bonds (including chain bonds)
    reaction_counter_A = np.zeros(len(positions_A))
    for i in range(len(idx_A)):
        reaction_counter_A[i] = np.sum(np.concatenate(bond_index)==idx_A[i])

    # Count how many type 1 bonds (chain bonds) exist for A beads
    chain_bonds_A = np.zeros(len(positions_A))
    for i in range(len(idx_A)):
        chain_bonds_A[i] = np.sum((bond_info[:,1]==1) &
                                  ((bond_info[:,2]-1==idx_A[i]) |
                                   (bond_info[:,3]-1==idx_A[i])))

    # Calculate remaining functionality for A beads based on all existing bonds
    remaining_functionality_A = functionality_A*num_A - np.sum(reaction_counter_A)

    # Calculate remaining functionality for C beads
    reaction_counter_C = np.zeros(len(positions_C))
    for i in range(len(idx_C)):
        reaction_counter_C[i] = np.sum(np.concatenate(bond_index)==idx_C[i])
    remaining_functionality_C = functionality_C*num_C - np.sum(reaction_counter_C)

    # Count existing crosslink bonds (type 2)
    existing_ck_A_bonds = np.sum(bond_info[:,1]==2)

    # Calculate the total theoretical capacity for A-A, A-C, C-C bonds
    # This is the total functionality minus the chain bonds (bond_type=1)
    chain_bonds = np.sum(bond_info[:,1]==1)
    total_functionality_AC = functionality_A*num_A + functionality_C*num_C
    theoretical_capacity_AC = (total_functionality_AC - chain_bonds) / 2  # Divide by 2 to avoid double-counting the bodns

    # Calculate remaining capacity after accounting for existing crosslinks
    remaining_capacity_AC = theoretical_capacity_AC - existing_ck_A_bonds

    # Calculate actual target number of bonds to form
    target_ck_A_bonds = percentage_ck_A * theoretical_capacity_AC - existing_ck_A_bonds
    if target_ck_A_bonds < 0:
        target_ck_A_bonds = 0

    print(f"A beads: total={num_A}, existing chain bonds={np.sum(chain_bonds_A)}")
    print(f"A beads: functionality={functionality_A}, remaining functionality={remaining_functionality_A}")
    print(f"C beads: total={num_C}, remaining functionality={remaining_functionality_C}")
    print(f"Total theoretical capacity for AC crosslinking={theoretical_capacity_AC}")
    print(f"Existing AC crosslinks={existing_ck_A_bonds}")
    print(f"Remaining available AC crosslink capacity={remaining_capacity_AC}")
    print(f"Target percentage={percentage_ck_A*100}%")
    print(f"Target AC crosslinks to add={target_ck_A_bonds}")

    reaction_counter_B = np.zeros(len(positions_B))
    for i in range(len(idx_B)):
        reaction_counter_B[i] = np.sum(np.concatenate(bond_index)==idx_B[i])

    # Count existing B chain bonds if any
    chain_bonds_B = np.zeros(len(positions_B))
    for i in range(len(idx_B)):
        chain_bonds_B[i] = np.sum((bond_info[:,1]==1) &
                                 ((bond_info[:,2]-1==idx_B[i]) |
                                  (bond_info[:,3]-1==idx_B[i])))

    # Calculate remaining functionality for B beads
    remaining_functionality_B = functionality_B * num_B - np.sum(reaction_counter_B)

    # Calculate B-B crosslink capacity and target
    existing_B_crosslinks = np.sum(bond_info[:,1]==3)
    total_functionality_B = functionality_B * num_B
    # Calculate theoretical capacity (accounting for chain bonds)
    theoretical_capacity_B = (total_functionality_B - np.sum(chain_bonds_B)) / 2

    # Calculate target B-B bonds to form
    target_ck_B_bonds = percentage_ck_B * theoretical_capacity_B - existing_B_crosslinks
    if target_ck_B_bonds < 0:
        target_ck_B_bonds = 0

    print(f"B beads: total={num_B}, existing chain bonds={np.sum(chain_bonds_B)}")
    print(f"B beads: functionality={functionality_B}, remaining functionality={remaining_functionality_B}")
    print(f"Total theoretical capacity for B crosslinking={theoretical_capacity_B}")
    print(f"Existing B crosslinks={existing_B_crosslinks}")
    print(f"Remaining available B crosslink capacity={theoretical_capacity_B - existing_B_crosslinks}")
    print(f"Target percentage={percentage_ck_B*100}%")
    print(f"Target B crosslinks to add={target_ck_B_bonds}")

    connected_molecules_AC = {}
    # Initialize for A beads
    for idx in idx_A:
        connected_molecules_AC[idx] = set()
    # Initialize for C beads
    for idx in idx_C:
        connected_molecules_AC[idx] = set()

    # Initialize for B beads (separate tracking)
    connected_molecules_B = {idx: set() for idx in idx_B}

    # Pre-populate the connected_molecules dictionaries from existing bonds
    for bond in bond_info:
        if bond[1] == 2:  # If it's an A-A, A-C, or C-C bond
            atom1, atom2 = int(bond[2])-1, int(bond[3])-1
            if atom1 in connected_molecules_AC and atom2 in connected_molecules_AC:
                connected_molecules_AC[atom1].add(atom2)
                connected_molecules_AC[atom2].add(atom1)
        elif bond[1] == 3:  # If it's a B-B bond
            atom1, atom2 = int(bond[2])-1, int(bond[3])-1
            if atom1 in connected_molecules_B and atom2 in connected_molecules_B:
                connected_molecules_B[atom1].add(atom2)
                connected_molecules_B[atom2].add(atom1)

    # Combine A and C bead information
    all_AC_positions = np.vstack([positions_A, positions_C])
    all_AC_indices = np.concatenate([idx_A, idx_C])
    all_AC_types = np.array(['A'] * len(positions_A) + ['C'] * len(positions_C))
    all_AC_functionality = np.concatenate([np.ones(len(positions_A)) * functionality_A,
                                         np.ones(len(positions_C)) * functionality_C])
    # Create a mapping between combined array indices and original indices
    combined_to_original = {}
    for i in range(len(positions_A)):
        combined_to_original[i] = ('A', i)
    for i in range(len(positions_C)):
        combined_to_original[i + len(positions_A)] = ('C', i)

    # Create mapping from combined index to actual atom index in the system
    combined_to_atom_idx = {}
    for i in range(len(positions_A)):
        combined_to_atom_idx[i] = idx_A[i]
    for i in range(len(positions_C)):
        combined_to_atom_idx[i + len(positions_A)] = idx_C[i]

    # Identify potential crosslinking pairs
    num_add_ck_A = 0  # Counter for all A-A, A-C, C-A, and C-C bonds
    num_add_ck_B = 0  # Counter for B-B bonds
    lmp_data.nbond_types = 3

    # Counters for tracking probability-based rejections
    rejected_AC_bonds = 0
    rejected_B_bonds = 0

    max_iterations = 1000  # Safety limit to prevent infinite loops
    iteration = 0

    # For snapshot saving
    target_total_bonds = target_ck_A_bonds + target_ck_B_bonds
    last_checkpoint_reached = -1

    # Initial snapshot at 0%
    if save_snapshots:
        last_checkpoint_reached = save_snapshot_if_needed(0, target_total_bonds, lmp_data, bond_info, last_checkpoint_reached)

    # Main crosslinking loop
    while ((target_ck_A_bonds > 0 and num_add_ck_A < target_ck_A_bonds) or
           (target_ck_B_bonds > 0 and num_add_ck_B < target_ck_B_bonds)) and iteration < max_iterations:
        iteration += 1
        bonds_added_this_iteration = 0

        # Initialize lists to store potential bonds
        potential_AC_bonds = []  # Will store tuples of (i, j, distances) for A-C bonds
        potential_B_bonds = []   # Will store tuples of (i, j, distances) for B-B bonds

        # Find all potential A-C bonds
        if target_ck_A_bonds > 0 and num_add_ck_A < target_ck_A_bonds:
            for i in range(len(all_AC_positions)):
                pos_i = all_AC_positions[i]
                type_i, orig_i = combined_to_original[i]
                atom_idx_i = combined_to_atom_idx[i]

                # Check if this bead has available functionality
                if (type_i == 'A' and reaction_counter_A[orig_i] >= functionality_A) or \
                   (type_i == 'C' and reaction_counter_C[orig_i] >= functionality_C):
                    continue

                # Find all potential partners (A or C beads) within distance
                j_list = np.argwhere((np.abs(all_AC_positions[:, 0] - pos_i[0]) < ck_dis_A) &
                                     (np.abs(all_AC_positions[:, 1] - pos_i[1]) < ck_dis_A) &
                                     (np.abs(all_AC_positions[:, 2] - pos_i[2]) < ck_dis_A)).flatten()

                for j in j_list:
                    if i == j:  # Skip self
                        continue

                    type_j, orig_j = combined_to_original[j]
                    atom_idx_j = combined_to_atom_idx[j]
                    pos_j = all_AC_positions[j]

                    # Check if partner has available functionality
                    if (type_j == 'A' and reaction_counter_A[orig_j] >= functionality_A) or \
                       (type_j == 'C' and reaction_counter_C[orig_j] >= functionality_C):
                        continue

                    # Check distance using PBC
                    distance = mc.pbc_distance(pos_i, pos_j, box)
                    if distance >= ck_dis_A:
                        continue

                    # Check if already connected - using the unified dictionary
                    if atom_idx_j in connected_molecules_AC[atom_idx_i]:
                        continue

                    # Add to potential bonds list - store all relevant information
                    potential_AC_bonds.append((
                        i, j, distance, type_i, orig_i, atom_idx_i, type_j, orig_j, atom_idx_j
                    ))

        # Find all potential B-B bonds
        if target_ck_B_bonds > 0 and num_add_ck_B < target_ck_B_bonds:
            for i, pos_B in enumerate(positions_B):
                if reaction_counter_B[i] < functionality_B:
                    j_list = np.argwhere((np.abs(positions_B[:, 0] - pos_B[0]) < ck_dis_B) &
                                         (np.abs(positions_B[:, 1] - pos_B[1]) < ck_dis_B) &
                                         (np.abs(positions_B[:, 2] - pos_B[2]) < ck_dis_B)).flatten()

                    for j in j_list:
                        if j == i:  # Skip self
                            continue

                        if reaction_counter_B[j] >= functionality_B:
                            continue

                        distance = mc.pbc_distance(pos_B, positions_B[j], box)
                        if distance >= ck_dis_B:
                            continue

                        if idx_B[j] in connected_molecules_B[idx_B[i]]:
                            continue

                        # Add to potential bonds list
                        potential_B_bonds.append((i, j, distance, idx_B[i], idx_B[j]))

        # Shuffle both lists of potential bonds
        np.random.shuffle(potential_AC_bonds)
        np.random.shuffle(potential_B_bonds)

        # Choose randomly between A-C and B-B bonds until no more potential bonds remain
        while potential_AC_bonds or potential_B_bonds:
            # Decide which type of bond to try next (if both are available)
            if potential_AC_bonds and potential_B_bonds:
                # 50/50 chance of picking A-C vs B-B
                if np.random.random() < 0.5:
                    bond_type = 'AC'
                else:
                    bond_type = 'B'
            elif potential_AC_bonds:
                bond_type = 'AC'
            else:
                bond_type = 'B'

            # Process the chosen bond type
            if bond_type == 'AC':
                # Get the next potential A-C bond
                bond_data = potential_AC_bonds.pop(0)
                i, j, distance, type_i, orig_i, atom_idx_i, type_j, orig_j, atom_idx_j = bond_data

                # Apply probability-based bond formation
                if np.random.random() > probability_AC:
                    rejected_AC_bonds += 1
                    continue

                # Form bond and update counters
                bond_info = np.vstack([bond_info, [len(bond_info) + 1, 2, atom_idx_j + 1, atom_idx_i + 1]])

                # Update reaction counters
                if type_i == 'A':
                    reaction_counter_A[orig_i] += 1
                else:  # type_i == 'C'
                    reaction_counter_C[orig_i] += 1

                if type_j == 'A':
                    reaction_counter_A[orig_j] += 1
                else:  # type_j == 'C'
                    reaction_counter_C[orig_j] += 1

                # Update connected molecules - using the unified dictionary
                connected_molecules_AC[atom_idx_i].add(atom_idx_j)
                connected_molecules_AC[atom_idx_j].add(atom_idx_i)

                # Increment the counter for all A and C bonds
                num_add_ck_A += 1
                bonds_added_this_iteration += 1

                # Filter out any potential bonds that involve these atoms that now have new connections
                potential_AC_bonds = [
                    bond for bond in potential_AC_bonds
                    if not (bond[5] == atom_idx_i or bond[5] == atom_idx_j or
                            bond[8] == atom_idx_i or bond[8] == atom_idx_j)
                ]

                # Check if we should save a snapshot after adding this bond
                if save_snapshots:
                    total_bonds_formed = num_add_ck_A + num_add_ck_B
                    last_checkpoint_reached = save_snapshot_if_needed(
                        total_bonds_formed, target_total_bonds, lmp_data, bond_info, last_checkpoint_reached
                    )

            else:  # bond_type == 'B'
                # Get the next potential B-B bond
                bond_data = potential_B_bonds.pop(0)
                i, j, distance, idx_B_i, idx_B_j = bond_data

                # Apply probability-based bond formation
                if np.random.random() > probability_B:
                    rejected_B_bonds += 1
                    continue

                # Form B-B bond
                bond_info = np.vstack([bond_info, [len(bond_info) + 1, 3, idx_B_j + 1, idx_B_i + 1]])
                reaction_counter_B[i] += 1
                reaction_counter_B[j] += 1
                num_add_ck_B += 1
                bonds_added_this_iteration += 1

                # Update connected molecules for B
                connected_molecules_B[idx_B_i].add(idx_B_j)
                connected_molecules_B[idx_B_j].add(idx_B_i)

                # Filter out any potential bonds that involve these atoms
                potential_B_bonds = [
                    bond for bond in potential_B_bonds
                    if not (bond[3] == idx_B_i or bond[3] == idx_B_j or
                            bond[4] == idx_B_i or bond[4] == idx_B_j)
                ]

                # Check if we should save a snapshot after adding this bond
                if save_snapshots:
                    total_bonds_formed = num_add_ck_A + num_add_ck_B
                    last_checkpoint_reached = save_snapshot_if_needed(
                        total_bonds_formed, target_total_bonds, lmp_data, bond_info, last_checkpoint_reached
                    )

        # Calculate current crosslinking percentages
        if target_ck_A_bonds > 0:
            ac_percent = num_add_ck_A / target_ck_A_bonds * 100
        else:
            ac_percent = 100.0  # If target is 0, we've already reached 100%

        if target_ck_B_bonds > 0:
            b_percent = num_add_ck_B / target_ck_B_bonds * 100
        else:
            b_percent = 100.0  # If target is 0, we've already reached 100%

        # Calculate total progress
        if target_total_bonds > 0:
            total_percent = (num_add_ck_A + num_add_ck_B) / target_total_bonds * 100
        else:
            total_percent = 100.0

        print(f'Iteration {iteration}:', flush=True)
        print(f'  - AA crosslinking: {ac_percent:.2f}% of target ({num_add_ck_A}/{target_ck_A_bonds:.0f} bonds)', flush=True)
        print(f'  - BB crosslinking: {b_percent:.2f}% of target ({num_add_ck_B}/{target_ck_B_bonds:.0f} bonds)', flush=True)
        print(f'  - Overall progress: {total_percent:.2f}%', flush=True)
        print(f'  - Bonds added this iteration: {bonds_added_this_iteration}', flush=True)
        print(f'  - Probability rejections: AC={rejected_AC_bonds}, B={rejected_B_bonds}', flush=True)

        # If no new bonds were added in this iteration and we still haven't reached the target,
        # run annealing to try to reposition atoms
        if ((target_ck_A_bonds > 0 and num_add_ck_A < target_ck_A_bonds) or
                                               (target_ck_B_bonds > 0 and num_add_ck_B < target_ck_B_bonds)):
            print("No new bonds added - running annealing step", flush=True)
            update_lammps_data(lmp_data, bond_info, lmp_command)

            # Read updated positions after annealing
            lmp_data = els.read_lammps_full('anneal.dat')
            lmp_data.atom_info = lmp_data.atom_info[np.argsort(lmp_data.atom_info[:, 0])]
            atom_types = lmp_data.atom_info[:, 2]
            box, coors = els.box_coors_from_lmp(lmp_data)

            # Update positions
            positions_A = coors[atom_types == 1]
            positions_B = coors[atom_types == 2]
            positions_C = coors[atom_types == 3]

            # Update combined positions
            all_AC_positions = np.vstack([positions_A, positions_C])

    # Print final statistics
    print("\nFinal Crosslinking Statistics:", flush=True)
    print(f"A-A/A-C/C-C Crosslinking:", flush=True)
    print(f"  - Theoretical capacity: {theoretical_capacity_AC} bonds", flush=True)
    print(f"  - Existing before process: {existing_ck_A_bonds} bonds", flush=True)
    print(f"  - Target to add: {target_ck_A_bonds} bonds", flush=True)
    print(f"  - Actually added: {num_add_ck_A} bonds ({num_add_ck_A/target_ck_A_bonds*100:.2f}% of target)" if target_ck_A_bonds > 0 else "  - Actually added: 0 bonds (100% of target)", flush=True)
    print(f"  - Total after process: {existing_ck_A_bonds + num_add_ck_A} bonds ({(existing_ck_A_bonds + num_add_ck_A)/theoretical_capacity_AC*100:.2f}% of total capacity)", flush=True)

    print(f"B-B Crosslinking:", flush=True)
    print(f"  - Theoretical capacity: {theoretical_capacity_B} bonds", flush=True)
    print(f"  - Existing before process: {existing_B_crosslinks} bonds", flush=True)
    print(f"  - Target to add: {target_ck_B_bonds} bonds", flush=True)
    print(f"  - Actually added: {num_add_ck_B} bonds ({num_add_ck_B/target_ck_B_bonds*100:.2f}% of target)" if target_ck_B_bonds > 0 else "  - Actually added: 0 bonds (100% of target)", flush=True)
    print(f"  - Total after process: {existing_B_crosslinks + num_add_ck_B} bonds ({(existing_B_crosslinks + num_add_ck_B)/theoretical_capacity_B*100:.2f}% of total capacity)", flush=True)

    print(f"Total Bonds Added: {num_add_ck_A + num_add_ck_B}", flush=True)
    print(f"Target Bonds: {target_total_bonds:.2f}", flush=True)
    print(f"Total Completion: {(num_add_ck_A + num_add_ck_B)/target_total_bonds*100:.2f}% (of target)" if target_total_bonds > 0 else "Total Completion: 100% (no new bonds needed)", flush=True)
    print(f"Probability-based rejections: AC={rejected_AC_bonds}, B={rejected_B_bonds}", flush=True)

    if iteration >= max_iterations:
        print("WARNING: Maximum iterations reached before completing crosslinking", flush=True)

    # Save the final snapshot if it hasn't been saved yet and we're at 100%
    if save_snapshots and last_checkpoint_reached < 10 and ((target_total_bonds > 0 and (num_add_ck_A + num_add_ck_B) / target_total_bonds >= 0.95) or target_total_bonds == 0):
        snapshot_filename = f"crosslink_100.dat"
        lmp_snapshot = els.lammps(
            natoms=lmp_data.natoms,
            nbonds=len(bond_info),
            natom_types=3,
            nbond_types=3,
            x=lmp_data.x, y=lmp_data.y, z=lmp_data.z,
            mass=lmp_data.mass,
            atom_info=lmp_data.atom_info,
            bond_info=bond_info
        )
        els.write_lammps_full(snapshot_filename, lmp_snapshot)
        print(f"Saved final 100% snapshot: {snapshot_filename}", flush=True)

    # Write updated structure with crosslinks to a file
    lmp_network = els.lammps(
        natoms=lmp_data.natoms,
        nbonds=len(bond_info),
        natom_types=3,
        nbond_types=3,
        x=lmp_data.x, y=lmp_data.y, z=lmp_data.z,
        mass=lmp_data.mass,
        atom_info=lmp_data.atom_info,
        bond_info=bond_info
    )
    els.write_lammps_full(output_file, lmp_network)

def crosslink_sys1_mod(relaxed_file, output_file, lmp_command, percentage_ck_A, percentage_ck_B,
              functionality_A, functionality_B, functionality_C, ck_dis_A, ck_dis_B,
              probability_AC, probability_B, save_snapshots=False):
    """
    Crosslinking function with probability-based bond formation
    MODIFIED VERSION: Checks target percentages after each individual bond formation

    Parameters:
    -----------
    relaxed_file : str
        Input relaxed data file
    output_file : str
        Output crosslinked data file
    lmp_command : str
        Command to run LAMMPS
    percentage_ck_A : float
        Target percentage for A-A, A-C, C-C crosslinking
    percentage_ck_B : float
        Target percentage for B-B crosslinking
    functionality_A : int
        Maximum number of bonds for A-type beads
    functionality_B : int
        Maximum number of bonds for B-type beads
    functionality_C : int
        Maximum number of bonds for C-type beads
    ck_dis_A : float
        Cutoff distance for A-A, A-C, C-C crosslinking
    ck_dis_B : float
        Cutoff distance for B-B crosslinking
    probability_AC : float
        Probability (0-1) of forming a bond between A and C atoms when all other criteria are met
    probability_B : float
        Probability (0-1) of forming a bond between B atoms when all other criteria are met
    save_snapshots : bool
        Whether to save snapshots at different crosslinking percentages (default: False)
    """

    lmp_data = els.read_lammps_full(relaxed_file)
    lmp_data.atom_info = lmp_data.atom_info[np.argsort(lmp_data.atom_info[:, 0])]
    atom_positions = lmp_data.atom_info[:, 4:7]  # Assuming positions are columns 4-7
    atom_types = lmp_data.atom_info[:, 2]  # Assuming atom types are in column 2

    bond_info = lmp_data.bond_info
    bond_index = bond_info[:,2:]-1
    box, coors = els.box_coors_from_lmp(lmp_data)

    positions_A = coors[atom_types == 1]
    idx_A = np.squeeze(np.argwhere(atom_types == 1))
    num_A = len(positions_A)

    positions_B = coors[atom_types == 2]
    idx_B = np.squeeze(np.argwhere(atom_types == 2))
    num_B = len(positions_B)

    positions_C = coors[atom_types == 3]
    idx_C = np.squeeze(np.argwhere(atom_types == 3))
    num_C = len(positions_C)

    # Initialize reaction counters for each A, B, and C bead
    reaction_counter_A = np.zeros(len(positions_A))
    for i in range(len(idx_A)):
        reaction_counter_A[i] = np.sum(np.concatenate(bond_index)==idx_A[i])
    capacity_AA_ideal = (functionality_A*num_A + functionality_C*num_C) - np.sum(reaction_counter_A)

    percentage_ck_A_actual = (percentage_ck_A*capacity_AA_ideal - np.sum(bond_info[:,1]==2))/capacity_AA_ideal
    print(f"AA max={capacity_AA_ideal}, already={np.sum(bond_info[:,1]==2)}, target = {percentage_ck_A*capacity_AA_ideal}")
    print(f"AA actual percentage = {percentage_ck_A_actual}")

    reaction_counter_C = np.zeros(len(positions_C))
    for i in range(len(idx_C)):
        reaction_counter_C[i] = np.sum(np.concatenate(bond_index)==idx_C[i])

    reaction_counter_B = np.zeros(len(positions_B))
    for i in range(len(idx_B)):
        reaction_counter_B[i] = np.sum(np.concatenate(bond_index)==idx_B[i])
    capacity_ck_B = functionality_B * num_B - np.sum(reaction_counter_B)

    print(f"BB max={capacity_ck_B}, target = {percentage_ck_B*capacity_ck_B}")

    # Create a unified connected molecules dictionary for A and C types
    connected_molecules_AC = {}
    # Initialize for A beads
    for idx in idx_A:
        connected_molecules_AC[idx] = set()
    # Initialize for C beads
    for idx in idx_C:
        connected_molecules_AC[idx] = set()

    # Initialize for B beads (separate tracking)
    connected_molecules_B = {idx: set() for idx in idx_B}

    # Pre-populate the connected_molecules dictionaries from existing bonds
    for bond in bond_info:
        if bond[1] == 2:  # If it's an A-A, A-C, or C-C bond
            atom1, atom2 = int(bond[2])-1, int(bond[3])-1
            if atom1 in connected_molecules_AC and atom2 in connected_molecules_AC:
                connected_molecules_AC[atom1].add(atom2)
                connected_molecules_AC[atom2].add(atom1)
        elif bond[1] == 3:  # If it's a B-B bond
            atom1, atom2 = int(bond[2])-1, int(bond[3])-1
            if atom1 in connected_molecules_B and atom2 in connected_molecules_B:
                connected_molecules_B[atom1].add(atom2)
                connected_molecules_B[atom2].add(atom1)

    # Combine A and C bead information
    all_AC_positions = np.vstack([positions_A, positions_C])
    all_AC_indices = np.concatenate([idx_A, idx_C])
    all_AC_types = np.array(['A'] * len(positions_A) + ['C'] * len(positions_C))
    all_AC_functionality = np.concatenate([np.ones(len(positions_A)) * functionality_A,
                                         np.ones(len(positions_C)) * functionality_C])
    # Create a mapping between combined array indices and original indices
    combined_to_original = {}
    for i in range(len(positions_A)):
        combined_to_original[i] = ('A', i)
    for i in range(len(positions_C)):
        combined_to_original[i + len(positions_A)] = ('C', i)

    # Create mapping from combined index to actual atom index in the system
    combined_to_atom_idx = {}
    for i in range(len(positions_A)):
        combined_to_atom_idx[i] = idx_A[i]
    for i in range(len(positions_C)):
        combined_to_atom_idx[i + len(positions_A)] = idx_C[i]

    # Identify potential crosslinking pairs
    num_add_ck_A = 0  # Counter for all A-A, A-C, C-A, and C-C bonds
    num_add_ck_B = 0  # Counter for B-B bonds
    lmp_data.nbond_types = 3

    # Counters for tracking probability-based rejections
    rejected_AC_bonds = 0
    rejected_B_bonds = 0

    max_iterations = 1000  # Safety limit to prevent infinite loops
    iteration = 0

    # For snapshot saving
    target_total_bonds = (capacity_AA_ideal/2 * percentage_ck_A_actual) + (capacity_ck_B/2 * percentage_ck_B)
    last_checkpoint_reached = -1

    # Initial snapshot at 0%
    if save_snapshots:
        last_checkpoint_reached = save_snapshot_if_needed(0, target_total_bonds, lmp_data, bond_info, last_checkpoint_reached)

    # Main crosslinking loop
    while ((num_add_ck_A < capacity_AA_ideal/2 * percentage_ck_A_actual) or
           (num_add_ck_B < capacity_ck_B/2 * percentage_ck_B)) and iteration < max_iterations:
        iteration += 1
        bonds_added_this_iteration = 0

        # Initialize lists to store potential bonds
        potential_AC_bonds = []  # Will store tuples of (i, j, distances) for A-C bonds
        potential_B_bonds = []   # Will store tuples of (i, j, distances) for B-B bonds

        # Find all potential A-C bonds
        if num_add_ck_A < capacity_AA_ideal/2 * percentage_ck_A_actual:
            for i in range(len(all_AC_positions)):
                pos_i = all_AC_positions[i]
                type_i, orig_i = combined_to_original[i]
                atom_idx_i = combined_to_atom_idx[i]

                # Check if this bead has available functionality
                if (type_i == 'A' and reaction_counter_A[orig_i] >= functionality_A) or \
                   (type_i == 'C' and reaction_counter_C[orig_i] >= functionality_C):
                    continue

                # Find all potential partners (A or C beads) within distance
                j_list = np.argwhere((np.abs(all_AC_positions[:, 0] - pos_i[0]) < ck_dis_A) &
                                     (np.abs(all_AC_positions[:, 1] - pos_i[1]) < ck_dis_A) &
                                     (np.abs(all_AC_positions[:, 2] - pos_i[2]) < ck_dis_A)).flatten()

                for j in j_list:
                    if i == j:  # Skip self
                        continue

                    type_j, orig_j = combined_to_original[j]
                    atom_idx_j = combined_to_atom_idx[j]
                    pos_j = all_AC_positions[j]

                    # Check if partner has available functionality
                    if (type_j == 'A' and reaction_counter_A[orig_j] >= functionality_A) or \
                       (type_j == 'C' and reaction_counter_C[orig_j] >= functionality_C):
                        continue

                    # Check distance using PBC
                    distance = mc.pbc_distance(pos_i, pos_j, box)
                    if distance >= ck_dis_A:
                        continue

                    # Check if already connected - using the unified dictionary
                    if atom_idx_j in connected_molecules_AC[atom_idx_i]:
                        continue

                    # Add to potential bonds list - store all relevant information
                    potential_AC_bonds.append((
                        i, j, distance, type_i, orig_i, atom_idx_i, type_j, orig_j, atom_idx_j
                    ))

        # Find all potential B-B bonds
        if num_add_ck_B < capacity_ck_B/2 * percentage_ck_B:
            for i, pos_B in enumerate(positions_B):
                if reaction_counter_B[i] < functionality_B:
                    j_list = np.argwhere((np.abs(positions_B[:, 0] - pos_B[0]) < ck_dis_B) &
                                         (np.abs(positions_B[:, 1] - pos_B[1]) < ck_dis_B) &
                                         (np.abs(positions_B[:, 2] - pos_B[2]) < ck_dis_B)).flatten()

                    for j in j_list:
                        if j == i:  # Skip self
                            continue

                        if reaction_counter_B[j] >= functionality_B:
                            continue

                        distance = mc.pbc_distance(pos_B, positions_B[j], box)
                        if distance >= ck_dis_B:
                            continue

                        if idx_B[j] in connected_molecules_B[idx_B[i]]:
                            continue

                        # Add to potential bonds list
                        potential_B_bonds.append((i, j, distance, idx_B[i], idx_B[j]))

        # Shuffle both lists of potential bonds
        np.random.shuffle(potential_AC_bonds)
        np.random.shuffle(potential_B_bonds)

        # Choose randomly between A-C and B-B bonds until no more potential bonds remain
        # OR until both targets are reached
        while (potential_AC_bonds or potential_B_bonds) and \
              ((num_add_ck_A < capacity_AA_ideal/2 * percentage_ck_A_actual) or
               (num_add_ck_B < capacity_ck_B/2 * percentage_ck_B)):

            # Decide which type of bond to try next (if both are available)
            if potential_AC_bonds and potential_B_bonds:
                # Only try AC bonds if we haven't reached the AC target
                # Only try B bonds if we haven't reached the B target
                ac_available = num_add_ck_A < capacity_AA_ideal/2 * percentage_ck_A_actual
                b_available = num_add_ck_B < capacity_ck_B/2 * percentage_ck_B

                if ac_available and b_available:
                    # 50/50 chance of picking A-C vs B-B
                    if np.random.random() < 0.5:
                        bond_type = 'AC'
                    else:
                        bond_type = 'B'
                elif ac_available:
                    bond_type = 'AC'
                elif b_available:
                    bond_type = 'B'
                else:
                    break  # Both targets reached
            elif potential_AC_bonds and num_add_ck_A < capacity_AA_ideal/2 * percentage_ck_A_actual:
                bond_type = 'AC'
            elif potential_B_bonds and num_add_ck_B < capacity_ck_B/2 * percentage_ck_B:
                bond_type = 'B'
            else:
                break  # No more valid bonds or targets reached

            # Process the chosen bond type
            if bond_type == 'AC':
                # Get the next potential A-C bond
                bond_data = potential_AC_bonds.pop(0)
                i, j, distance, type_i, orig_i, atom_idx_i, type_j, orig_j, atom_idx_j = bond_data

                # Apply probability-based bond formation
                if np.random.random() > probability_AC:
                    rejected_AC_bonds += 1
                    continue

                # Form bond and update counters
                bond_info = np.vstack([bond_info, [len(bond_info) + 1, 2, atom_idx_j + 1, atom_idx_i + 1]])

                # Update reaction counters
                if type_i == 'A':
                    reaction_counter_A[orig_i] += 1
                else:  # type_i == 'C'
                    reaction_counter_C[orig_i] += 1

                if type_j == 'A':
                    reaction_counter_A[orig_j] += 1
                else:  # type_j == 'C'
                    reaction_counter_C[orig_j] += 1

                # Update connected molecules - using the unified dictionary
                connected_molecules_AC[atom_idx_i].add(atom_idx_j)
                connected_molecules_AC[atom_idx_j].add(atom_idx_i)

                # Increment the counter for all A and C bonds
                num_add_ck_A += 1
                bonds_added_this_iteration += 1

                # Filter out any potential bonds that involve these atoms that now have new connections
                potential_AC_bonds = [
                    bond for bond in potential_AC_bonds
                    if not (bond[5] == atom_idx_i or bond[5] == atom_idx_j or
                            bond[8] == atom_idx_i or bond[8] == atom_idx_j)
                ]

                # Check if we should save a snapshot after adding this bond
                if save_snapshots:
                    total_bonds_formed = num_add_ck_A + num_add_ck_B
                    last_checkpoint_reached = save_snapshot_if_needed(
                        total_bonds_formed, target_total_bonds, lmp_data, bond_info, last_checkpoint_reached
                    )

            else:  # bond_type == 'B'
                # Get the next potential B-B bond
                bond_data = potential_B_bonds.pop(0)
                i, j, distance, idx_B_i, idx_B_j = bond_data

                # Apply probability-based bond formation
                if np.random.random() > probability_B:
                    rejected_B_bonds += 1
                    continue

                # Form B-B bond
                bond_info = np.vstack([bond_info, [len(bond_info) + 1, 3, idx_B_j + 1, idx_B_i + 1]])
                reaction_counter_B[i] += 1
                reaction_counter_B[j] += 1
                num_add_ck_B += 1
                bonds_added_this_iteration += 1

                # Update connected molecules for B
                connected_molecules_B[idx_B_i].add(idx_B_j)
                connected_molecules_B[idx_B_j].add(idx_B_i)

                # Filter out any potential bonds that involve these atoms
                potential_B_bonds = [
                    bond for bond in potential_B_bonds
                    if not (bond[3] == idx_B_i or bond[3] == idx_B_j or
                            bond[4] == idx_B_i or bond[4] == idx_B_j)
                ]

                # Check if we should save a snapshot after adding this bond
                if save_snapshots:
                    total_bonds_formed = num_add_ck_A + num_add_ck_B
                    last_checkpoint_reached = save_snapshot_if_needed(
                        total_bonds_formed, target_total_bonds, lmp_data, bond_info, last_checkpoint_reached
                    )

        # Calculate current crosslinking percentages
        ac_percent = num_add_ck_A/(capacity_AA_ideal/2) * 100
        b_percent = num_add_ck_B/(capacity_ck_B/2) * 100

        # Calculate total crosslinking percentage based on both types
        # Weighted average based on capacities
        total_capacity = capacity_AA_ideal/2 + capacity_ck_B/2
        total_bonds_formed = num_add_ck_A + num_add_ck_B
        total_percent = total_bonds_formed / total_capacity * 100
        relative_percent = total_bonds_formed / target_total_bonds * 100

        print(f'Iteration {iteration}:', flush=True)
        print(f'  - AA crosslinking: {ac_percent:.2f}% of target {percentage_ck_A_actual*100:.2f}%', flush=True)
        print(f'  - BB crosslinking: {b_percent:.2f}% of target {percentage_ck_B*100:.2f}%', flush=True)
        print(f'  - Total crosslinking: {total_percent:.2f}%', flush=True)
        print(f'  - Relative to target: {relative_percent:.2f}%', flush=True)
        print(f'  - Bonds added this iteration: {bonds_added_this_iteration}', flush=True)
        print(f'  - Probability rejections: AC={rejected_AC_bonds}, B={rejected_B_bonds}', flush=True)

        # If we still haven't reached the target,
        # run annealing to try to reposition atoms
        if ((num_add_ck_A < capacity_AA_ideal/2 * percentage_ck_A_actual) or
                                               (num_add_ck_B < capacity_ck_B/2 * percentage_ck_B)):

            update_lammps_data(lmp_data, bond_info, lmp_command)

            # Read updated positions after annealing
            lmp_data = els.read_lammps_full('anneal.dat')
            lmp_data.atom_info = lmp_data.atom_info[np.argsort(lmp_data.atom_info[:, 0])]
            atom_types = lmp_data.atom_info[:, 2]
            box, coors = els.box_coors_from_lmp(lmp_data)

            # Update positions
            positions_A = coors[atom_types == 1]
            positions_B = coors[atom_types == 2]
            positions_C = coors[atom_types == 3]

            # Update combined positions
            all_AC_positions = np.vstack([positions_A, positions_C])

    # Print final statistics
    print("\nFinal Crosslinking Statistics:", flush=True)
    print(f"A-A/A-C/C-C Crosslinking: {num_add_ck_A} bonds added ({num_add_ck_A/(capacity_AA_ideal/2)*100:.2f}% of target)", flush=True)
    print(f"B-B Crosslinking: {num_add_ck_B} bonds added ({num_add_ck_B/(capacity_ck_B/2)*100:.2f}% of target)", flush=True)
    print(f"Total Crosslinking: {(num_add_ck_A + num_add_ck_B)/total_capacity*100:.2f}%", flush=True)
    print(f"Probability-based rejections: AC={rejected_AC_bonds}, B={rejected_B_bonds}", flush=True)

    if iteration >= max_iterations:
        print("WARNING: Maximum iterations reached before completing crosslinking", flush=True)

    # Save the final snapshot if it hasn't been saved yet and we're at 100%
    if save_snapshots and last_checkpoint_reached < 10 and total_bonds_formed / target_total_bonds >= 0.95:
        snapshot_filename = f"crosslink_100.dat"
        lmp_snapshot = els.lammps(
            natoms=lmp_data.natoms,
            nbonds=len(bond_info),
            natom_types=3,
            nbond_types=3,
            x=lmp_data.x, y=lmp_data.y, z=lmp_data.z,
            mass=lmp_data.mass,
            atom_info=lmp_data.atom_info,
            bond_info=bond_info
        )
        els.write_lammps_full(snapshot_filename, lmp_snapshot)
        print(f"Saved final 100% snapshot: {snapshot_filename}", flush=True)

    # Write updated structure with crosslinks to a file
    lmp_network = els.lammps(
        natoms=lmp_data.natoms,
        nbonds=len(bond_info),
        natom_types=3,
        nbond_types=3,
        x=lmp_data.x, y=lmp_data.y, z=lmp_data.z,
        mass=lmp_data.mass,
        atom_info=lmp_data.atom_info,
        bond_info=bond_info
    )
    els.write_lammps_full(output_file, lmp_network)


def crosslink_sys2_mod(relaxed_file, output_file, lmp_command, percentage_ck_A, percentage_ck_B,
              functionality_A, functionality_B, functionality_C, ck_dis_A, ck_dis_B,
              probability_AC=1.0, probability_B=1.0, save_snapshots=False):
    """
    Crosslinking function with probability-based bond formation for systems with pre-existing chain bonds
    MODIFIED VERSION: Checks target percentages after each individual bond formation

    Parameters:
    -----------
    relaxed_file : str
        Input relaxed data file
    output_file : str
        Output crosslinked data file
    lmp_command : str
        Command to run LAMMPS
    percentage_ck_A : float
        Target percentage for A-A, A-C, C-C crosslinking (of available functionality)
    percentage_ck_B : float
        Target percentage for B-B crosslinking (of available functionality)
    functionality_A : int
        Maximum number of bonds for A-type beads
    functionality_B : int
        Maximum number of bonds for B-type beads
    functionality_C : int
        Maximum number of bonds for C-type beads
    ck_dis_A : float
        Cutoff distance for A-A, A-C, C-C crosslinking
    ck_dis_B : float
        Cutoff distance for B-B crosslinking
    probability_AC : float
        Probability (0-1) of forming a bond between A and C atoms when all other criteria are met
    probability_B : float
        Probability (0-1) of forming a bond between B atoms when all other criteria are met
    save_snapshots : bool
        Whether to save snapshots at different crosslinking percentages (default: False)
    """
    import numpy as np

    lmp_data = els.read_lammps_full(relaxed_file)
    lmp_data.atom_info = lmp_data.atom_info[np.argsort(lmp_data.atom_info[:, 0])]
    atom_positions = lmp_data.atom_info[:, 4:7]  # Assuming positions are columns 4-7
    atom_types = lmp_data.atom_info[:, 2]  # Assuming atom types are in column 2

    bond_info = lmp_data.bond_info
    bond_index = bond_info[:,2:]-1
    box, coors = els.box_coors_from_lmp(lmp_data)

    positions_A = coors[atom_types == 1]
    idx_A = np.squeeze(np.argwhere(atom_types == 1))
    num_A = len(positions_A)

    positions_B = coors[atom_types == 2]
    idx_B = np.squeeze(np.argwhere(atom_types == 2))
    num_B = len(positions_B)

    positions_C = coors[atom_types == 3]
    idx_C = np.squeeze(np.argwhere(atom_types == 3))
    num_C = len(positions_C)

    # Initialize reaction counters for each A, B, and C bead
    # For A beads, count ALL existing bonds (including chain bonds)
    reaction_counter_A = np.zeros(len(positions_A))
    for i in range(len(idx_A)):
        reaction_counter_A[i] = np.sum(np.concatenate(bond_index)==idx_A[i])

    # Count how many type 1 bonds (chain bonds) exist for A beads
    chain_bonds_A = np.zeros(len(positions_A))
    for i in range(len(idx_A)):
        chain_bonds_A[i] = np.sum((bond_info[:,1]==1) &
                                  ((bond_info[:,2]-1==idx_A[i]) |
                                   (bond_info[:,3]-1==idx_A[i])))

    # Calculate remaining functionality for A beads based on all existing bonds
    remaining_functionality_A = functionality_A*num_A - np.sum(reaction_counter_A)

    # Calculate remaining functionality for C beads
    reaction_counter_C = np.zeros(len(positions_C))
    for i in range(len(idx_C)):
        reaction_counter_C[i] = np.sum(np.concatenate(bond_index)==idx_C[i])
    remaining_functionality_C = functionality_C*num_C - np.sum(reaction_counter_C)

    # Count existing crosslink bonds (type 2)
    existing_ck_A_bonds = np.sum(bond_info[:,1]==2)

    # Calculate the total theoretical capacity for A-A, A-C, C-C bonds
    # This is the total functionality minus the chain bonds (bond_type=1)
    chain_bonds = np.sum(bond_info[:,1]==1)
    total_functionality_AC = functionality_A*num_A + functionality_C*num_C
    theoretical_capacity_AC = (total_functionality_AC - chain_bonds) / 2  # Divide by 2 to avoid double-counting the bodns

    # Calculate remaining capacity after accounting for existing crosslinks
    remaining_capacity_AC = theoretical_capacity_AC - existing_ck_A_bonds

    # Calculate actual target number of bonds to form
    target_ck_A_bonds = percentage_ck_A * theoretical_capacity_AC - existing_ck_A_bonds
    if target_ck_A_bonds < 0:
        target_ck_A_bonds = 0

    print(f"A beads: total={num_A}, existing chain bonds={np.sum(chain_bonds_A)}")
    print(f"A beads: functionality={functionality_A}, remaining functionality={remaining_functionality_A}")
    print(f"C beads: total={num_C}, remaining functionality={remaining_functionality_C}")
    print(f"Total theoretical capacity for AC crosslinking={theoretical_capacity_AC}")
    print(f"Existing AC crosslinks={existing_ck_A_bonds}")
    print(f"Remaining available AC crosslink capacity={remaining_capacity_AC}")
    print(f"Target percentage={percentage_ck_A*100}%")
    print(f"Target AC crosslinks to add={target_ck_A_bonds}")

    reaction_counter_B = np.zeros(len(positions_B))
    for i in range(len(idx_B)):
        reaction_counter_B[i] = np.sum(np.concatenate(bond_index)==idx_B[i])

    # Count existing B chain bonds if any
    chain_bonds_B = np.zeros(len(positions_B))
    for i in range(len(idx_B)):
        chain_bonds_B[i] = np.sum((bond_info[:,1]==1) &
                                 ((bond_info[:,2]-1==idx_B[i]) |
                                  (bond_info[:,3]-1==idx_B[i])))

    # Calculate remaining functionality for B beads
    remaining_functionality_B = functionality_B * num_B - np.sum(reaction_counter_B)

    # Calculate B-B crosslink capacity and target
    existing_B_crosslinks = np.sum(bond_info[:,1]==3)
    total_functionality_B = functionality_B * num_B
    # Calculate theoretical capacity (accounting for chain bonds)
    theoretical_capacity_B = (total_functionality_B - np.sum(chain_bonds_B)) / 2

    # Calculate target B-B bonds to form
    target_ck_B_bonds = percentage_ck_B * theoretical_capacity_B - existing_B_crosslinks
    if target_ck_B_bonds < 0:
        target_ck_B_bonds = 0

    print(f"B beads: total={num_B}, existing chain bonds={np.sum(chain_bonds_B)}")
    print(f"B beads: functionality={functionality_B}, remaining functionality={remaining_functionality_B}")
    print(f"Total theoretical capacity for B crosslinking={theoretical_capacity_B}")
    print(f"Existing B crosslinks={existing_B_crosslinks}")
    print(f"Remaining available B crosslink capacity={theoretical_capacity_B - existing_B_crosslinks}")
    print(f"Target percentage={percentage_ck_B*100}%")
    print(f"Target B crosslinks to add={target_ck_B_bonds}")

    connected_molecules_AC = {}
    # Initialize for A beads
    for idx in idx_A:
        connected_molecules_AC[idx] = set()
    # Initialize for C beads
    for idx in idx_C:
        connected_molecules_AC[idx] = set()

    # Initialize for B beads (separate tracking)
    connected_molecules_B = {idx: set() for idx in idx_B}

    # Pre-populate the connected_molecules dictionaries from existing bonds
    for bond in bond_info:
        if bond[1] == 2:  # If it's an A-A, A-C, or C-C bond
            atom1, atom2 = int(bond[2])-1, int(bond[3])-1
            if atom1 in connected_molecules_AC and atom2 in connected_molecules_AC:
                connected_molecules_AC[atom1].add(atom2)
                connected_molecules_AC[atom2].add(atom1)
        elif bond[1] == 3:  # If it's a B-B bond
            atom1, atom2 = int(bond[2])-1, int(bond[3])-1
            if atom1 in connected_molecules_B and atom2 in connected_molecules_B:
                connected_molecules_B[atom1].add(atom2)
                connected_molecules_B[atom2].add(atom1)

    # Combine A and C bead information
    all_AC_positions = np.vstack([positions_A, positions_C])
    all_AC_indices = np.concatenate([idx_A, idx_C])
    all_AC_types = np.array(['A'] * len(positions_A) + ['C'] * len(positions_C))
    all_AC_functionality = np.concatenate([np.ones(len(positions_A)) * functionality_A,
                                         np.ones(len(positions_C)) * functionality_C])
    # Create a mapping between combined array indices and original indices
    combined_to_original = {}
    for i in range(len(positions_A)):
        combined_to_original[i] = ('A', i)
    for i in range(len(positions_C)):
        combined_to_original[i + len(positions_A)] = ('C', i)

    # Create mapping from combined index to actual atom index in the system
    combined_to_atom_idx = {}
    for i in range(len(positions_A)):
        combined_to_atom_idx[i] = idx_A[i]
    for i in range(len(positions_C)):
        combined_to_atom_idx[i + len(positions_A)] = idx_C[i]

    # Identify potential crosslinking pairs
    num_add_ck_A = 0  # Counter for all A-A, A-C, C-A, and C-C bonds
    num_add_ck_B = 0  # Counter for B-B bonds
    lmp_data.nbond_types = 3

    # Counters for tracking probability-based rejections
    rejected_AC_bonds = 0
    rejected_B_bonds = 0

    max_iterations = 1000  # Safety limit to prevent infinite loops
    iteration = 0

    # For snapshot saving
    target_total_bonds = target_ck_A_bonds + target_ck_B_bonds
    last_checkpoint_reached = -1

    # Initial snapshot at 0%
    if save_snapshots:
        last_checkpoint_reached = save_snapshot_if_needed(0, target_total_bonds, lmp_data, bond_info, last_checkpoint_reached)

    # Main crosslinking loop
    while ((target_ck_A_bonds > 0 and num_add_ck_A < target_ck_A_bonds) or
           (target_ck_B_bonds > 0 and num_add_ck_B < target_ck_B_bonds)) and iteration < max_iterations:
        iteration += 1
        bonds_added_this_iteration = 0

        # Initialize lists to store potential bonds
        potential_AC_bonds = []  # Will store tuples of (i, j, distances) for A-C bonds
        potential_B_bonds = []   # Will store tuples of (i, j, distances) for B-B bonds

        # Find all potential A-C bonds
        if target_ck_A_bonds > 0 and num_add_ck_A < target_ck_A_bonds:
            for i in range(len(all_AC_positions)):
                pos_i = all_AC_positions[i]
                type_i, orig_i = combined_to_original[i]
                atom_idx_i = combined_to_atom_idx[i]

                # Check if this bead has available functionality
                if (type_i == 'A' and reaction_counter_A[orig_i] >= functionality_A) or \
                   (type_i == 'C' and reaction_counter_C[orig_i] >= functionality_C):
                    continue

                # Find all potential partners (A or C beads) within distance
                j_list = np.argwhere((np.abs(all_AC_positions[:, 0] - pos_i[0]) < ck_dis_A) &
                                     (np.abs(all_AC_positions[:, 1] - pos_i[1]) < ck_dis_A) &
                                     (np.abs(all_AC_positions[:, 2] - pos_i[2]) < ck_dis_A)).flatten()

                for j in j_list:
                    if i == j:  # Skip self
                        continue

                    type_j, orig_j = combined_to_original[j]
                    atom_idx_j = combined_to_atom_idx[j]
                    pos_j = all_AC_positions[j]

                    # Check if partner has available functionality
                    if (type_j == 'A' and reaction_counter_A[orig_j] >= functionality_A) or \
                       (type_j == 'C' and reaction_counter_C[orig_j] >= functionality_C):
                        continue

                    # Check distance using PBC
                    distance = mc.pbc_distance(pos_i, pos_j, box)
                    if distance >= ck_dis_A:
                        continue

                    # Check if already connected - using the unified dictionary
                    if atom_idx_j in connected_molecules_AC[atom_idx_i]:
                        continue

                    # Add to potential bonds list - store all relevant information
                    potential_AC_bonds.append((
                        i, j, distance, type_i, orig_i, atom_idx_i, type_j, orig_j, atom_idx_j
                    ))

        # Find all potential B-B bonds
        if target_ck_B_bonds > 0 and num_add_ck_B < target_ck_B_bonds:
            for i, pos_B in enumerate(positions_B):
                if reaction_counter_B[i] < functionality_B:
                    j_list = np.argwhere((np.abs(positions_B[:, 0] - pos_B[0]) < ck_dis_B) &
                                         (np.abs(positions_B[:, 1] - pos_B[1]) < ck_dis_B) &
                                         (np.abs(positions_B[:, 2] - pos_B[2]) < ck_dis_B)).flatten()

                    for j in j_list:
                        if j == i:  # Skip self
                            continue

                        if reaction_counter_B[j] >= functionality_B:
                            continue

                        distance = mc.pbc_distance(pos_B, positions_B[j], box)
                        if distance >= ck_dis_B:
                            continue

                        if idx_B[j] in connected_molecules_B[idx_B[i]]:
                            continue

                        # Add to potential bonds list
                        potential_B_bonds.append((i, j, distance, idx_B[i], idx_B[j]))

        # Shuffle both lists of potential bonds
        np.random.shuffle(potential_AC_bonds)
        np.random.shuffle(potential_B_bonds)

        # Choose randomly between A-C and B-B bonds until no more potential bonds remain
        # OR until both targets are reached
        while (potential_AC_bonds or potential_B_bonds) and \
              ((target_ck_A_bonds > 0 and num_add_ck_A < target_ck_A_bonds) or
               (target_ck_B_bonds > 0 and num_add_ck_B < target_ck_B_bonds)):

            # Decide which type of bond to try next (if both are available)
            if potential_AC_bonds and potential_B_bonds:
                # Only try AC bonds if we haven't reached the AC target
                # Only try B bonds if we haven't reached the B target
                ac_available = target_ck_A_bonds > 0 and num_add_ck_A < target_ck_A_bonds
                b_available = target_ck_B_bonds > 0 and num_add_ck_B < target_ck_B_bonds

                if ac_available and b_available:
                    # 50/50 chance of picking A-C vs B-B
                    if np.random.random() < 0.5:
                        bond_type = 'AC'
                    else:
                        bond_type = 'B'
                elif ac_available:
                    bond_type = 'AC'
                elif b_available:
                    bond_type = 'B'
                else:
                    break  # Both targets reached
            elif potential_AC_bonds and target_ck_A_bonds > 0 and num_add_ck_A < target_ck_A_bonds:
                bond_type = 'AC'
            elif potential_B_bonds and target_ck_B_bonds > 0 and num_add_ck_B < target_ck_B_bonds:
                bond_type = 'B'
            else:
                break  # No more valid bonds or targets reached

            # Process the chosen bond type
            if bond_type == 'AC':
                # Get the next potential A-C bond
                bond_data = potential_AC_bonds.pop(0)
                i, j, distance, type_i, orig_i, atom_idx_i, type_j, orig_j, atom_idx_j = bond_data

                # Apply probability-based bond formation
                if np.random.random() > probability_AC:
                    rejected_AC_bonds += 1
                    continue

                # Form bond and update counters
                bond_info = np.vstack([bond_info, [len(bond_info) + 1, 2, atom_idx_j + 1, atom_idx_i + 1]])

                # Update reaction counters
                if type_i == 'A':
                    reaction_counter_A[orig_i] += 1
                else:  # type_i == 'C'
                    reaction_counter_C[orig_i] += 1

                if type_j == 'A':
                    reaction_counter_A[orig_j] += 1
                else:  # type_j == 'C'
                    reaction_counter_C[orig_j] += 1

                # Update connected molecules - using the unified dictionary
                connected_molecules_AC[atom_idx_i].add(atom_idx_j)
                connected_molecules_AC[atom_idx_j].add(atom_idx_i)

                # Increment the counter for all A and C bonds
                num_add_ck_A += 1
                bonds_added_this_iteration += 1

                # Filter out any potential bonds that involve these atoms that now have new connections
                potential_AC_bonds = [
                    bond for bond in potential_AC_bonds
                    if not (bond[5] == atom_idx_i or bond[5] == atom_idx_j or
                            bond[8] == atom_idx_i or bond[8] == atom_idx_j)
                ]

                # Check if we should save a snapshot after adding this bond
                if save_snapshots:
                    total_bonds_formed = num_add_ck_A + num_add_ck_B
                    last_checkpoint_reached = save_snapshot_if_needed(
                        total_bonds_formed, target_total_bonds, lmp_data, bond_info, last_checkpoint_reached
                    )

            else:  # bond_type == 'B'
                # Get the next potential B-B bond
                bond_data = potential_B_bonds.pop(0)
                i, j, distance, idx_B_i, idx_B_j = bond_data

                # Apply probability-based bond formation
                if np.random.random() > probability_B:
                    rejected_B_bonds += 1
                    continue

                # Form B-B bond
                bond_info = np.vstack([bond_info, [len(bond_info) + 1, 3, idx_B_j + 1, idx_B_i + 1]])
                reaction_counter_B[i] += 1
                reaction_counter_B[j] += 1
                num_add_ck_B += 1
                bonds_added_this_iteration += 1

                # Update connected molecules for B
                connected_molecules_B[idx_B_i].add(idx_B_j)
                connected_molecules_B[idx_B_j].add(idx_B_i)

                # Filter out any potential bonds that involve these atoms
                potential_B_bonds = [
                    bond for bond in potential_B_bonds
                    if not (bond[3] == idx_B_i or bond[3] == idx_B_j or
                            bond[4] == idx_B_i or bond[4] == idx_B_j)
                ]

                # Check if we should save a snapshot after adding this bond
                if save_snapshots:
                    total_bonds_formed = num_add_ck_A + num_add_ck_B
                    last_checkpoint_reached = save_snapshot_if_needed(
                        total_bonds_formed, target_total_bonds, lmp_data, bond_info, last_checkpoint_reached
                    )

        # Calculate current crosslinking percentages
        if target_ck_A_bonds > 0:
            ac_percent = num_add_ck_A / target_ck_A_bonds * 100
        else:
            ac_percent = 100.0  # If target is 0, we've already reached 100%

        if target_ck_B_bonds > 0:
            b_percent = num_add_ck_B / target_ck_B_bonds * 100
        else:
            b_percent = 100.0  # If target is 0, we've already reached 100%

        # Calculate total progress
        if target_total_bonds > 0:
            total_percent = (num_add_ck_A + num_add_ck_B) / target_total_bonds * 100
        else:
            total_percent = 100.0

        print(f'Iteration {iteration}:', flush=True)
        print(f'  - AA crosslinking: {ac_percent:.2f}% of target ({num_add_ck_A}/{target_ck_A_bonds:.0f} bonds)', flush=True)
        print(f'  - BB crosslinking: {b_percent:.2f}% of target ({num_add_ck_B}/{target_ck_B_bonds:.0f} bonds)', flush=True)
        print(f'  - Overall progress: {total_percent:.2f}%', flush=True)
        print(f'  - Bonds added this iteration: {bonds_added_this_iteration}', flush=True)
        print(f'  - Probability rejections: AC={rejected_AC_bonds}, B={rejected_B_bonds}', flush=True)

        # If no new bonds were added in this iteration and we still haven't reached the target,
        # run annealing to try to reposition atoms
        if ((target_ck_A_bonds > 0 and num_add_ck_A < target_ck_A_bonds) or
                                               (target_ck_B_bonds > 0 and num_add_ck_B < target_ck_B_bonds)):
            print("No new bonds added - running annealing step", flush=True)
            update_lammps_data(lmp_data, bond_info, lmp_command)

            # Read updated positions after annealing
            lmp_data = els.read_lammps_full('anneal.dat')
            lmp_data.atom_info = lmp_data.atom_info[np.argsort(lmp_data.atom_info[:, 0])]
            atom_types = lmp_data.atom_info[:, 2]
            box, coors = els.box_coors_from_lmp(lmp_data)

            # Update positions
            positions_A = coors[atom_types == 1]
            positions_B = coors[atom_types == 2]
            positions_C = coors[atom_types == 3]

            # Update combined positions
            all_AC_positions = np.vstack([positions_A, positions_C])

    # Print final statistics
    print("\nFinal Crosslinking Statistics:", flush=True)
    print(f"A-A/A-C/C-C Crosslinking:", flush=True)
    print(f"  - Theoretical capacity: {theoretical_capacity_AC} bonds", flush=True)
    print(f"  - Existing before process: {existing_ck_A_bonds} bonds", flush=True)
    print(f"  - Target to add: {target_ck_A_bonds} bonds", flush=True)
    print(f"  - Actually added: {num_add_ck_A} bonds ({num_add_ck_A/target_ck_A_bonds*100:.2f}% of target)" if target_ck_A_bonds > 0 else "  - Actually added: 0 bonds (100% of target)", flush=True)
    print(f"  - Total after process: {existing_ck_A_bonds + num_add_ck_A} bonds ({(existing_ck_A_bonds + num_add_ck_A)/theoretical_capacity_AC*100:.2f}% of total capacity)", flush=True)

    print(f"B-B Crosslinking:", flush=True)
    print(f"  - Theoretical capacity: {theoretical_capacity_B} bonds", flush=True)
    print(f"  - Existing before process: {existing_B_crosslinks} bonds", flush=True)
    print(f"  - Target to add: {target_ck_B_bonds} bonds", flush=True)
    print(f"  - Actually added: {num_add_ck_B} bonds ({num_add_ck_B/target_ck_B_bonds*100:.2f}% of target)" if target_ck_B_bonds > 0 else "  - Actually added: 0 bonds (100% of target)", flush=True)
    print(f"  - Total after process: {existing_B_crosslinks + num_add_ck_B} bonds ({(existing_B_crosslinks + num_add_ck_B)/theoretical_capacity_B*100:.2f}% of total capacity)", flush=True)

    print(f"Total Bonds Added: {num_add_ck_A + num_add_ck_B}", flush=True)
    print(f"Target Bonds: {target_total_bonds:.2f}", flush=True)
    print(f"Total Completion: {(num_add_ck_A + num_add_ck_B)/target_total_bonds*100:.2f}% (of target)" if target_total_bonds > 0 else "Total Completion: 100% (no new bonds needed)", flush=True)
    print(f"Probability-based rejections: AC={rejected_AC_bonds}, B={rejected_B_bonds}", flush=True)

    if iteration >= max_iterations:
        print("WARNING: Maximum iterations reached before completing crosslinking", flush=True)

    # Save the final snapshot if it hasn't been saved yet and we're at 100%
    if save_snapshots and last_checkpoint_reached < 10 and ((target_total_bonds > 0 and (num_add_ck_A + num_add_ck_B) / target_total_bonds >= 0.95) or target_total_bonds == 0):
        snapshot_filename = f"crosslink_100.dat"
        lmp_snapshot = els.lammps(
            natoms=lmp_data.natoms,
            nbonds=len(bond_info),
            natom_types=3,
            nbond_types=3,
            x=lmp_data.x, y=lmp_data.y, z=lmp_data.z,
            mass=lmp_data.mass,
            atom_info=lmp_data.atom_info,
            bond_info=bond_info
        )
        els.write_lammps_full(snapshot_filename, lmp_snapshot)
        print(f"Saved final 100% snapshot: {snapshot_filename}", flush=True)

    # Write updated structure with crosslinks to a file
    lmp_network = els.lammps(
        natoms=lmp_data.natoms,
        nbonds=len(bond_info),
        natom_types=3,
        nbond_types=3,
        x=lmp_data.x, y=lmp_data.y, z=lmp_data.z,
        mass=lmp_data.mass,
        atom_info=lmp_data.atom_info,
        bond_info=bond_info
    )
    els.write_lammps_full(output_file, lmp_network)

##########################################################################################################################################################################################################################

# SYS 3: Diamond architecture initial system

# Part 1: Crosslinking the B beads

def crosslink_sys3(relaxed_file, output_file, lmp_command, percentage_ck_B, functionality_B, ck_dis_B, bond_probability=0.7):

    lmp_data = els.read_lammps_full(relaxed_file)
    lmp_data.atom_info = lmp_data.atom_info[np.argsort(lmp_data.atom_info[:, 0])]
    atom_positions = lmp_data.atom_info[:, 4:7]  # Assuming positions are columns 4-7
    atom_types = lmp_data.atom_info[:, 2]  # Assuming atom types are in column 2

    bond_info = lmp_data.bond_info
    bond_index = bond_info[:,2:]-1
    box, coors = els.box_coors_from_lmp(lmp_data)

    positions_B = coors[atom_types == 2]
    idx_B = np.squeeze(np.argwhere(atom_types == 2))
    num_B = len(positions_B)

    # Calculate initial reaction counters for B beads
    reaction_counter_B = np.zeros(len(positions_B))
    for i in range(len(idx_B)):
        reaction_counter_B[i] = np.sum(np.concatenate(bond_index)==idx_B[i])
    capacity_ck_B = functionality_B * num_B - np.sum(reaction_counter_B)

    # Track connected molecules for B type atoms
    connected_molecules_B = {idx: set() for idx in idx_B}
    # Initialize the connection sets based on existing bonds
    for bond in bond_info:
        bond_type = bond[1]
        atom1, atom2 = int(bond[2])-1, int(bond[3])-1  # Convert to 0-indexed
        if atom_types[atom1] == 2 and atom_types[atom2] == 2:
            connected_molecules_B[atom1].add(atom2)
            connected_molecules_B[atom2].add(atom1)

    lmp_data.nbond_types = 4

    num_add_ck_B = 0
    target_ck_B = capacity_ck_B/2 * percentage_ck_B
    iterations_since_last_bond = 0
    max_iterations = 1000  # Prevent infinite loop if no more bonds can be formed
    iteration = 0

    # Track attempted vs successful bonds for reporting
    attempted_bonds = 0
    rejected_by_probability = 0

    print(f"Starting crosslinking: Target B bonds = {target_ck_B:.1f}", flush=True)
    print(f"Total B atoms: {num_B}, Total capacity: {capacity_ck_B}, Target percentage: {percentage_ck_B*100:.1f}%", flush=True)
    print(f"Bond formation probability: {bond_probability:.2f}", flush=True)

    while (num_add_ck_B < target_ck_B) and (iteration < max_iterations):
        iteration += 1
        bonds_added_this_iteration = 0

        if (num_add_ck_B < target_ck_B):
            random_indices = np.random.permutation(len(positions_B))
            for idx in random_indices:
                pos_B = positions_B[idx]
                i = idx

                if num_add_ck_B >= target_ck_B:
                    break
                if reaction_counter_B[i] < functionality_B:
                    # Find all B beads within cutoff distance (using box for initial filtering)
                    j_list = np.argwhere((np.abs(positions_B[:, 0] - pos_B[0]) < ck_dis_B) &
                                         (np.abs(positions_B[:, 1] - pos_B[1]) < ck_dis_B) &
                                         (np.abs(positions_B[:, 2] - pos_B[2]) < ck_dis_B)).flatten()

                    # Also randomize the order of potential partners
                    np.random.shuffle(j_list)


                    for _, j in enumerate(j_list):
                        if j != i and reaction_counter_B[j] < functionality_B:
                            # Calculate actual distance with PBC
                            if mc.pbc_distance(pos_B, positions_B[j], box) < ck_dis_B:
                                # Check if already connected (directly or indirectly)
                                if idx_B[j] not in connected_molecules_B[idx_B[i]]:
                                    attempted_bonds += 1

                                    # Apply probability-based bond formation
                                    if np.random.random() < bond_probability:
                                        # Create new bond
                                        bond_info = np.vstack([bond_info, [len(bond_info) + 1, 4, idx_B[j] + 1, idx_B[i] + 1]])
                                        reaction_counter_B[i] += 1
                                        reaction_counter_B[j] += 1
                                        num_add_ck_B += 1
                                        bonds_added_this_iteration += 1

                                        # Update connected molecules for B[i] and B[j]
                                        connected_molecules_B[idx_B[i]].add(idx_B[j])
                                        connected_molecules_B[idx_B[j]].add(idx_B[i])

                                        if num_add_ck_B >= target_ck_B:
                                            break
                                        break
                                    else:
                                        rejected_by_probability += 1

        print(f'Progress: {num_add_ck_B:.0f}/{target_ck_B:.0f} bonds created ({num_add_ck_B/target_ck_B*100:.1f}%)', flush=True)
        print(f'Bonds attempted: {attempted_bonds}, Rejected by probability: {rejected_by_probability}', flush=True)

        # Update LAMMPS data and run annealing step if we haven't reached the target
        if (num_add_ck_B < target_ck_B):
            update_lammps_data(lmp_data, bond_info, lmp_command)

            # Read updated positions after annealing
            lmp_data = els.read_lammps_full('anneal.dat')
            lmp_data.atom_info = lmp_data.atom_info[np.argsort(lmp_data.atom_info[:, 0])]
            atom_types = lmp_data.atom_info[:, 2]
            box, coors = els.box_coors_from_lmp(lmp_data)
            positions_B = coors[atom_types == 2]

    # Final status report
    print(f"Crosslinking complete. Created {num_add_ck_B:.0f} out of {target_ck_B:.0f} bonds ({num_add_ck_B/target_ck_B*100:.1f}%)", flush=True)
    print(f"Bond formation probability: {bond_probability:.2f}", flush=True)
    print(f"Total bonds attempted: {attempted_bonds}, Rejected by probability: {rejected_by_probability}", flush=True)
    print(f"Average functionality achieved: {np.mean(reaction_counter_B):.2f} per B atom (target: {functionality_B})", flush=True)

    if iteration >= max_iterations:
        print("WARNING: Maximum iterations reached before completing crosslinking", flush=True)
    # Write updated structure with crosslinks to a file
    lmp_network = els.lammps(
        natoms=lmp_data.natoms,
        nbonds=len(bond_info),
        natom_types=3,
        nbond_types=4,
        x=lmp_data.x, y=lmp_data.y, z=lmp_data.z,
        mass=lmp_data.mass,
        atom_info=lmp_data.atom_info,
        bond_info=bond_info
    )
    els.write_lammps_full(output_file, lmp_network)

    return lmp_network

# Part 2: Faux-crosslinking procedure

def parse_data_file(filename):
    """Parse a LAMMPS data file and extract atoms and bonds information."""
    atoms = []
    bonds = []
    atom_types = set()
    bond_types = set()

    # File structure information
    box_bounds = {}
    masses = {}

    with open(filename, 'r') as f:
        lines = f.readlines()

    # Parse header
    header_info = {}
    for i, line in enumerate(lines):
        line = line.strip()
        if "atoms" in line:
            header_info["num_atoms"] = int(line.split()[0])
        elif "atom types" in line:
            header_info["num_atom_types"] = int(line.split()[0])
        elif "bonds" in line:
            header_info["num_bonds"] = int(line.split()[0])
        elif "bond types" in line:
            header_info["num_bond_types"] = int(line.split()[0])
        elif "xlo xhi" in line:
            parts = line.split()
            box_bounds["x"] = (float(parts[0]), float(parts[1]))
        elif "ylo yhi" in line:
            parts = line.split()
            box_bounds["y"] = (float(parts[0]), float(parts[1]))
        elif "zlo zhi" in line:
            parts = line.split()
            box_bounds["z"] = (float(parts[0]), float(parts[1]))
        elif line == "Masses":
            mass_section = i + 1
            break

    # Parse masses
    i = mass_section
    while i < len(lines) and lines[i].strip() == "":
        i += 1

    while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith("Atoms"):
        if not lines[i].startswith("#"):
            parts = lines[i].strip().split()
            if len(parts) >= 2:
                masses[int(parts[0])] = float(parts[1])
        i += 1

    # Find Atoms section
    for i, line in enumerate(lines):
        if "Atoms" in line:
            atoms_start = i + 1
            break

    # Parse atoms
    i = atoms_start
    while i < len(lines) and not lines[i].strip():
        i += 1

    while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith("Bonds"):
        parts = lines[i].strip().split()
        if len(parts) >= 7 and not lines[i].strip().startswith("#"):  # Atom ID, Molecule ID, Atom Type, etc.
            atom_id = int(parts[0])
            atom_type = int(parts[2])
            atom_types.add(atom_type)
            # Parse atom coordinates and image flags
            if len(parts) >= 10:  # Full format with image flags
                atoms.append({
                    'id': atom_id,
                    'mol_id': int(parts[1]),
                    'type': atom_type,
                    'q': float(parts[3]),
                    'x': float(parts[4]),
                    'y': float(parts[5]),
                    'z': float(parts[6]),
                    'ix': int(parts[7]),
                    'iy': int(parts[8]),
                    'iz': int(parts[9]),
                })
            else:  # No image flags or incomplete data
                atoms.append({
                    'id': atom_id,
                    'mol_id': int(parts[1]),
                    'type': atom_type,
                    'q': float(parts[3]),
                    'x': float(parts[4]),
                    'y': float(parts[5]),
                    'z': float(parts[6]),
                    'ix': 0,
                    'iy': 0,
                    'iz': 0,
                })
        i += 1

    # Find Bonds section
    for i, line in enumerate(lines):
        if "Bonds" in line:
            bonds_start = i + 1
            break

    # Parse bonds
    i = bonds_start
    while i < len(lines) and not lines[i].strip():
        i += 1

    while i < len(lines) and lines[i].strip():
        parts = lines[i].strip().split()
        if len(parts) >= 4 and not lines[i].strip().startswith("#"):  # Bond ID, Bond Type, Atom 1, Atom 2
            bond_id = int(parts[0])
            bond_type = int(parts[1])
            bond_types.add(bond_type)
            bonds.append({
                'id': bond_id,
                'type': bond_type,
                'atom1': int(parts[2]),
                'atom2': int(parts[3]),
            })
        i += 1

    header_info["atom_types"] = list(atom_types)
    header_info["bond_types"] = list(bond_types)

    return header_info, box_bounds, masses, atoms, bonds

def create_initial_system(header_info, box_bounds, masses, atoms, bonds):
    """Create initial system with only beads and bond_type = 1."""
    # Keep all atoms
    initial_atoms = copy.deepcopy(atoms)

    # Keep only bond_type = 1 bonds
    initial_bonds = [bond for bond in bonds if bond['type'] == 1]

    # Update bond IDs to be sequential
    for i, bond in enumerate(initial_bonds):
        bond['id'] = i + 1

    return initial_atoms, initial_bonds

def get_bonds_to_add_pool(initial_bonds, all_bonds):
    """
    Create a pool of bonds that can be added to the system.
    Returns a list of bonds not in the initial system.
    """
    # Create a set of (atom1, atom2) pairs for existing bonds
    initial_bond_pairs = set()
    for bond in initial_bonds:
        pair = (bond['atom1'], bond['atom2'])
        reverse_pair = (bond['atom2'], bond['atom1'])
        initial_bond_pairs.add(pair)
        initial_bond_pairs.add(reverse_pair)

    # Create a list of bonds not in the initial system
    bonds_pool = []
    for bond in all_bonds:
        pair = (bond['atom1'], bond['atom2'])
        if pair not in initial_bond_pairs:
            bonds_pool.append(bond)

    return bonds_pool

def calculate_crosslinking_percentage(current_bonds, initial_bonds, total_bonds):
    """Calculate the current crosslinking percentage."""
    added_bonds = len(current_bonds) - len(initial_bonds)
    total_to_add = len(total_bonds) - len(initial_bonds)
    return (added_bonds / total_to_add) * 100 if total_to_add > 0 else 100

def write_data_file(filename, header_info, box_bounds, masses, atoms, bonds):
    """Write a LAMMPS data file with the current system state."""
    with open(filename, 'w') as f:
        # Header with exactly matching format
        f.write("Generated by ZY\n\n")
        f.write(f"{len(atoms)} atoms\n")
        f.write(f"{header_info['num_atom_types']} atom types\n")
        f.write(f"{len(bonds)} bonds\n")
        f.write(f"{header_info['num_bond_types']} bond types\n\n")

        # Write box bounds with full precision
        f.write(f"{box_bounds['x'][0]} {box_bounds['x'][1]} xlo xhi\n")
        f.write(f"{box_bounds['y'][0]} {box_bounds['y'][1]} ylo yhi\n")
        f.write(f"{box_bounds['z'][0]} {box_bounds['z'][1]} zlo zhi\n\n")

        # Write masses with a blank line
        f.write("Masses\n\n")
        for atom_type in sorted(masses.keys()):
            f.write(f"{atom_type} {masses[atom_type]:.3f}\n")

        # Write atoms with a blank line
        f.write("\nAtoms # full\n\n")
        for atom in sorted(atoms, key=lambda x: x['id']):
            # Include image flags if present
            if 'ix' in atom and 'iy' in atom and 'iz' in atom:
                f.write(f"{atom['id']} {atom['mol_id']} {atom['type']} {atom['q']:.6f} "
                      f"{atom['x']:.6f} {atom['y']:.6f} {atom['z']:.6f} {atom['ix']} {atom['iy']} {atom['iz']}\n")
            else:
                f.write(f"{atom['id']} {atom['mol_id']} {atom['type']} {atom['q']:.6f} "
                      f"{atom['x']:.6f} {atom['y']:.6f} {atom['z']:.6f} 0 0 0\n")

        # Write bonds with a blank line
        f.write("\nBonds \n\n")
        for bond in sorted(bonds, key=lambda x: x['id']):
            f.write(f"{bond['id']} {bond['type']} {bond['atom1']} {bond['atom2']}\n")

def main(input_file, output_prefix, bond_weight_list):
    #input_file = "test_ck_10.dat"
    #output_prefix = "crosslink_"

    # Parse input data file
    print(f"Parsing {input_file}...")
    header_info, box_bounds, masses, atoms, all_bonds = parse_data_file(input_file)

    # Create initial system with only beads and bond_type = 1
    print("Creating initial system...")
    initial_atoms, initial_bonds = create_initial_system(header_info, box_bounds, masses, atoms, all_bonds)

    # bond_weight_list = for bond_types 2, 3, 4 - length=3
    # Set bond_type weights (higher for types 2 and 3, lower for type 4)
    bond_type_weights = {
        1: 1.0,  # Not used as type 1 bonds are already in the initial system
        2: bond_weight_list[0],  # Higher probability
        3: bond_weight_list[1],  # Higher probability
        4: bond_weight_list[2]   # Lower probability
    }

    # Get pool of bonds that can be added
    bonds_pool = get_bonds_to_add_pool(initial_bonds, all_bonds)

    # Track progress and save snapshots
    current_bonds = copy.deepcopy(initial_bonds)
    current_bond_pairs = set()
    for bond in current_bonds:
        pair = (bond['atom1'], bond['atom2'])
        reverse_pair = (bond['atom2'], bond['atom1'])
        current_bond_pairs.add(pair)
        current_bond_pairs.add(reverse_pair)

    last_percentage = 0
    target_percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    print(f"Initial system has {len(initial_bonds)} bonds (bond_type = 1)")
    print(f"Need to add {len(bonds_pool)} more bonds to reach 100% crosslinking")

    # Save initial state (0% crosslinking)
    initial_file = f"{output_prefix}0.dat"
    write_data_file(initial_file, header_info, box_bounds, masses, initial_atoms, current_bonds)
    print(f"Saved initial state (0% crosslinking) to {initial_file}")

    # Use a copy of bonds_pool that we'll modify
    remaining_bonds = copy.deepcopy(bonds_pool)
    total_bonds_to_add = len(remaining_bonds)

# Initialize tqdm progress bar
    pbar = tqdm(total=total_bonds_to_add, desc="Crosslinking Progress")

    # Add bonds until we've added all possible bonds
    bonds_added = 0
    while remaining_bonds:
        # Calculate weights for each remaining bond based on its type
        weights = [bond_type_weights[bond['type']] for bond in remaining_bonds]

        # Randomly select a bond based on weights
        selected_bond_index = random.choices(range(len(remaining_bonds)), weights=weights, k=1)[0]
        selected_bond = remaining_bonds[selected_bond_index]

        # Add the selected bond to current bonds
        next_id = len(current_bonds) + 1
        bond_copy = copy.deepcopy(selected_bond)
        bond_copy['id'] = next_id
        current_bonds.append(bond_copy)

        # Update the set of current bond pairs
        pair = (selected_bond['atom1'], selected_bond['atom2'])
        reverse_pair = (selected_bond['atom2'], selected_bond['atom1'])
        current_bond_pairs.add(pair)
        current_bond_pairs.add(reverse_pair)

        # Remove the selected bond from the remaining bonds
        remaining_bonds.pop(selected_bond_index)

        # Also remove any bonds that involve the same atom pair
        remaining_bonds = [bond for bond in remaining_bonds
                         if (bond['atom1'], bond['atom2']) not in current_bond_pairs and
                            (bond['atom2'], bond['atom1']) not in current_bond_pairs]

        # Calculate current percentage
        bonds_added += 1
        pbar.update(1)

        current_percentage = (bonds_added / total_bonds_to_add) * 100

        # Check if we've reached a target percentage
        for target in target_percentages:
            if last_percentage < target <= current_percentage:
                output_file = f"{output_prefix}{target}.dat"
                write_data_file(output_file, header_info, box_bounds, masses, initial_atoms, current_bonds)
                print(f"Saved {target}% crosslinking to {output_file}")
                last_percentage = target
                break
    pbar.close()
    print("Crosslinking simulation complete!")
