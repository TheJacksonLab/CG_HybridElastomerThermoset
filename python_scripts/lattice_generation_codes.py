import numpy as np
import random
import extract_local_str as els
import my_common as mc
import os
from scipy.spatial import KDTree, cKDTree
import re
import sys
from collections import Counter

# SYS 1: Individual hybrid monomers and crosslinkers in simulation box

def generate_lattice_sys1(num_A_B_units, num_C_units, output_file):

    a = int(np.ceil(np.cbrt(num_A_B_units)))  # Side length of cubic lattice for A

    #box_size = int(np.sqrt(num_A_B_units))
    #la = box_size / a  # Lattice constant

    bond_length = 1.2
    min_distance = 0.5  # Minimum distance between beads to prevent overlap

    # Calculate spacing needed for A-B units
    unit_space_needed = 2 * bond_length  # Space for A-B pair (they're placed bond_length apart)

    # Calculate spacing between lattice points
    # Need to ensure there's enough space between adjacent A-B pairs
    lattice_spacing = max(unit_space_needed + min_distance, 2 * bond_length)

    # Calculate box size with buffer for C units
    # The 1.1 factor adds a 10% buffer
    base_box_size = a * lattice_spacing * 1.1

    # Ensure box is large enough for all particles
    # This calculation ensures minimum volume considering total particle count
    density_factor = 1.0 + (num_C_units / (num_A_B_units * 2)) * 0.5
    adjusted_box_size = base_box_size * np.cbrt(density_factor)

    # Use the larger of the two calculated sizes
    box_size = max(base_box_size, adjusted_box_size)
    la = box_size / a  # Lattice constant

    total_atoms_per_A_B_unit = 2
    total_atoms_per_C_unit = 1
    total_bonds_per_A_B_unit = 1
    total_bonds_per_C_unit = 0

    total_atoms_A_B = num_A_B_units * total_atoms_per_A_B_unit
    total_atoms_C = num_C_units * total_atoms_per_C_unit
    total_bonds_A_B = num_A_B_units * total_bonds_per_A_B_unit

    total_atoms = total_atoms_A_B + total_atoms_C
    total_bonds = total_bonds_A_B

    atom_positions = np.zeros((total_atoms, 3))
    atom_types = np.zeros(total_atoms, dtype=int)
    mol_ids = np.zeros(total_atoms, dtype=int)
    bond_info = np.zeros((total_bonds, 4))  # bond id, bond type, atom1 id, atom2 id

    # Place A-B units in lattice and create bonds
    for i in range(num_A_B_units):
        ix, iy, iz = np.unravel_index(i, (a, a, a))
        base_index_A_B = i * total_atoms_per_A_B_unit

        atom_positions[base_index_A_B] = [ix * la, iy * la, iz * la]
        atom_positions[base_index_A_B + 1] = atom_positions[base_index_A_B] + np.array([bond_length, 0, 0])

        atom_types[base_index_A_B:base_index_A_B + 2] = [1, 2]  # Atom type 1 for A, type 2 for B
        mol_ids[base_index_A_B:base_index_A_B + 2] = i + 1  # Corrected indexing for A
        mol_ids[base_index_A_B + 1:base_index_A_B + 3] = i + 1 # Corrected indexing for B

        if i < total_bonds_A_B:
            bond_info[i] = [i + 1, 1, base_index_A_B + 1, base_index_A_B + 2]

    # Place A-A units halfway between A-B units
    for i in range(num_C_units):
        ix, iy, iz = np.unravel_index(i, (a, a, a))
        base_index_A_A = total_atoms_A_B + i * total_atoms_per_C_unit

        corresponding_A_B_unit_index = i // 2  # Determine the corresponding A-B unit index

        x_offset = 0.5 * bond_length  # Offset the x-coordinate by half the bond length
        x_coord = ix * la + x_offset
        y_coord = iy * la
        z_coord = (iz + 0.5) * la  # Place A-A units halfway between A-B layers

        atom_positions[base_index_A_A] = [x_coord, y_coord, z_coord]

        atom_types[base_index_A_A] = 3  # Atom type 1 for A
        mol_ids[base_index_A_A] = num_A_B_units + i + 1  # Molecular ID

    # Combine all atom information
    atom_info = np.hstack([
        np.arange(1, total_atoms + 1).reshape(-1, 1),  # Atom ID
        mol_ids.reshape(-1, 1),                       # Molecular ID
        atom_types.reshape(-1, 1),                    # Atom Type
        np.repeat(0, total_atoms).reshape(-1, 1),
        atom_positions,                               # Coordinates
        np.zeros((total_atoms, 3), dtype=int)         # Image index (0,0,0) for all atoms
    ])

    # Create LAMMPS polymer object
    lmp_polymer = els.lammps(
        natoms=total_atoms,
        nbonds=total_bonds,
        natom_types=3,
        nbond_types=1,
        x=[0, box_size], y=[0, box_size], z=[0, box_size],
        mass=np.array([[1.0, 1.0], [2.0, 1.5], [3.0, 2.0]]),
        atom_info=atom_info,
        bond_info=bond_info
    )

    # Write to LAMMPS data file
    els.write_lammps_full(output_file, lmp_polymer)

#############################################################################################################################################################################################################################

# SYS 2: Hybrid-monomer chains

def generate_lattice_sys2(num_chains, chain_length, num_C_units, output_file, min_distance, box_size):

    a = int(np.ceil(np.cbrt(num_chains)))

    bond_length = 1.2

    chain_space_needed = chain_length * bond_length  # Space each chain occupies

    la = box_size / a

    total_atoms_AB = num_chains*chain_length*2
    total_atoms_C = num_C_units
    total_atoms = total_atoms_AB + total_atoms_C
    total_bonds = (chain_length-1)*num_chains + num_chains*chain_length

    atom_positions = np.zeros((total_atoms, 3))
    atom_types = np.zeros(total_atoms, dtype=int)
    mol_ids = np.zeros(total_atoms, dtype=int)
    bond_info = np.zeros((total_bonds, 4))

    n_bond = 0
    # Place AB molecules in lattice and create bonds
    for i in range(num_chains):
        ix, iy, iz = np.unravel_index(i, (a, a, a))
        base_index_A = 2 * chain_length * i

        for j in range(chain_length):
            atom_positions[base_index_A+j*2] = [ix * la+j*bond_length, iy * la, iz * la]
            atom_types[base_index_A+j*2] = 1
            atom_positions[base_index_A+j*2+1] = atom_positions[base_index_A+j*2] + np.array([0, bond_length, 0])
            atom_types[base_index_A+j*2+1] = 2
            bond_info[n_bond] = [n_bond + 1, 1, base_index_A+j*2 + 1, base_index_A+j*2 + 2] # A-B is type 1
            n_bond+=1
            if j>=1:
                bond_info[n_bond] = [n_bond + 1, 2, base_index_A+j*2 + 1, base_index_A+j*2-2 + 1] # A-A is type 2
                n_bond+=1
        mol_ids[base_index_A:base_index_A + 2*num_chains] = i + 1

    # Place C molecules randomly, ensuring they don't overlap with AB molecules
    kd_tree = cKDTree(atom_positions[:total_atoms_AB])
    for i in range(num_C_units):
        while True:
            position_C = np.random.uniform(min_distance, box_size - 2*min_distance, 3) # Changed
            if kd_tree.query(position_C)[0] >= min_distance:
                break
        atom_positions[total_atoms_AB+i] = position_C
        atom_types[total_atoms_AB+i] = 3
        mol_ids[total_atoms_AB+i] = num_chains + i + 1

    if n_bond != total_bonds:
        print('WARNING: something with the bonds is wrong!!')
    # Combine all atom information
    atom_info = np.hstack([
        np.arange(1, total_atoms + 1).reshape(-1, 1),  # Atom ID
        mol_ids.reshape(-1, 1),                       # Molecular ID
        atom_types.reshape(-1, 1),                    # Atom Type
        np.repeat(0,total_atoms).reshape(-1,1),
        atom_positions,                               # Coordinates
        np.zeros((total_atoms, 3), dtype=int)         # Image index (0,0,0) for all atoms
    ])

    # Create LAMMPS polymer object
    lmp_polymer = els.lammps(
        natoms=total_atoms,
        nbonds=total_bonds,
        natom_types=3,
        nbond_types=2,
        x=[0, box_size], y=[0, box_size], z=[0, box_size],
        mass=np.array([[1.0, 1.0], [2.0, 1.5], [3.0, 2.0]]),
        atom_info=atom_info,
        bond_info=bond_info
    )

    # Write to LAMMPS data file
    els.write_lammps_full(output_file, lmp_polymer)

#########################################################################################################################################################################################################################

# SYS 3: Diamond architecture initial system

def minimum_image_distance(pos1, pos2, box_size):

    diff = pos2 - pos1
    diff = diff - box_size * np.round(diff / box_size)
    dist = np.linalg.norm(diff)
    return diff, dist

def get_neighbors_periodic(pos, positions, box_size, max_dist=None):

    neighbors = []
    for i in range(len(positions)):
        diff, dist = minimum_image_distance(pos, positions[i], box_size)
        if dist > 0.1 and (max_dist is None or dist < max_dist):
            neighbors.append(i)
    return neighbors

def generate_lattice_sys3(chain_length, num_cells, output_file):

    bond_length = 1.2

    a = (4 / np.sqrt(3)) * (chain_length + 1) * bond_length
    box_size = num_cells * a  # For nxnxn supercell

    basis_positions = np.array([
        [0, 0, 0],
        [0, 0.5, 0.5],
        [0.5, 0, 0.5],
        [0.5, 0.5, 0],
        [0.25, 0.25, 0.25],
        [0.25, 0.75, 0.75],
        [0.75, 0.25, 0.75],
        [0.75, 0.75, 0.25]
    ])

    # supercell
    crosslinker_positions = []
    for i in range(num_cells):
        for j in range(num_cells):
            for k in range(num_cells):
                offset = np.array([i, j, k])
                for basis in basis_positions:
                    pos = (offset + basis) * a
                    crosslinker_positions.append(pos)

    crosslinker_positions = np.array(crosslinker_positions)
    num_crosslinkers = len(crosslinker_positions)

    connections = []
    connection_vectors = []  # Store the minimum image vectors for chain placement
    for i in range(num_crosslinkers):
        neighbors = get_neighbors_periodic(crosslinker_positions[i],
                                        crosslinker_positions,
                                        box_size,
                                        max_dist=a*0.45)
        for j in neighbors:
            if i < j:  # Avoid duplicate connections
                diff, _ = minimum_image_distance(crosslinker_positions[i],
                                              crosslinker_positions[j],
                                              box_size)
                connections.append((i, j))
                connection_vectors.append(diff)

    atoms_per_chain = chain_length * 2  # A-B pairs
    num_chains = len(connections)
    total_atoms = num_crosslinkers + num_chains * atoms_per_chain
    #bonds_per_chain = 2 * chain_length - 1  # A-A bonds + A-B bonds
    #bonds_per_chain = chain_length + 1 if chain_length > 1 else 2
    #total_bonds = num_chains * bonds_per_chain + num_chains * 2  # Include crosslinker bonds
    bonds_per_chain = (chain_length * 1) + (chain_length - 1) + 2
    total_bonds = num_chains * bonds_per_chain

    atom_positions = np.zeros((total_atoms, 3))
    atom_types = np.zeros(total_atoms, dtype=int)
    mol_ids = np.zeros(total_atoms, dtype=int)
    bond_info = np.zeros((total_bonds, 4))

    # Place crosslinkers (type 3)
    atom_positions[:num_crosslinkers] = crosslinker_positions
    atom_types[:num_crosslinkers] = 3
    mol_ids[:num_crosslinkers] = np.arange(1, num_crosslinkers + 1)

    n_bond = 0
    curr_atom = num_crosslinkers

    for chain_idx, ((c1, c2), connection_vector) in enumerate(zip(connections, connection_vectors)):
        start_pos = crosslinker_positions[c1]
        direction = connection_vector / np.linalg.norm(connection_vector)
        step_length = bond_length  # Use fixed bond length for A-A spacing

        perp_direction = np.cross(direction, [0, 0, 1])
        if np.all(perp_direction == 0):
            perp_direction = np.cross(direction, [0, 1, 0])
        perp_direction = perp_direction / np.linalg.norm(perp_direction)

        for j in range(chain_length):
            # Position along the chain with fixed bond length spacing
            pos_A = start_pos + direction * ((j + 1) * step_length)
            # Wrap positions into primary box
            pos_A = pos_A % box_size
            atom_positions[curr_atom] = pos_A
            atom_types[curr_atom] = 1

            pos_B = pos_A + perp_direction * bond_length
            # Wrap positions into primary box
            pos_B = pos_B % box_size
            atom_positions[curr_atom + 1] = pos_B
            atom_types[curr_atom + 1] = 2

            bond_info[n_bond] = [n_bond + 1, 1, curr_atom + 1, curr_atom + 2]
            n_bond += 1

            if j > 0:
                bond_info[n_bond] = [n_bond + 1, 2, curr_atom + 1, curr_atom - 1]
                n_bond += 1

            if j == 0:
                bond_info[n_bond] = [n_bond + 1, 3, curr_atom + 1, c1 + 1]
                n_bond += 1
            elif j == chain_length - 1:
                bond_info[n_bond] = [n_bond + 1, 3, curr_atom + 1, c2 + 1]
                n_bond += 1

            curr_atom += 2
            mol_ids[curr_atom-2:curr_atom] = chain_idx + num_crosslinkers + 1

    # Combine atom information
    atom_info = np.hstack([
        np.arange(1, total_atoms + 1).reshape(-1, 1),  # Atom ID
        mol_ids.reshape(-1, 1),                        # Molecular ID
        atom_types.reshape(-1, 1),                     # Atom Type
        np.repeat(0, total_atoms).reshape(-1, 1),      # Charge
        atom_positions,                                # Coordinates
        np.zeros((total_atoms, 3), dtype=int)          # Image index
    ])

    # LAMMPS polymer object
    lmp_polymer = els.lammps(
        natoms=total_atoms,
        nbonds=total_bonds,
        natom_types=3,  # A, B, and C (crosslinker)
        nbond_types=3,  # A-B, A-A, and A-C bonds
        x=[0, box_size], y=[0, box_size], z=[0, box_size],
        mass=np.array([[1, 1.0], [2, 1.5], [3, 2.0]]),  # Masses for A, B, and C
        atom_info=atom_info,
        bond_info=bond_info
    )

    els.write_lammps_full(output_file, lmp_polymer)

    return lmp_polymer
