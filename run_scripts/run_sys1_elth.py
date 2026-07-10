import sys
import os
# Original el/thermoset-extremes version (from GitHub).
sys.path.append('../python_scripts')
from lattice_generation_codes import generate_lattice_sys1
from crosslinking_codes2 import crosslink_sys1
from deform_codes import deform_relax, extract_final_stress_avg

run_lammps = 'mpirun -np 4 lmp_mpi'

# Initialization: generate lattice
generate_lattice_sys1(num_A_B_units=32000, num_C_units=1000, output_file='lattice.dat')

# Relax the lattice
os.system('{} -in in.relax'.format(run_lammps))

# Crosslinking the system
crosslink_sys1(relaxed_file='relax.dat', output_file='test_ck.dat', lmp_command='{} -in in.anneal'.format(run_lammps), percentage_ck_A=1, percentage_ck_B=1, functionality_A=3, functionality_B=3, functionality_C=4, ck_dis_A=1.2, ck_dis_B=1.2, probability_AC=0.8, probability_B=0.4, save_snapshots=True)

# Relax the system before deformation
os.system('{} -in in.relax_deform > out_relax_deform'.format(run_lammps))

path = './'

# Deformation
deform_relax(final_itime=100, step_size=100, path=path,run_lammps=run_lammps)
extract_final_stress_avg(final_itime=100, step_size=100, path=path, filename='Final_deform_sys1')
os.system('rm log_iter_*.csv')
os.system('rm log_noopt_*.csv')
