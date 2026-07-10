import sys
import os
# Original el/thermoset-extremes version (from GitHub).
sys.path.append('../python_scripts')
from lattice_generation_codes import generate_lattice_sys3
from crosslinking_codes2 import crosslink_sys3
from deform_codes import deform_relax, extract_final_stress_avg

run_lammps = 'mpirun -np 4 lmp_mpi'

# Initialization: generate lattice
generate_lattice_sys3(chain_length=16, num_cells=5, output_file='lattice.dat')

# Relax the lattice
os.system('{} -in in.relax'.format(run_lammps))

# Crosslinking the system
crosslink_sys3('relax.dat', 'test_ck.dat', '{} -in in.anneal'.format(run_lammps), percentage_ck_B=1, functionality_B=3, ck_dis_B=1.2)

# Relax the system before deformation
os.system('{} -in in.relax_deform > out_relax_deform'.format(run_lammps))

path = './'

# Deformation
deform_relax(final_itime=100, step_size=100, path=path,run_lammps=run_lammps)
extract_final_stress_avg(final_itime=100, step_size=100, path=path, filename='Final_deform_sys2')
os.system('rm log_iter_*.csv')
os.system('rm log_noopt_*.csv')
