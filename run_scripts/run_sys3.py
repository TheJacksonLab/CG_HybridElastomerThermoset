import sys
import os
import re
# Crosslink-percent sweep version (default/active workflow).
sys.path.append('../python_scripts')
from lattice_generation_codes import generate_lattice_sys3
from crosslinking_codes2 import crosslink_sys3, main
from deform_codes import deform_relax, extract_final_stress_avg

#run_lammps = 'mpirun -np 12 /home/shrutii2/lmp'
run_lammps = 'mpirun -np 48 /home/shrutii2/lammps_29Aug2024/build_scruggs/lmp_scruggs -sf intel'

# Initialization: generate lattice
#generate_lattice_sys3(chain_length=100, num_cells=2, output_file='lattice.dat')

# Relax the lattice
#os.system('{} -in in.relax_long > out_relax'.format(run_lammps))
#os.system('mv relax.dat relax_deform_0.dat')  # consistent 0%-crosslink / elastomer naming across sys1/sys2/sys3

# Crosslinking the system for the different B percentages
#crosslink_sys3('relax_deform_0.dat', 'test_ck_10.dat', '{} -in in.anneal'.format(run_lammps), percentage_ck_B=0.1, functionality_B=3, ck_dis_B=1.2)
#crosslink_sys3('relax_deform_0.dat', 'test_ck_20.dat', '{} -in in.anneal'.format(run_lammps), percentage_ck_B=0.2, functionality_B=3, ck_dis_B=1.2)
#crosslink_sys3('relax_deform_0.dat', 'test_ck_30.dat', '{} -in in.anneal'.format(run_lammps), percentage_ck_B=0.3, functionality_B=3, ck_dis_B=1.2)
#crosslink_sys3('relax_deform_0.dat', 'test_ck_40.dat', '{} -in in.anneal'.format(run_lammps), percentage_ck_B=0.4, functionality_B=3, ck_dis_B=1.2)
#crosslink_sys3('relax_deform_0.dat', 'test_ck_50.dat', '{} -in in.anneal'.format(run_lammps), percentage_ck_B=0.5, functionality_B=3, ck_dis_B=1.2)
#crosslink_sys3('relax_deform_0.dat', 'test_ck_60.dat', '{} -in in.anneal'.format(run_lammps), percentage_ck_B=0.6, functionality_B=3, ck_dis_B=1.2)
#crosslink_sys3('relax_deform_0.dat', 'test_ck_70.dat', '{} -in in.anneal'.format(run_lammps), percentage_ck_B=0.7, functionality_B=3, ck_dis_B=1.2)
#crosslink_sys3('relax_deform_0.dat', 'test_ck_80.dat', '{} -in in.anneal'.format(run_lammps), percentage_ck_B=0.8, functionality_B=3, ck_dis_B=1.2)
#crosslink_sys3('relax_deform_0.dat', 'test_ck_90.dat', '{} -in in.anneal'.format(run_lammps), percentage_ck_B=0.9, functionality_B=3, ck_dis_B=1.2)
#crosslink_sys3('relax_deform_0.dat', 'test_ck_100.dat', '{} -in in.anneal'.format(run_lammps), percentage_ck_B=1, functionality_B=3, ck_dis_B=1.2)

# Relax the system before deformation
#os.system('{} -in in.relax_deform > out_relax_deform'.format(run_lammps))

# Deform relax
#input_file = "in.relax_deform"

#for i in range(10, 110, 10):
#    with open(input_file, "r") as f:
#        text = f.read()

#    text = re.sub(r"test_ck_\d+\.dat", f"test_ck_{i}.dat", text)
#    text = re.sub(r"relax_deform_\d+\.dat", f"relax_deform_{i}.dat", text)

#    with open(input_file, "w") as f:
#        f.write(text)

#    os.system(f"{run_lammps} -in {input_file} > out_relax_deform_{i}")

path = './'

# Deformation

#input_file = "in.deform"
#deform_relax(final_itime=60, step_size=1000, path=path,run_lammps=run_lammps)
#extract_final_stress_avg(final_itime=60, step_size=1000, path=path, filename='deform_sys3_cl10')
#os.system('rm log_iter_*.csv')
#os.system('rm log_noopt_*.csv')

# Loop through all your input files
for file_num in range(10, 110, 10):
    current_path = f'{path}/relax_deform_{file_num}'
    os.makedirs(current_path, exist_ok=True)

    os.system(f"sed 's/relax_deform.dat/relax_deform_{file_num}.dat/g' in.deform > in.deform_{file_num}")

    os.system(f'mv in.deform in.deform_original')
    os.system(f'mv in.deform_{file_num} in.deform')

    deform_relax(final_itime=160, step_size=200, path=current_path, run_lammps=run_lammps)
    extract_final_stress_avg(final_itime=160, step_size=200, path=current_path, filename=f'deform_sys3_cl{file_num}')

#    deform_relax(final_itime=1, step_size=200, path=current_path, run_lammps=run_lammps)
#    extract_final_stress_avg(final_itime=1, step_size=200, path=current_path, filename=f'test_cl{file_num}')

    os.system(f'mv in.deform in.deform_{file_num}')
    os.system(f'mv in.deform_original in.deform')

    os.system(f'rm {current_path}/log_iter_*.csv')
    os.system(f'rm {current_path}/log_noopt_*.csv')
    os.system(f'rm in.deform_{file_num}')

#input_file = 'test_ck.dat'
#output_prefix = 'crosslink_same'
#bond_weight_list = [1.0, 1.0, 1.0]
#main(input_file, output_prefix, bond_weight_list)

#input_file = 'test_ck.dat'
#output_prefix = 'crosslink'
#bond_weight_list = [5.0, 5.0, 1.0]
#main(input_file, output_prefix, bond_weight_list)
