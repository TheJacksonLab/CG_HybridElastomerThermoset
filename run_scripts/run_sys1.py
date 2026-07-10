import sys
import os
import re
# Crosslink-percent sweep version (default/active workflow).
sys.path.append('../python_scripts')
from lattice_generation_codes import generate_lattice_sys1
from crosslinking_codes2 import crosslink_sys1_mod
from deform_codes import deform_relax, extract_final_stress_avg

#run_lammps = 'mpirun -np 12 /home/shrutii2/lmp'
run_lammps = 'mpirun -np 48 /home/shrutii2/lammps_29Aug2024/build_scruggs/lmp_scruggs -sf intel'

# Initialization: generate lattice
generate_lattice_sys1(num_A_B_units=12800, num_C_units=64, output_file='lattice.dat')

# Relax the lattice
os.system('{} -in in.relax_long'.format(run_lammps))

# Crosslink elastomer
crosslink_sys1_mod(relaxed_file='relax.dat', output_file='test_ck_0.dat', lmp_command='{} -in in.anneal'.format(run_lammps), percentage_ck_A=1, percentage_ck_B=0, functionality_A=3, functionality_B=3, functionality_C=4, ck_dis_A=1.2, ck_dis_B=1.2, probability_AC=0.9, probability_B=0.5, save_snapshots=False)

# Deform relax the acrylate
os.system(f"{run_lammps} -in in.relax_deform_el > out_relax_deform_el")

# Crosslinking the system
crosslink_sys1_mod(relaxed_file='relax_deform_el.dat', output_file='test_ck_10.dat', lmp_command='{} -in in.anneal'.format(run_lammps), percentage_ck_A=0, percentage_ck_B=0.1, functionality_A=3, functionality_B=3, functionality_C=4, ck_dis_A=1.2, ck_dis_B=1.2, probability_AC=0.8, probability_B=0.5, save_snapshots=False)
crosslink_sys1_mod(relaxed_file='relax_deform_el.dat', output_file='test_ck_20.dat', lmp_command='{} -in in.anneal'.format(run_lammps), percentage_ck_A=0, percentage_ck_B=0.2, functionality_A=3, functionality_B=3, functionality_C=4, ck_dis_A=1.2, ck_dis_B=1.2, probability_AC=0.8, probability_B=0.5, save_snapshots=False)
crosslink_sys1_mod(relaxed_file='relax_deform_el.dat', output_file='test_ck_30.dat', lmp_command='{} -in in.anneal'.format(run_lammps), percentage_ck_A=0, percentage_ck_B=0.3, functionality_A=3, functionality_B=3, functionality_C=4, ck_dis_A=1.2, ck_dis_B=1.2, probability_AC=0.8, probability_B=0.5, save_snapshots=False)

crosslink_sys1_mod(relaxed_file='relax_deform_el.dat', output_file='test_ck_40.dat', lmp_command='{} -in in.anneal'.format(run_lammps), percentage_ck_A=0, percentage_ck_B=0.4, functionality_A=3, functionality_B=3, functionality_C=4, ck_dis_A=1.2, ck_dis_B=1.2, probability_AC=0.8, probability_B=0.5, save_snapshots=False)
crosslink_sys1_mod(relaxed_file='relax_deform_el.dat', output_file='test_ck_50.dat', lmp_command='{} -in in.anneal'.format(run_lammps), percentage_ck_A=0, percentage_ck_B=0.5, functionality_A=3, functionality_B=3, functionality_C=4, ck_dis_A=1.2, ck_dis_B=1.2, probability_AC=0.8, probability_B=0.5, save_snapshots=False)
crosslink_sys1_mod(relaxed_file='relax_deform_el.dat', output_file='test_ck_60.dat', lmp_command='{} -in in.anneal'.format(run_lammps), percentage_ck_A=0, percentage_ck_B=0.6, functionality_A=3, functionality_B=3, functionality_C=4, ck_dis_A=1.2, ck_dis_B=1.2, probability_AC=0.8, probability_B=0.5, save_snapshots=False)
crosslink_sys1_mod(relaxed_file='relax_deform_el.dat', output_file='test_ck_70.dat', lmp_command='{} -in in.anneal'.format(run_lammps), percentage_ck_A=0, percentage_ck_B=0.7, functionality_A=3, functionality_B=3, functionality_C=4, ck_dis_A=1.2, ck_dis_B=1.2, probability_AC=0.9, probability_B=0.5, save_snapshots=False)
crosslink_sys1_mod(relaxed_file='relax_deform_el.dat', output_file='test_ck_80.dat', lmp_command='{} -in in.anneal'.format(run_lammps), percentage_ck_A=0, percentage_ck_B=0.8, functionality_A=3, functionality_B=3, functionality_C=4, ck_dis_A=1.2, ck_dis_B=1.2, probability_AC=0.9, probability_B=0.5, save_snapshots=False)
crosslink_sys1_mod(relaxed_file='relax_deform_el.dat', output_file='test_ck_90.dat', lmp_command='{} -in in.anneal'.format(run_lammps), percentage_ck_A=0, percentage_ck_B=0.9, functionality_A=3, functionality_B=3, functionality_C=4, ck_dis_A=1.2, ck_dis_B=1.2, probability_AC=0.9, probability_B=0.5, save_snapshots=False)
crosslink_sys1_mod(relaxed_file='relax_deform_el.dat', output_file='test_ck_100.dat', lmp_command='{} -in in.anneal'.format(run_lammps), percentage_ck_A=0, percentage_ck_B=1, functionality_A=3, functionality_B=3, functionality_C=4, ck_dis_A=1.2, ck_dis_B=1.2, probability_AC=0.9, probability_B=0.5, save_snapshots=False)

# Relax system before deformation
input_file = "in.relax_deform"

for i in range(10, 110, 10):
    with open(input_file, "r") as f:
        text = f.read()

    text = re.sub(r"test_ck_\d+\.dat", f"test_ck_{i}.dat", text)
    text = re.sub(r"relax_deform_\d+\.dat", f"relax_deform_{i}.dat", text)

    with open(input_file, "w") as f:
        f.write(text)

    os.system(f"{run_lammps} -in {input_file} > out_relax_deform_{i}")

path = './'

# Deformation
for file_num in range(10, 110, 10):
    current_path = f'{path}/relax_deform_{file_num}'
    os.makedirs(current_path, exist_ok=True)

    os.system(f"sed 's/relax_deform.dat/relax_deform_{file_num}.dat/g' in.deform > in.deform_{file_num}")

    os.system(f'mv in.deform in.deform_original')
    os.system(f'mv in.deform_{file_num} in.deform')

    deform_relax(final_itime=160, step_size=200, path=current_path, run_lammps=run_lammps)
    extract_final_stress_avg(final_itime=160, step_size=200, path=current_path, filename=f'deform_sys3_cl{file_num}')

    os.system(f'mv in.deform in.deform_{file_num}')
    os.system(f'mv in.deform_original in.deform')

    os.system(f'rm {current_path}/log_iter_*.csv')
    os.system(f'rm {current_path}/log_noopt_*.csv')
    os.system(f'rm in.deform_{file_num}')
