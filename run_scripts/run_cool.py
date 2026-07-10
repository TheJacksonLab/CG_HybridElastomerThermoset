import sys
import os
import re
#sys.path.append('../python_scripts')
#from lattice_generation_codes import generate_lattice_sys3
#from crosslinking_codes2 import crosslink_sys3, main
#from deform_codes import deform_relax, extract_final_stress_avg

#run_lammps = 'mpirun -np 12 /home/shrutii2/lmp'
run_lammps = 'mpirun -np 48 /home/shrutii2/lammps_29Aug2024/build_scruggs/lmp_scruggs -sf intel'

# Deform relax
input_file = "in.cool"

for i in range(0, 110, 10):
    with open(input_file, "r") as f:
        text = f.read()

    text = re.sub(r"relax_deform_\d+\.dat", f"relax_deform_{i}.dat", text)
    text = re.sub(r"cool_\d+\.dat", f"cool_{i}.dat", text)

    with open(input_file, "w") as f:
        f.write(text)

    os.system(f"{run_lammps} -in {input_file} > out_cool_{i}")
