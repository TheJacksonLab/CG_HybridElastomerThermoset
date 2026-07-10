# CG Hybrid Monomer System
<img width="1461" height="1247" alt="TOC" src="https://github.com/user-attachments/assets/44d606b4-5062-47ec-a45d-5a0a32029dbb" />

Coarse-grained molecular dynamics (LAMMPS) code for simulating a hybrid epoxy-acrylate dual-cure polymer network.

This repository accompanies the paper *"Computational Exploration of the Structure and Mechanical Behaviour of Hybrid Epoxy-Acrylate Dual-Cure Systems"* (Iyer, Yu, Page, and Jackson). The model is based on a hybrid monomer (ECA) that carries both an acrylate and an epoxy functional group, crosslinked with a tetrafunctional acrylate crosslinker (TEGDA). Violet light drives acrylate-only (radical) polymerization to give a loosely crosslinked elastomer; UV light drives both acrylate and epoxy (radical + cationic) polymerization to give a densely crosslinked thermoset. This code simulates that whole space computationally: the two limiting cases (pure elastomer and pure thermoset) as well as a full sweep across epoxy crosslink percentage between them, tracking structural, thermal (Tg), and mechanical evolution as the network densifies.

## Model systems

Three initial network architectures are used throughout, representing increasing degrees of structural order:

- **sys1** — individual ECA monomers and TEGDA crosslinkers placed on a cubic lattice as independent, unpolymerized units. Produces the most topologically disordered networks.
- **sys2** — acrylate beads pre-assembled into linear chains (length 30 or 100 units) before lattice placement, with free crosslinkers. Improves chain connectivity relative to sys1 while retaining some disorder.
- **sys3** — an idealized, minimally-constrained reference: TEGDA crosslinkers on a diamond lattice, interconnected by acrylate chains of fixed length, forming an ordered periodic network.

Each system is run in two ways:

1. **el/thermoset extremes** — the two limiting cases only: pure elastomer (acrylate-only crosslinking) and pure thermoset (acrylate + epoxy crosslinking), matching the two experimental violet/UV-light curing pathways.
2. **crosslink-percent sweep** — a full sweep of epoxy crosslink percentage from 0% to 100% in 10% increments, tracking how structure and mechanics evolve continuously between the elastomer and thermoset limits.

## Repository structure

```
lammps_scripts/    LAMMPS input scripts for lattice relaxation, crosslinking anneal steps,
                   pre-deformation relaxation, and stepwise tensile deformation.
                   Most scripts are shared between both workflows - where the crosslink-percent
                   sweep requires different run lengths/protocol (in.deform, in.deform_opt),
                   the el/thermoset-extremes version is suffixed "_elth".

run_scripts/       Python driver scripts that run the full pipeline per system (sys1/sys2/sys3):
                   lattice generation -> relaxation -> crosslinking -> pre-deformation relaxation
                   -> stepwise deformation (-> cooling, for Tg estimation). The default scripts
                   run the crosslink-percent sweep; "_elth" variants run the el/thermoset extremes.

python_scripts/    Core simulation logic imported by run_scripts:
                     lattice_generation_codes.py  - builds the initial sys1/sys2/sys3 lattices
                     crosslinking_codes2.py       - crosslinking procedures for all three systems
                     deform_codes.py              - stepwise deformation driver and stress extraction

utils/             Shared utils imported by python_scripts:
                     extract_local_str.py - LAMMPS data/dump file I/O, local structure extraction
                     my_common.py         - general utilities (periodic-boundary distances, file I/O,
                                            structural analysis helpers)
```

## Overall simulation workflow

1. **Lattice generation** — build the initial sys1/sys2/sys3 configuration (`python_scripts/lattice_generation_codes.py`).
2. **Relaxation** — relax and compress the initial lattice to a stable melt density (`lammps_scripts/in.relax_long`, or `in.relax` for the el/thermoset workflow).
3. **Crosslinking** — form acrylate (and, for the thermoset/higher-percent cases, epoxy) bonds, interleaved with short annealing runs (`python_scripts/crosslinking_codes2.py`, `lammps_scripts/in.anneal`).
4. **Pre-deformation relaxation** — relax the crosslinked network before mechanical testing (`lammps_scripts/in.relax_deform`, `in.relax_deform_el`).
5. **Deformation** — apply stepwise uniaxial strain and extract stress-strain response (`lammps_scripts/in.deform`, `in.deform_opt`, driven by `python_scripts/deform_codes.py`).
6. **Cooling (optional)** — gradual cooling ramp for glass transition temperature (Tg) estimation (`lammps_scripts/in.cool`, `run_scripts/run_cool.py`).

## Requirements

- LAMMPS (compiled with the packages used by the input scripts)
- Python 3 with `numpy`, `scipy`, `pandas`, `tqdm`
