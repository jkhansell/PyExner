import os
import subprocess
import yaml
import shutil
import math

def get_boundaries(Lx, Ly, dh):
    dh_round = round(dh, 5)
    
    return {
        "outlet1": { # Left
            "type": "Reflective",
            "polygon": [
                [-Lx-dh_round, -Ly-dh_round],
                [-Lx+dh_round, -Ly-dh_round],
                [-Lx+dh_round,  Ly+dh_round],
                [-Lx-dh_round,  Ly+dh_round]
            ],
            "values": [0.0, 0.0, 0.0, 0.0],
            "normal": [-1.0, 0.0]
        },
        "outlet2": { # Right
            "type": "Reflective",
            "polygon": [
                [ Lx-dh_round, -Ly-dh_round],
                [ Lx+dh_round, -Ly-dh_round],
                [ Lx+dh_round,  Ly+dh_round],
                [ Lx-dh_round,  Ly+dh_round]
            ],
            "values": [0.0, 0.0, 0.0, 0.0],
            "normal": [1.0, 0.0]
        },
        "outlet3": { # Bottom
            "type": "Reflective",
            "polygon": [
                [-Lx-dh_round, -Ly-dh_round],
                [ Lx+dh_round, -Ly-dh_round],
                [ Lx+dh_round, -Ly+dh_round],
                [-Lx-dh_round, -Ly+dh_round]
            ],
            "values": [0.0, 0.0, 0.0, 0.0],
            "normal": [0.0, -1.0]
        },
        "outlet4": { # Top
            "type": "Reflective",
            "polygon": [
                [-Lx-dh_round,  Ly-dh_round],
                [ Lx+dh_round,  Ly-dh_round],
                [ Lx+dh_round,  Ly+dh_round],
                [-Lx-dh_round,  Ly+dh_round]
            ],
            "values": [0.0, 0.0, 0.0, 0.0],
            "normal": [0.0, 1.0]
        }
    }

def generate_slurm(output_dir, parNx, parNy, job_name):
    gpus = parNx * parNy
    nodes = math.ceil(gpus / 4)
    gpus_per_node = 4 if gpus >= 4 else gpus
    
    slurm_content = f"""#!/bin/bash
#SBATCH -A ehrtas
#SBATCH --nodes={nodes}
#SBATCH --job-name={job_name}
#SBATCH --output={job_name}-%j.out
#SBATCH --error={job_name}-%j.err
#SBATCH --time=00:10:00
#SBATCH --gres gpu:{gpus_per_node}
#SBATCH --partition booster

source /p/project/ehrtas/villalobos1/bedload/PyExner/machines/juwels.gpu

rm -f output.nc
MPI4JAX_USE_CUDA_MPI=1 srun -N {nodes} -n {gpus} --gpu-bind=closest --cpu-bind=cores mamba run --name PyExner python ../../2D_dambreak.py
"""
    with open(os.path.join(output_dir, "distributed.slurm"), "w") as f:
        f.write(slurm_content)

def generate_pbs(output_dir, parNx, parNy, job_name):
    gpus = parNx * parNy
    nodes = math.ceil(gpus / 4)  # adjust if Polaris differs
    select = max(1, nodes)

    gpus_per_node = 4 if gpus >= 4 else gpus

    queue = "preemptable" if nodes <= 10 else "prod"

    pbs_content = f"""#!/bin/bash
#PBS -N {job_name}
#PBS -A insitu
#PBS -q {queue}
#PBS -l walltime=00:30:00
#PBS -l place=scatter
#PBS -k doe
#PBS -l select={select}:system=polaris
#PBS -l filesystems=grand
#PBS -o {job_name}.out
#PBS -e {job_name}.err

source ~/.bashrc
source /lus/grand/projects/insitu/jkcenat/PyExner/machines/polaris.gpu
export MPICH_GPU_SUPPORT_ENABLED=1

cd $PBS_O_WORKDIR

MPI4JAX_USE_CUDA_MPI=1 mpiexec -n {gpus} --ppn {gpus_per_node} ../../set_affinity.sh mamba run -n pyexner python ../../run.py ./input.yaml
"""
    with open(os.path.join(output_dir, "distributed.pbs"), "w") as f:
        f.write(pbs_content)

def generate_job_script(output_dir, parNx, parNy, job_name, system="slurm"):
    if system == "slurm":
        return generate_slurm(output_dir, parNx, parNy, job_name)
    elif system == "pbs":
        return generate_pbs(output_dir, parNx, parNy, job_name)

def generate_case(output_dir, Lx, Ly, dh, parNx, parNy, end_time=1.0, grass_factor=0.005, job_name="test_multi_node"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Generate NetCDF input
    print(f"Generating input for {output_dir} (Lx={Lx}, Ly={Ly}, dh={dh}, parNx={parNx}, parNy={parNy})")
    subprocess.run(
        ["python3", "build_2D_dambreak_input.py", str(Lx), str(Ly), str(dh)],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        check=True
    )
    
    # Move the generated input.nc to the target directory
    shutil.move("input.nc", os.path.join(output_dir, "input.nc"))

    # 2. Write input.yaml
    config = {
        "end_time": end_time,
        "out_freq": end_time+1,
        "io_freq": end_time+1,
        "cfl": 0.5,
        "flux_scheme": "Roe Exner",
        "integrator": "Forward Euler",
        "parNx": parNx,
        "parNy": parNy,
        "input_file": "input.nc",
        "output_file": "output.nc",
        "erosion": {
            "grass_factor": grass_factor
        },
        "boundaries": get_boundaries(Lx, Ly, dh)
    }

    with open(os.path.join(output_dir, "input.yaml"), "w") as f:
        yaml.dump(config, f, sort_keys=False, default_flow_style=None)

    # 3. Write distributed.slurm
    generate_job_script(output_dir, parNx, parNy, job_name, system="pbs")


if __name__ == "__main__":
    topologies = [
        (1, 1),    # 1 gpu
        (2, 1),    # 2 gpus
        (2, 2),    # 4 gpus
        (4, 2),    # 8 gpus
        (4, 4),    # 16 gpus
        (8, 4),    # 32 gpus
        (8, 8),    # 64 gpus
        (16, 8),   # 128 gpus
        (16, 16)   # 256 gpus
    ]

    base_dh = 0.1
    grass_factor = 0.005
    end_time = 1.0

    # ---------------------------
    # STRONG SCALING
    # ---------------------------
    # Target: keep global domain fixed.
    # We choose Lx = 200.0, Ly = 200.0 so total domain is 400x400 with dh=0.1
    # This means total grid is 4000x4000 = 16M cells.
    strong_scaling_Lx = 400.0
    strong_scaling_Ly = 400.0

    slurm_dirs = []

    print("Building Strong Scaling Cases...")
    for (parNx, parNy) in topologies:
        n_gpus = parNx * parNy
        out_dir = os.path.join("strong_scaling", f"gpus_{n_gpus}")
        job_name = f"strong_{n_gpus}"
        generate_case(
            out_dir, 
            Lx=strong_scaling_Lx, 
            Ly=strong_scaling_Ly, 
            dh=base_dh, 
            parNx=parNx, 
            parNy=parNy, 
            end_time=end_time, 
            grass_factor=grass_factor,
            job_name=job_name
        )
        slurm_dirs.append(out_dir)

    # ---------------------------
    # WEAK SCALING
    # ---------------------------
    # Target: keep local grid constant per GPU.
    # At 1 GPU, say we have Lx_local = 25.0, Ly_local = 25.0 for a local 500x500 cells (250K cells)
    # So for N GPUs, Lx = parNx * Lx_local, Ly = parNy * Ly_local
    weak_scaling_Lx_local = 25.0
    weak_scaling_Ly_local = 25.0

    print("Building Weak Scaling Cases...")
    for (parNx, parNy) in topologies:
        n_gpus = parNx * parNy
        out_dir = os.path.join("weak_scaling", f"gpus_{n_gpus}")
        job_name = f"weak_{n_gpus}"
        
        Lx = parNx * weak_scaling_Lx_local
        Ly = parNy * weak_scaling_Ly_local
        
        generate_case(
            out_dir, 
            Lx=Lx, 
            Ly=Ly, 
            dh=base_dh, 
            parNx=parNx, 
            parNy=parNy, 
            end_time=end_time, 
            grass_factor=grass_factor,
            job_name=job_name
        )
        slurm_dirs.append(out_dir)

    print("Writing submit_all.sh...")
    with open("submit_all.sh", "w") as f:
        f.write("#!/bin/bash\n\n")
        for d in slurm_dirs:
            f.write(f"cd {d}\n")
            f.write("sbatch distributed.slurm\n")
            f.write("cd - > /dev/null\n\n")
            
    os.chmod("submit_all.sh", 0o755)
