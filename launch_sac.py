import subprocess
from dataclasses import dataclass

import tyro


@dataclass
class Args:
    gpu: str = "V100"
    n_cpus: int = 2
    mem: str = "4G"
    env: str = "Humanoid-v4"
    resets: bool = False


if __name__ == "__main__":
    args = tyro.cli(Args)

    if args.resets:
        n_reset_steps = 200000
        n_updates_per_step = 4
    else:
        n_reset_steps = int(1e10)  # no resets since we have 1e6 steps in total
        n_updates_per_step = 1

    seed_arr = [1, 2, 3, 4, 5]
    # n_steps = int(1e6)
    n_steps = int(0.5e6)

    resets_str = "resets"
    if not args.resets:
        resets_str = "no" + resets_str

    for seed in seed_arr:
        job_name = f"SAC_{args.env}_{resets_str}_{seed}"  # __{args.gpu}__{args.n_cpus}cpu__"
        command = [
            "sbatch",
            f"-J{job_name}",
            f"-o{job_name}-%j.out",
            "-N1",
            f"--export=ALL,ENV={args.env},SEED={seed},N_STEPS={n_steps},N_RESET_STEPS={n_reset_steps},N_UPDATES_PER_STEP={n_updates_per_step}",
            f"--gres=gpu:{args.gpu}:1",
            f"--ntasks-per-node={args.n_cpus}",
            f"--mem={args.mem}",
            "launch_sac.sbatch",
        ]
        print("will launch: ", command)
        subprocess.run(command)
