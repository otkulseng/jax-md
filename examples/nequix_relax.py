from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from ase.io import read
from ase.optimize import LBFGS
from jax import config
from jax_md import minimize, quantity, space
from jax_md.custom_partition import estimate_max_neighbors_from_box
from nequix.calculator import NequixCalculator
from nequix.data import atomic_numbers_to_indices
from nequix.jax_md import nequix_neighbor_list

SEED = 0xC00FFEE
N_STEPS = 200
MEMORY_SIZE = 100
MODEL_NAME = "nequix-mp-1"
MPTRJ_DIR = Path("data/mptrj")
PLOT_PATH = Path("nequix_relax_convergence.png")


def load_random_atoms(seed: int, mptrj_dir: Path):
    if not mptrj_dir.exists():
        raise ValueError(
            f"{mptrj_dir} does not exist. Download mptrj data and place it there."
        )

    file_paths = sorted([p for p in mptrj_dir.iterdir() if p.is_file()])
    if not file_paths:
        raise ValueError(f"No files found in {mptrj_dir}")

    rng = np.random.default_rng(seed)
    file_path = file_paths[rng.integers(len(file_paths))]
    frames = read(file_path, index=":")
    frame_idx = int(rng.integers(len(frames)))
    atoms = frames[frame_idx]

    if not atoms.pbc.all():
        raise ValueError("Expected a fully periodic frame.")

    return atoms, file_path.name, frame_idx


def run_jax_lbfgs(atoms, model_name: str, n_steps: int, optimizer, optimizer_name: str):
    atoms = atoms.copy()
    config.update("jax_enable_x64", False)
    calc = NequixCalculator(model_name, backend="jax", use_kernel=False)
    config.update("jax_enable_x64", True)

    atom_indices = atomic_numbers_to_indices(calc.config["atomic_numbers"])
    species = jnp.array([atom_indices[z] for z in atoms.get_atomic_numbers()], dtype=jnp.int32)

    box = jnp.asarray(atoms.cell.array.T, dtype=jnp.float64)
    positions = jnp.asarray(atoms.get_scaled_positions(wrap=True), dtype=jnp.float64)

    displacement_fn, shift_fn = space.periodic_general(
        box, fractional_coordinates=True, wrapped=True
    )

    max_neighbors = estimate_max_neighbors_from_box(
        box, calc.cutoff, len(atoms), safety_factor=2.0
    )
    neighbor_fn, energy_fn = nequix_neighbor_list(
        displacement_fn,
        box,
        calc.model,
        species=species,
        max_neighbors=max_neighbors,
        fractional_coordinates=True,
    )

    init_fn, apply_fn = minimize.optax_descent(energy_fn, optimizer, neighbor_fn)
    apply_fn = jax.jit(apply_fn)
    force_fn = quantity.force(energy_fn)

    state = init_fn(positions)
    force_norms = []

    for _ in range(n_steps):
        state = apply_fn(state)
        force = force_fn(state.position, neighbor=state.neighbor_fn)
        force_norms.append(float(jax.device_get(jnp.linalg.norm(force))))

    print(f"Finished JAX run with {optimizer_name}: {len(force_norms)} steps")
    return np.asarray(force_norms)


def run_ase_lbfgs(atoms, model_name: str, n_steps: int):
    atoms = atoms.copy()
    config.update("jax_enable_x64", False)
    atoms.calc = NequixCalculator(model_name, backend="jax", use_kernel=False)
    config.update("jax_enable_x64", True)

    force_norms = []

    def record_force_norm():
        force_norms.append(float(np.linalg.norm(atoms.get_forces())))

    optimizer = LBFGS(atoms, logfile=None, memory=MEMORY_SIZE)
    optimizer.attach(record_force_norm, interval=1)
    optimizer.run(fmax=0.0, steps=n_steps)

    return np.asarray(force_norms)


def plot_convergence(jax_curves: dict[str, np.ndarray], ase_force_norms: np.ndarray, path: Path):
    eps = 1e-16

    plt.figure(figsize=(7, 4.5))
    for label, curve in jax_curves.items():
        steps = np.arange(1, len(curve) + 1)
        plt.plot(steps, np.maximum(curve, eps), label=f"OPTAX {label}")

    ase_steps = np.arange(1, len(ase_force_norms) + 1)
    plt.plot(ase_steps, np.maximum(ase_force_norms, eps), label="ASE LBFGS", linewidth=2.0, linestyle="--")

    plt.yscale("log")
    plt.xlabel("Step")
    plt.ylabel("||F||")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)


def main():
    atoms, file_name, frame_idx = load_random_atoms(SEED, MPTRJ_DIR)
    print(f"Loaded frame: {file_name}[{frame_idx}] with {len(atoms)} atoms")

    optax_optimizers = {
        "LBFGS": optax.lbfgs(memory_size=MEMORY_SIZE),
        "AdaGrad": optax.adagrad(1e-2),
    }

    jax_curves = {}
    for name, optimizer in optax_optimizers.items():
        jax_curves[name] = run_jax_lbfgs(atoms, MODEL_NAME, N_STEPS, optimizer, name)

    print("Starting ASE LBFGS")
    ase_force_norms = run_ase_lbfgs(atoms, MODEL_NAME, N_STEPS)

    for name, curve in jax_curves.items():
        print(f"{name} steps: {len(curve)}")
    print(f"ASE steps: {len(ase_force_norms)}")

    plot_convergence(jax_curves, ase_force_norms, PLOT_PATH)
    print(f"Saved convergence plot to {PLOT_PATH}")


if __name__ == "__main__":
    main()
