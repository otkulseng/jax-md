"""Batched nequix structure relaxation over random MPtrj frames.

Many MPtrj frames are assembled into a single jraph disjoint-union graph and
relaxed together. The graph *is* the system: positions live in
``nodes["positions"]`` (cartesian), the per-structure cells in
``globals["cell"]``, and the neighbor list in ``senders``/``receivers``/
``edges["shifts"]``.

The neighbor list is rebuilt on the host with matscipy -- exact and identical
to nequix's training pipeline -- only when an atom drifts past the skin. Between
rebuilds every optimizer step runs as a jit-compiled block over a fixed-capacity
graph (padded with ``jraph.pad_with_graphs``), so it never recompiles.

The optimizer itself is the canonical ``jax_md.minimize.lbfgs``; this example
only supplies the energy and the (rebuildable) neighbor list. To run a
relaxation, implement ``jax_md.minimize.lbfgs_update`` and call ``relax``.
"""

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jraph
import matscipy.neighbours
import numpy as np
from ase.geometry import complete_cell
from ase.io import read
from jax_md import minimize, space
from nequix.calculator import from_pretrained
from nequix.data import atomic_numbers_to_indices

SEED = 0xC00FFEE
NUM_FRAMES = 3
MODEL_NAME = "nequix-mp-1"
MPTRJ_DIR = Path("data/mptrj")

MAX_STEPS = 200
MEMORY_SIZE = 10
FMAX = 0.05  # eV/A, convergence on the max per-atom force norm
SKIN = 1.0  # A, rebuild the neighbor list once an atom drifts this far
EDGE_CAPACITY_MULTIPLIER = 1.25


# --- random batch of frames --------------------------------------------------


def load_random_frames(num_frames: int, mptrj_dir: Path, seed: int) -> list:
    """One random frame from each of ``num_frames`` distinct random files."""
    file_paths = sorted(p for p in mptrj_dir.iterdir() if p.is_file())
    if len(file_paths) < num_frames:
        raise ValueError(f"Need >= {num_frames} files in {mptrj_dir}, found {len(file_paths)}")

    rng = np.random.default_rng(seed)
    frames = []
    for file_idx in rng.choice(len(file_paths), size=num_frames, replace=False):
        path = file_paths[int(file_idx)]
        trajectory = read(path, index=":")
        atoms = trajectory[int(rng.integers(len(trajectory)))]
        if not atoms.pbc.all():
            raise ValueError(f"Expected a fully periodic frame: {path.name}")
        frames.append(atoms)
    return frames


def unpack_frames(frames: list, atom_indices: dict[int, int]):
    """Concatenate frames into flat cartesian arrays for the disjoint union."""
    species = np.concatenate(
        [[atom_indices[z] for z in a.get_atomic_numbers()] for a in frames]
    ).astype(np.int32)
    positions = np.concatenate([a.get_positions() for a in frames]).astype(np.float32)
    cells = np.stack([np.asarray(a.cell.array) for a in frames]).astype(np.float32)
    n_node = np.array([len(a) for a in frames], dtype=np.int32)
    return species, positions, cells, n_node


# --- jraph disjoint-union graph + matscipy neighbor list ---------------------


def _matscipy_edges(positions, cells, n_node, cutoff):
    """Per-structure matscipy neighbor lists, concatenated with node offsets.

    Mirrors ``nequix.data.preprocess_graph`` (``"ijS"``: senders=dst,
    receivers=src), but spans the whole batch so it can be recomputed as atoms
    move. ``shifts`` are integer lattice offsets; the model forms the edge
    displacement as ``positions[senders] - positions[receivers] + shifts @ cell``.
    """
    offsets = np.concatenate([[0], np.cumsum(n_node)[:-1]])
    pbc = np.array([True, True, True])
    senders, receivers, shifts, n_edge = [], [], [], []
    for cell, start, count in zip(cells, offsets, n_node):
        src, dst, shift = matscipy.neighbours.neighbour_list(
            "ijS",
            positions=positions[start : start + count].astype(np.float64),
            cell=complete_cell(np.asarray(cell, np.float64)),
            pbc=pbc,
            cutoff=float(cutoff),
        )
        senders.append(dst + start)
        receivers.append(src + start)
        shifts.append(shift)
        n_edge.append(len(src))
    return (
        np.concatenate(senders).astype(np.int32),
        np.concatenate(receivers).astype(np.int32),
        np.concatenate(shifts).astype(np.float32),
        np.array(n_edge, dtype=np.int32),
    )


def _round_capacity(n_edge_total: int) -> int:
    raw = int(np.ceil(n_edge_total * EDGE_CAPACITY_MULTIPLIER))
    return ((raw + 63) // 64) * 64  # nearest 64 to avoid recompiles on rebuild


def make_graph(positions, species, cells, n_node, cutoff, edge_capacity=None):
    """Build the padded batched ``GraphsTuple`` consumed by the nequix model.

    ``edge_capacity`` is grown (and rounded up) whenever the real edge count
    would overflow it; pass the previous value back in to keep shapes -- and
    thus the jit cache -- stable across rebuilds.
    """
    senders, receivers, shifts, n_edge = _matscipy_edges(positions, cells, n_node, cutoff)

    total_edges = int(n_edge.sum())
    if edge_capacity is None or total_edges > edge_capacity:
        edge_capacity = _round_capacity(total_edges)

    graph = jraph.GraphsTuple(
        n_node=np.asarray(n_node, np.int32),
        n_edge=n_edge,
        nodes={"species": species.astype(np.int32), "positions": positions.astype(np.float32)},
        edges={"shifts": shifts},
        senders=senders,
        receivers=receivers,
        globals={"cell": cells.astype(np.float32)},
    )
    padded = jraph.pad_with_graphs(
        graph,
        n_node=int(positions.shape[0]) + 1,
        n_edge=edge_capacity,
        n_graph=len(n_node) + 1,
    )
    return jax.tree_util.tree_map(jnp.asarray, padded), edge_capacity


# --- energy ------------------------------------------------------------------


def make_energy_fn(model):
    """``energy_fn(positions, graph) -> total energy``, jit-compiled.

    Sums the per-structure energies over the graph padding mask. Because the
    graphs are disjoint and the padding graph is excluded from the sum, the
    gradient on each atom depends only on its own structure and padded-node
    forces are zero -- so ``jax_md.minimize`` gets correct forces from
    ``quantity.canonicalize_force`` with no manual masking.
    """

    @eqx.filter_jit
    def energy_fn(positions, graph):
        graph = graph._replace(nodes={**graph.nodes, "positions": positions})
        energies, _, _ = model(graph)
        mask = jraph.get_graph_padding_mask(graph)
        return jnp.sum(jnp.where(mask, energies, 0.0))

    return energy_fn


# --- driver ------------------------------------------------------------------


def relax(model, frames, atom_indices, cutoff, max_steps=MAX_STEPS, fmax=FMAX, skin=SKIN):
    """Relax a batch of frames together; return per-structure positions + energy."""
    species, positions, cells, n_node = unpack_frames(frames, atom_indices)
    n_real = positions.shape[0]

    graph, capacity = make_graph(positions, species, cells, n_node, cutoff)
    _, shift_fn = space.free()
    init_fn, apply_fn = minimize.lbfgs(make_energy_fn(model), shift_fn, memory_size=MEMORY_SIZE)
    apply_fn = eqx.filter_jit(apply_fn)

    state = init_fn(jnp.asarray(graph.nodes["positions"]), graph=graph)
    reference = np.asarray(state.position)

    for _ in range(max_steps):
        state = apply_fn(state, graph=graph)

        if np.linalg.norm(np.asarray(state.force)[:n_real], axis=1).max() < fmax:
            break

        position = np.asarray(state.position)
        if np.linalg.norm(position - reference, axis=1).max() > 0.5 * skin:
            graph, capacity = make_graph(position[:n_real], species, cells, n_node, cutoff, capacity)
            reference = position

    relaxed = np.split(np.asarray(state.position)[:n_real], np.cumsum(n_node)[:-1])
    return relaxed, float(state.energy)


def main():
    model, config = from_pretrained(MODEL_NAME, backend="jax", use_kernel=False)
    atom_indices = atomic_numbers_to_indices(config["atomic_numbers"])
    cutoff = config["cutoff"]

    frames = load_random_frames(NUM_FRAMES, MPTRJ_DIR, SEED)
    species, positions, cells, n_node = unpack_frames(frames, atom_indices)
    graph, capacity = make_graph(positions, species, cells, n_node, cutoff)

    energy_fn = make_energy_fn(model)
    energy = energy_fn(jnp.asarray(graph.nodes["positions"]), graph=graph)

    print(f"Batched {NUM_FRAMES} structures: {int(n_node.sum())} atoms, "
          f"{int(graph.n_edge[:-1].sum())} edges (capacity {capacity}, cutoff {cutoff} A)")
    print(f"  total energy {float(energy):.4f} eV")

    _, capacity = make_graph(positions, species, cells, n_node, cutoff, capacity)
    print(f"  neighbor list rebuilt (capacity {capacity})")
    print("Implement `jax_md.minimize.lbfgs_update`, then call "
          "`relax(model, frames, atom_indices, cutoff)`.")


if __name__ == "__main__":
    main()
