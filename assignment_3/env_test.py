"""Check mesh setup and Warp kernel compilation on the selected device."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from cloth_pbd import ClothSimulator, make_adjacency, make_grid, make_mesh_cloth


def main():
    parser = argparse.ArgumentParser(description="Check the Warp assignment starter")
    parser.add_argument("--device", default="cpu", help="Warp device, e.g. cpu or cuda:0")
    args = parser.parse_args()

    positions, faces, edges, inverse_mass, lengths = make_grid(5, 4)
    offsets, incident = make_adjacency(len(positions), edges)
    assert positions.shape == (20, 3)
    assert faces.shape == (24, 3)
    assert len(incident) == 2 * len(edges)
    assert offsets[-1] == len(incident)
    assert np.all(lengths > 0)
    assert np.array_equal(np.flatnonzero(inverse_mass == 0), [2])

    sim = ClothSimulator(nx=5, ny=4, device=args.device, iterations=2)
    sim.step()
    result = sim.positions.numpy()
    assert np.all(np.isfinite(result))
    assert np.allclose(result[2], positions[2])

    with TemporaryDirectory() as directory:
        mesh = Path(directory) / "patch.obj"
        mesh.write_text(
            "v -1 1 0\nv 1 1 0\nv -1 0 0\nv 1 0 0\n"
            "f 1 3 4 2 # quad\n", encoding="ascii",
        )
        _, mesh_faces, mesh_edges, mesh_mass, _ = make_mesh_cloth(mesh)
        assert mesh_faces.shape == (2, 3)
        assert len(mesh_edges) == 5
        assert np.array_equal(np.flatnonzero(mesh_mass == 0), [1])
        custom = ClothSimulator(mesh=mesh, device=args.device, iterations=1)
        custom.step()
        custom_result = custom.positions.numpy()
        assert np.all(np.isfinite(custom_result))
        assert np.allclose(custom_result[1], [1, 1, 0])
    print(f"Warp {args.device} environment check passed.")


if __name__ == "__main__":
    main()
