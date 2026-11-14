"""Position-based cloth simulation with NVIDIA Warp."""

import argparse
from pathlib import Path

import numpy as np
import warp as wp


@wp.kernel
def predict_positions(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    inverse_mass: wp.array(dtype=float),
    predicted: wp.array(dtype=wp.vec3),
    dt: float,
    gravity: wp.vec3,
    damping: float,
):
    i = wp.tid()
    # TODO 1: Update velocity under gravity and damping, respecting fixed vertices.

    # TODO 2: Predict the next position.
    predicted[i] = positions[i]


@wp.kernel
def project_edges(
    predicted: wp.array(dtype=wp.vec3),
    inverse_mass: wp.array(dtype=float),
    edge_a: wp.array(dtype=wp.int32),
    edge_b: wp.array(dtype=wp.int32),
    rest_length: wp.array(dtype=float),
    correction_a: wp.array(dtype=wp.vec3),
    correction_b: wp.array(dtype=wp.vec3),
):
    e = wp.tid()
    a = edge_a[e]
    b = edge_b[e]
    # TODO 3: Evaluate this edge's distance constraint.

    # TODO 4: Compute an inverse-mass-weighted correction for each endpoint.
    correction_a[e] = wp.vec3(0.0, 0.0, 0.0)
    correction_b[e] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def gather_corrections(
    predicted: wp.array(dtype=wp.vec3),
    rest_positions: wp.array(dtype=wp.vec3),
    inverse_mass: wp.array(dtype=float),
    offsets: wp.array(dtype=wp.int32),
    incident: wp.array(dtype=wp.int32),
    correction_a: wp.array(dtype=wp.vec3),
    correction_b: wp.array(dtype=wp.vec3),
    next_predicted: wp.array(dtype=wp.vec3),
    alpha: float,
):
    i = wp.tid()
    if inverse_mass[i] == 0.0:
        next_predicted[i] = rest_positions[i]
        return

    total = wp.vec3(0.0, 0.0, 0.0)
    count = offsets[i + 1] - offsets[i]
    for k in range(offsets[i], offsets[i + 1]):
        endpoint = incident[k]
        edge = endpoint // 2
        if endpoint % 2 == 0:
            total += correction_a[edge]
        else:
            total += correction_b[edge]
    if count > 0:
        # TODO 5: Apply the gathered corrections with relaxation factor alpha.
        next_predicted[i] = predicted[i]
    else:
        next_predicted[i] = predicted[i]


@wp.kernel
def update_state(
    positions: wp.array(dtype=wp.vec3),
    predicted: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    inverse_mass: wp.array(dtype=float),
    dt: float,
):
    i = wp.tid()
    # TODO 6: Recover velocity from the corrected position.
    velocities[i] = wp.vec3(0.0, 0.0, 0.0)

    # TODO 7: Commit the corrected position, retaining fixed vertices.
    positions[i] = positions[i]


def read_obj_file(path):
    """Load vertices and triangulate polygon faces from an OBJ mesh."""
    vertices = []
    faces = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.partition("#")[0].split()
        if parts[:1] == ["v"]:
            vertices.append([float(value) for value in parts[1:4]])
        elif parts[:1] == ["f"]:
            indices = []
            for token in parts[1:]:
                value = int(token.split("/")[0])
                indices.append(value - 1 if value > 0 else len(vertices) + value)
            for j in range(1, len(indices) - 1):
                faces.append((indices[0], indices[j], indices[j + 1]))
    positions = np.asarray(vertices, dtype=np.float32)
    triangles = np.asarray(faces, dtype=np.int32)
    if positions.ndim != 2 or positions.shape[1] != 3 or triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("OBJ must contain 3D vertices and polygon faces")
    if not np.isfinite(positions).all():
        raise ValueError("OBJ contains a non-finite vertex")
    if np.any(triangles < 0) or np.any(triangles >= len(positions)):
        raise ValueError("OBJ face references a missing vertex")
    return positions, triangles


def make_topology(positions, faces, pin):
    """Build edge constraints and fix one selected vertex."""
    edges = sorted({tuple(sorted((int(face[j]), int(face[(j + 1) % 3]))))
                    for face in faces for j in range(3)})
    edges = np.asarray(edges, dtype=np.int32)
    inverse_mass = np.ones(len(positions), dtype=np.float32)
    inverse_mass[pin] = 0.0
    rest_length = np.linalg.norm(
        positions[edges[:, 0]] - positions[edges[:, 1]], axis=1,
    ).astype(np.float32)
    if np.any(rest_length == 0):
        raise ValueError("Mesh contains a zero-length edge")
    return positions, faces, edges, inverse_mass, rest_length


def make_mesh_cloth(path):
    positions, faces = read_obj_file(path)
    top = np.flatnonzero(np.isclose(positions[:, 1], positions[:, 1].max(), rtol=0, atol=1e-5))
    pin = int(top[np.argsort(positions[top, 0])[len(top) // 2]])
    return make_topology(positions, faces, pin)


def make_grid(nx, ny, width=4.0, height=3.0):
    """Create a vertical triangulated cloth pinned at its top midpoint."""
    if nx < 2 or ny < 2:
        raise ValueError("nx and ny must each be at least 2")
    positions = np.array(
        [[width * (x / (nx - 1) - 0.5), height * (1 - y / (ny - 1)), 0.0]
         for y in range(ny) for x in range(nx)], dtype=np.float32,
    )
    faces = []
    for y in range(ny - 1):
        for x in range(nx - 1):
            a = y * nx + x
            faces.extend(((a, a + nx, a + 1), (a + 1, a + nx, a + nx + 1)))
    faces = np.asarray(faces, dtype=np.int32)
    return make_topology(positions, faces, nx // 2)


def make_adjacency(vertex_count, edges):
    """CSR adjacency; each endpoint is encoded as 2 * edge_id + side."""
    buckets = [[] for _ in range(vertex_count)]
    for edge_id, (a, b) in enumerate(edges):
        buckets[int(a)].append(2 * edge_id)
        buckets[int(b)].append(2 * edge_id + 1)
    offsets = np.zeros(vertex_count + 1, dtype=np.int32)
    for i, bucket in enumerate(buckets):
        offsets[i + 1] = offsets[i] + len(bucket)
    incident = np.asarray([item for bucket in buckets for item in bucket], dtype=np.int32)
    return offsets, incident


class ClothSimulator:
    def __init__(self, nx=21, ny=17, device="cpu", dt=0.01,
                 iterations=64, alpha=0.8, damping=0.995, mesh=None):
        if dt <= 0 or iterations < 1 or not 0 < alpha <= 1 or not 0 <= damping <= 1:
            raise ValueError("Require dt > 0, iterations >= 1, 0 < alpha <= 1, 0 <= damping <= 1")
        wp.init()
        self.device = device
        self.dt = dt
        self.iterations = iterations
        self.alpha = alpha
        self.damping = damping
        if mesh is None:
            positions, self.faces, edges, inverse_mass, lengths = make_grid(nx, ny)
        else:
            positions, self.faces, edges, inverse_mass, lengths = make_mesh_cloth(mesh)
        offsets, incident = make_adjacency(len(positions), edges)
        self.vertex_count = len(positions)
        self.edge_count = len(edges)
        self.positions = wp.from_numpy(positions, dtype=wp.vec3, device=device)
        self.rest_positions = wp.from_numpy(positions, dtype=wp.vec3, device=device)
        self.velocities = wp.zeros(self.vertex_count, dtype=wp.vec3, device=device)
        self.predicted = wp.zeros(self.vertex_count, dtype=wp.vec3, device=device)
        self.scratch = wp.zeros(self.vertex_count, dtype=wp.vec3, device=device)
        self.inverse_mass = wp.from_numpy(inverse_mass, dtype=float, device=device)
        self.edge_a = wp.from_numpy(edges[:, 0].copy(), dtype=wp.int32, device=device)
        self.edge_b = wp.from_numpy(edges[:, 1].copy(), dtype=wp.int32, device=device)
        self.rest_length = wp.from_numpy(lengths, dtype=float, device=device)
        self.offsets = wp.from_numpy(offsets, dtype=wp.int32, device=device)
        self.incident = wp.from_numpy(incident, dtype=wp.int32, device=device)
        self.correction_a = wp.zeros(self.edge_count, dtype=wp.vec3, device=device)
        self.correction_b = wp.zeros(self.edge_count, dtype=wp.vec3, device=device)

    def step(self):
        wp.launch(predict_positions, dim=self.vertex_count,
                  inputs=[self.positions, self.velocities, self.inverse_mass,
                          self.predicted, self.dt, wp.vec3(0.0, -9.8, -3.0),
                          self.damping], device=self.device)
        for _ in range(self.iterations):
            wp.launch(project_edges, dim=self.edge_count,
                      inputs=[self.predicted, self.inverse_mass, self.edge_a,
                              self.edge_b, self.rest_length, self.correction_a,
                              self.correction_b], device=self.device)
            wp.launch(gather_corrections, dim=self.vertex_count,
                      inputs=[self.predicted, self.rest_positions,
                              self.inverse_mass, self.offsets, self.incident,
                              self.correction_a, self.correction_b,
                              self.scratch, self.alpha], device=self.device)
            self.predicted, self.scratch = self.scratch, self.predicted
        wp.launch(update_state, dim=self.vertex_count,
                  inputs=[self.positions, self.predicted, self.velocities,
                          self.inverse_mass, self.dt], device=self.device)

    def save_obj(self, path):
        positions = self.positions.numpy()
        with open(path, "w", encoding="ascii") as out:
            for x, y, z in positions:
                out.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
            for a, b, c in self.faces:
                out.write(f"f {a + 1} {b + 1} {c + 1}\n")


def main():
    parser = argparse.ArgumentParser(description="Warp PBD cloth simulation")
    parser.add_argument("--device", default="cpu", help="Warp device, e.g. cpu or cuda:0")
    parser.add_argument("--nx", type=int, default=21)
    parser.add_argument("--ny", type=int, default=17)
    parser.add_argument("--mesh", type=Path, help="Optional cloth OBJ mesh")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--export-every", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=64)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--damping", type=float, default=0.995)
    parser.add_argument("--output", type=Path, default=Path("frames"))
    parser.add_argument("--video", type=Path, default=Path("cloth.mp4"),
                        help="Video output path (default: cloth.mp4)")
    parser.add_argument("--no-video", action="store_true", help="Export OBJ frames only")
    parser.add_argument("--fps", type=int, help="Video frame rate; defaults to simulation time")
    args = parser.parse_args()
    if args.steps < 0 or args.export_every < 1 or args.dt <= 0:
        parser.error("steps must be nonnegative, export-every positive, and dt > 0")
    if args.fps is not None and args.fps < 1:
        parser.error("fps must be positive")
    if not args.no_video and args.video.suffix.lower() not in {".gif", ".mp4"}:
        parser.error("video path must end in .gif or .mp4")
    sim = ClothSimulator(nx=args.nx, ny=args.ny, mesh=args.mesh, device=args.device,
                         dt=args.dt, iterations=args.iterations,
                         alpha=args.alpha, damping=args.damping)
    args.output.mkdir(parents=True, exist_ok=True)
    first_frame = args.output / "frame_0000.obj"
    sim.save_obj(first_frame)
    written_frames = [first_frame]
    for step in range(1, args.steps + 1):
        sim.step()
        if step % args.export_every == 0 or step == args.steps:
            frame = args.output / f"frame_{step:04d}.obj"
            sim.save_obj(frame)
            written_frames.append(frame)
    print(f"Wrote OBJ frames to {args.output.resolve()}")
    if not args.no_video:
        from preview import render_frames

        fps = args.fps or max(1, round(1.0 / (args.dt * args.export_every)))
        render_frames(args.output, args.video, fps, frame_files=written_frames)


if __name__ == "__main__":
    main()
