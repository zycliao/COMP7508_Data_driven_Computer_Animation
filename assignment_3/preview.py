"""Preview exported cloth OBJ frames or render them to GIF/MP4."""

import argparse
from pathlib import Path

import imageio_ffmpeg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def read_obj(path):
    vertices = []
    faces = []
    for line in path.read_text(encoding="ascii").splitlines():
        parts = line.split()
        if parts[:1] == ["v"]:
            vertices.append([float(value) for value in parts[1:4]])
        elif parts[:1] == ["f"]:
            faces.append([int(value) - 1 for value in parts[1:4]])
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def render_frames(input_dir, output=None, fps=50, frame_files=None):
    input_dir = Path(input_dir)
    output = Path(output) if output is not None else None
    if fps < 1:
        raise ValueError("fps must be positive")
    if output and output.suffix.lower() not in {".gif", ".mp4"}:
        raise ValueError("output must end in .gif or .mp4")

    files = list(frame_files) if frame_files is not None else sorted(input_dir.glob("frame_*.obj"))
    if not files:
        raise ValueError(f"no frame_*.obj files found in {input_dir}")
    frames = [read_obj(path)[0] for path in files]
    _, faces = read_obj(files[0])
    if not all(frame.shape == frames[0].shape for frame in frames):
        raise ValueError("all OBJ frames must have the same number of vertices")
    if not all(np.isfinite(frame).all() for frame in frames):
        raise ValueError("OBJ frames contain non-finite vertex positions")

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    mesh = Poly3DCollection(frames[0][faces], facecolor="#d66358",
                            edgecolor="#554d4a", linewidth=0.25, alpha=0.92)
    ax.add_collection3d(mesh)
    top = np.flatnonzero(np.isclose(frames[0][:, 1], frames[0][:, 1].max(),
                                       rtol=0, atol=1e-5))
    pin = top[np.argsort(frames[0][top, 0])[len(top) // 2]]
    ax.scatter(*frames[0][pin], color="black", s=24)
    bounds = np.concatenate(frames, axis=0)
    low = bounds.min(axis=0)
    high = bounds.max(axis=0)
    span = np.maximum(high - low, 1.0)
    center = (low + high) / 2
    for setter, coordinate in ((ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)):
        setter(center[coordinate] - 0.6 * span[coordinate],
               center[coordinate] + 0.6 * span[coordinate])
    ax.set_box_aspect(tuple(span))
    ax.view_init(elev=78, azim=-90)
    ax.set_axis_off()
    title = ax.set_title(files[0].stem)

    def update(index):
        mesh.set_verts(frames[index][faces])
        title.set_text(files[index].stem)
        return mesh, title

    animation = FuncAnimation(fig, update, frames=len(files),
                              interval=1000 / fps, blit=False)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.suffix.lower() == ".gif":
            writer = PillowWriter(fps=fps)
        else:
            plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
            writer = FFMpegWriter(fps=fps)
        animation.save(output, writer=writer)
        print(f"Wrote {output.resolve()}")
    else:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Preview Assignment 3 OBJ frames")
    parser.add_argument("--input", type=Path, default=Path("frames"))
    parser.add_argument("--output", type=Path, help="Optional .gif or .mp4 path")
    parser.add_argument("--fps", type=int, default=50)
    args = parser.parse_args()
    try:
        render_frames(args.input, args.output, args.fps)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
