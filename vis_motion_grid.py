import argparse
import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from mld.data.humanml.utils import paramUtil


def draw_pose(ax, joints, colors, chains):
    ax.view_init(elev=120, azim=-90)
    ax.set_axis_off()

    joints = joints.copy()
    joints[:, 0] -= joints[0, 0]
    joints[:, 2] -= joints[0, 2]

    for chain, color in zip(chains, colors):
        ax.plot(
            joints[chain, 0],
            joints[chain, 1],
            joints[chain, 2],
            color=color,
            linewidth=2.0,
        )
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], s=6, c="black")


def _normalize_input(poses):
    if poses.ndim == 3 and poses.shape[2] == 3:
        return poses
    if poses.ndim == 3 and poses.shape[1] == 3:
        return poses.transpose(0, 2, 1)
    if poses.ndim == 2 and poses.shape[1] % 3 == 0:
        joints = poses.shape[1] // 3
        return poses.reshape(poses.shape[0], joints, 3)
    raise ValueError("input must be [T,J,3], [T,3,J], or [T,J*3]")


def plot_grid(poses, out_path, cols):
    matplotlib.use("Agg")
    poses = _normalize_input(poses)
    if poses.shape[1] < 22:
        raise ValueError("input must have at least 22 joints")
    if poses.shape[1] != 22:
        poses = poses[:, :22, :]

    colors = [
        "#DD5A37",
        "#D69E00",
        "#B75A39",
        "#DD5A37",
        "#D69E00",
        "#FF6D00",
        "#FF6D00",
        "#FF6D00",
        "#FF6D00",
        "#FF6D00",
        "#DDB50E",
        "#DDB50E",
        "#DDB50E",
        "#DDB50E",
        "#DDB50E",
    ]
    chains = paramUtil.t2m_kinematic_chain

    num_frames = poses.shape[0]
    rows = max(1, math.ceil(num_frames / cols))
    fig = plt.figure(figsize=(cols * 3, rows * 3))

    for idx in range(rows * cols):
        ax = fig.add_subplot(rows, cols, idx + 1, projection="3d")
        if idx < num_frames:
            draw_pose(ax, poses[idx], colors, chains)
            ax.set_title(f"t={idx}", fontsize=8)
        else:
            ax.set_axis_off()

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="npy file [T,22,3]")
    parser.add_argument("--output", required=True, help="output png path")
    parser.add_argument("--cols", type=int, default=8, help="grid columns")
    args = parser.parse_args()

    poses = np.load(args.input)
    plot_grid(poses, args.output, args.cols)


if __name__ == "__main__":
    main()
