import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os

# Correct Kinematic Chain for HumanML3D (22 joints)
t2m_kinematic_chain = [
    [0, 1, 4, 7, 10],  # Left leg
    [0, 2, 5, 8, 11],  # Right leg
    [0, 3, 6, 9, 12, 15],  # Spine and Head
    [9, 13, 16, 18, 20],  # Left arm
    [9, 14, 17, 19, 21],  # Right arm
]


def render_motion_to_gif(data_path, motion_id, output_name=None):
    joints_file = os.path.join(data_path, f"{motion_id}.npy")
    if not os.path.exists(joints_file):
        print(f"Error: File {joints_file} not found.")
        return

    # Load joints (Frames, 22, 3)
    joints = np.load(joints_file)

    # --- CENTERING LOGIC ---
    # Subtract root (joint 0) X and Z positions to keep skeleton in center
    # We keep Y (height) as is to see jumping/squatting
    root_pos = joints[:, 0, :]  # (Frames, 3)
    joints[:, :, 0] -= root_pos[:, None, 0]  # Center X
    joints[:, :, 2] -= root_pos[:, None, 2]  # Center Z

    # --- SETUP PLOT ---
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # HumanML3D scales are roughly in meters, set a fixed box size
    radius = 1.5

    def update(frame):
        ax.clear()

        # Set stable limits
        ax.set_xlim3d([-radius, radius])
        ax.set_ylim3d([-radius, radius])
        ax.set_zlim3d([0, radius * 2])

        # Map labels to match HumanML3D: Y is Up
        ax.set_xlabel("X (Side)")
        ax.set_ylabel("Z (Forward)")
        ax.set_zlabel("Y (Up)")

        curr = joints[frame]

        for chain in t2m_kinematic_chain:
            # IMPORTANT: We map Data Y to Plot Z to keep the skeleton standing up
            x = curr[chain, 0]
            y = curr[chain, 2]  # Data Z -> Plot Y
            z = curr[chain, 1]  # Data Y -> Plot Z (Vertical)
            ax.plot(x, y, z, linewidth=3, marker="o", markersize=5)

        ax.view_init(elev=20, azim=-45)
        ax.set_title(f"{motion_id} - Frame {frame}")

    # Create Animation
    ani = FuncAnimation(fig, update, frames=len(joints), interval=50)

    # Save as GIF
    if output_name is None:
        output_name = f"{motion_id}.gif"

    print(f"Saving GIF to {output_name}...")
    writer = PillowWriter(fps=20)
    ani.save(output_name, writer=writer)
    plt.close()
    print("Done!")


# --- CONFIGURATION ---
# PATH_TO_DATA = "/Users/aleksandrrazin/work/research/3d/motion-latent-diffusion/results/mld/test_ft/samples_2025-12-24-16-04-18"
PATH_TO_DATA = "/Users/aleksandrrazin/work/research/3d/motion-latent-diffusion/results/mld/base_mld_train/samples_2025-12-24-18-28-33"
ID_TO_SHOW = "Example_50_batch0_0"

render_motion_to_gif(
    PATH_TO_DATA, ID_TO_SHOW, output_name="base_mld_train_Example_50_batch0_0.gif"
)
