import logging
import os
import time
from builtins import ValueError
from multiprocessing.sharedctypes import Value
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import ConcatDataset, DataLoader

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
# from torchsummary import summary
from tqdm import tqdm

from mld.config import parse_args

# from mld.datasets.get_dataset import get_datasets
from mld.data.get_data import get_datasets
from mld.data.sampling import subsample, upsample
from mld.models.get_model import get_model
from mld.utils.logger import create_logger
from mld.data.humanml.scripts import motion_process as mp
from mld.data.humanml.common.skeleton import Skeleton
from mld.data.humanml.utils.paramUtil import t2m_raw_offsets, t2m_kinematic_chain


def normalize_control_pose(pose_seq, joints_num=22):
    mp.l_idx1, mp.l_idx2 = 5, 8
    mp.fid_r, mp.fid_l = [8, 11], [7, 10]
    mp.face_joint_indx = [2, 1, 17, 16]
    mp.n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
    mp.kinematic_chain = t2m_kinematic_chain
    tgt_skel = Skeleton(mp.n_raw_offsets, mp.kinematic_chain, "cpu")
    mp.tgt_offsets = tgt_skel.get_offsets_joints(torch.from_numpy(pose_seq[0]))
    feats, _, _, _ = mp.process_file(pose_seq, feet_thre=0.002)
    feats = torch.from_numpy(feats).float().unsqueeze(0)
    joints = mp.recover_from_ric(feats, joints_num).squeeze(0).cpu().numpy()
    return joints


# 3D stick-figure rendering for a single motion to GIF
t2m_kinematic_chain = [
    [0, 1, 4, 7, 10],
    [0, 2, 5, 8, 11],
    [0, 3, 6, 9, 12, 15],
    [9, 13, 16, 18, 20],
    [9, 14, 17, 19, 21],
]


def render_motion_to_gif(joints_np, out_path, fps=20):
    # joints_np: [T, 22, 3]
    if joints_np.shape[1] > 22:
        joints_np = joints_np[:, :22, :]

    # center on root in X/Z
    root = joints_np[:, 0, :]
    joints = joints_np.copy()
    joints[:, :, 0] -= root[:, None, 0]
    joints[:, :, 2] -= root[:, None, 2]

    # radius for consistent view
    extent = float(np.max(np.abs(joints))) if joints.size else 1.0
    radius = max(1.0, extent + 0.2)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    def update(frame):
        ax.clear()
        ax.set_xlim3d([-radius, radius])
        ax.set_ylim3d([-radius, radius])
        ax.set_zlim3d([0, radius * 2])
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_zlabel("Y")
        curr = joints[frame]
        for chain in t2m_kinematic_chain:
            x = curr[chain, 0]
            y = curr[chain, 2]  # forward
            z = curr[chain, 1]  # up
            ax.plot(x, y, z, linewidth=3, marker="o", markersize=5)
        ax.view_init(elev=20, azim=-45)
        ax.set_title(f"Frame {frame}")

    ani = FuncAnimation(fig, update, frames=len(joints), interval=50)
    writer = PillowWriter(fps=fps)
    ani.save(out_path, writer=writer)
    plt.close(fig)
    return out_path


def save_control_poses_image(hint_np, out_path):
    # hint_np: [2, J, 3]
    if hint_np.shape[1] > 22:
        hint_np = hint_np[:, :22, :]
    fig = plt.figure(figsize=(6, 3))
    titles = ["Start", "End"]
    for idx in range(2):
        ax = fig.add_subplot(1, 2, idx + 1, projection="3d")
        pose = hint_np[idx]
        pose = pose.copy()
        pose[:, 0] -= pose[0, 0]
        pose[:, 2] -= pose[0, 2]
        extent = float(np.max(np.abs(pose))) if pose.size else 1.0
        radius = max(1.0, extent + 0.2)
        for chain in t2m_kinematic_chain:
            x = pose[chain, 0]
            y = pose[chain, 2]
            z = pose[chain, 1]
            ax.plot(x, y, z, linewidth=3, marker="o", markersize=5)
        ax.set_xlim3d([-radius, radius])
        ax.set_ylim3d([-radius, radius])
        ax.set_zlim3d([0, radius * 2])
        ax.view_init(elev=20, azim=-45)
        ax.set_title(titles[idx])
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def main():
    """
    get input text
    ToDo skip if user input text in command
    current tasks:
         1 text 2 mtion
         2 motion transfer
         3 random sampling
         4 reconstruction

    ToDo
    1 use one funtion for all expoert
    2 fitting smpl and export fbx in this file
    3

    """
    # parse options
    cfg = parse_args(phase="demo")
    cfg.FOLDER = cfg.TEST.FOLDER
    cfg.Name = "demo--" + cfg.NAME
    logger = create_logger(cfg, phase="demo")

    if cfg.DEMO.EXAMPLE:
        # Check txt file input
        # load txt
        from mld.utils.demo_utils import load_example_input

        text, length = load_example_input(cfg.DEMO.EXAMPLE)
        task = "Example"
    elif cfg.DEMO.TASK:
        task = cfg.DEMO.TASK
        text = None
    else:
        # keyborad input
        task = "Keyborad_input"
        text = input("Please enter texts, none for random latent sampling:")
        length = input(
            "Please enter length, range 16~196, e.g. 50, none for random latent sampling:"
        )
        if text:
            motion_path = input(
                "Please enter npy_path for motion transfer, none for skip:"
            )
        # text 2 motion
        if text and not motion_path:
            cfg.DEMO.MOTION_TRANSFER = False
        # motion transfer
        elif text and motion_path:
            # load referred motion
            joints = np.load(motion_path)
            frames = subsample(
                len(joints),
                last_framerate=cfg.DEMO.FRAME_RATE,
                new_framerate=cfg.DATASET.KIT.FRAME_RATE,
            )
            joints_sample = torch.from_numpy(joints[frames]).float()

            features = model.transforms.joints2jfeats(joints_sample[None])
            motion = xx
            # datastruct = model.transforms.Datastruct(features=features).to(model.device)
            cfg.DEMO.MOTION_TRANSFER = True

        # default lengths
        length = 200 if not length else length
        length = [int(length)]
        text = [text]

    output_dir = Path(
        os.path.join(
            cfg.FOLDER, str(cfg.model.model_type), str(cfg.NAME), "samples_" + cfg.TIME
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # device options
    if cfg.ACCELERATOR == "gpu" and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in cfg.DEVICE)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # load dataset to extract nfeats dim of model
    dataset = get_datasets(cfg, logger=logger, phase="test")[0]

    # create mld model
    total_time = time.time()
    model = get_model(cfg, dataset)

    # ToDo
    # 1 choose task, input motion reference, text, lengths
    # 2 print task, input, output path
    #
    # logger.info(f"Input Text: {text}\nInput Length: {length}\nReferred Motion: {motion_path}")
    # random samlping
    if not text:
        logger.info(f"Begin specific task{task}")

    # debugging
    # vae
    # ToDo Remove this
    # temp loading
    # if cfg.TRAIN.PRETRAINED_VAE:
    #     logger.info("Loading pretrain vae from {}".format(cfg.TRAIN.PRETRAINED_VAE))
    #     ckpt = torch.load(cfg.TRAIN.PRETRAINED_VAE, map_location="cpu")
    #     model.load_state_dict(ckpt["state_dict"], strict=False)

    # /apdcephfs/share_1227775/shingxchen/AIMotion/TMOSTData/exps/actor/ACTOR_1010_vae_feats_kl/checkpoints/epoch=1599.ckpt

    # loading checkpoints
    logger.info("Loading checkpoints from {}".format(cfg.TEST.CHECKPOINTS))
    # state_dict = torch.load(cfg.TEST.CHECKPOINTS, map_location="cpu")["state_dict"]
    state_dict = torch.load(
        cfg.TEST.CHECKPOINTS, map_location="cpu", weights_only=False
    )["state_dict"]
    # # remove mismatched and unused params
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     old, new = "denoiser.decoder.0.", "denoiser.decoder."
    #     # old1, new1 = "text_encoder.text_model.text_model", "text_encoder.text_model.vision_model"
    #     old1 = "text_encoder.text_model.vision_model"
    #     if k[: len(old)] == old:
    #         name = k.replace(old, new)
    #     # elif k[: len(old)] == old:
    #     #     name = k.replace(old, new)
    #     else:
    #         name = k

    #     new_state_dict[name] = v
    #     # if k.split(".")[0] not in ["text_encoder", "denoiser"]:
    #     #     new_state_dict[k] = v
    # model.load_state_dict(new_state_dict, strict=False)

    model.load_state_dict(state_dict, strict=True)

    logger.info("model {} loaded".format(cfg.model.model_type))
    model.sample_mean = cfg.TEST.MEAN
    model.fact = cfg.TEST.FACT
    model.to(device)
    model.eval()

    mld_time = time.time()

    # sample
    with torch.no_grad():
        rep_lst = []
        rep_ref_lst = []
        texts_lst = []
        # task: input or Example
        if text:
            # prepare batch data
            batch = {"length": length, "text": text}
            if getattr(cfg.DEMO, "CONTROL_POSE", None):
                control_pose = np.load(cfg.DEMO.CONTROL_POSE)
                if control_pose.ndim == 3:
                    control_pose = control_pose[None, ...]
                if control_pose.shape[1] != 2 or control_pose.shape[-1] != 3:
                    raise ValueError(
                        "control_pose must have shape [2,J,3] or [B,2,J,3]"
                    )
                if control_pose.shape[0] == 1:
                    control_pose = np.repeat(control_pose, len(text), axis=0)
                raw_control_pose = torch.from_numpy(control_pose).float().to(device)
                normed = []
                for pose_seq in control_pose:
                    normed_seq = normalize_control_pose(pose_seq, joints_num=22)
                    normed.append(normed_seq[[0, -1]])
                control_pose = np.stack(normed, axis=0).astype(np.float32)
                batch["control_hint"] = (
                    torch.from_numpy(control_pose).float().to(device)
                )
                batch["control_hint_raw"] = raw_control_pose

            for rep in range(cfg.DEMO.REPLICATION):
                # text motion transfer
                if cfg.DEMO.MOTION_TRANSFER:
                    joints = model.forward_motion_style_transfer(batch)
                # text to motion synthesis
                else:
                    joints = model(batch)

                # cal inference time
                infer_time = time.time() - mld_time
                num_batch = 1
                num_all_frame = sum(batch["length"])
                num_ave_frame = sum(batch["length"]) / len(batch["length"])

                # upscaling to compare with other methods
                # joints = upsample(joints, cfg.DATASET.KIT.FRAME_RATE, cfg.DEMO.FRAME_RATE)
                nsample = len(joints)
                id = 0
                for i in range(nsample):
                    npypath = str(output_dir / f"{task}_{length[i]}_batch{id}_{i}.npy")
                    with open(npypath.replace(".npy", ".txt"), "w") as text_file:
                        text_file.write(batch["text"][i])
                    np.save(npypath, joints[i].detach().cpu().numpy())
                    logger.info(f"Motions are generated here:\n{npypath}")

                    # auto GIF export
                    ckpt_tag = os.path.basename(cfg.TEST.CHECKPOINTS) if hasattr(cfg, "TEST") else ""
                    ckpt_tag = ckpt_tag.replace(".ckpt", "").replace("epoch=", "e")
                    scale_tag = getattr(cfg.model, "controlnet_scale", None)
                    input_tag = os.path.splitext(os.path.basename(cfg.DEMO.EXAMPLE or ""))[0]
                    gif_name_parts = [
                        input_tag or "sample",
                        f"len{length[i]}",
                        ckpt_tag or "ckpt",
                    ]
                    if scale_tag is not None:
                        gif_name_parts.append(f"scale{scale_tag}")
                    gif_name_parts.append(f"{i}")
                    gif_name = "_".join(gif_name_parts) + ".gif"
                    gif_path = os.path.join(output_dir, gif_name)
                    try:
                        render_motion_to_gif(joints[i].detach().cpu().numpy(), gif_path, fps=cfg.DEMO.FRAME_RATE)
                        logger.info(f"GIF saved to {gif_path}")
                    except Exception as e:
                        logger.warning(f"Failed to save GIF {gif_path}: {e}")

                    # control pose preview and simple HTML report
                    pose_img_path = None
                    if "control_hint_raw" in batch:
                        try:
                            pose_img_path = os.path.join(output_dir, gif_name.replace(".gif", "_poses.png"))
                            save_control_poses_image(
                                batch["control_hint_raw"][i].detach().cpu().numpy(),
                                pose_img_path,
                            )
                        except Exception as e:
                            logger.warning(f"Failed to save control poses image: {e}")
                    elif "control_hint" in batch:
                        try:
                            pose_img_path = os.path.join(output_dir, gif_name.replace(".gif", "_poses.png"))
                            save_control_poses_image(
                                batch["control_hint"][i].detach().cpu().numpy(),
                                pose_img_path,
                            )
                        except Exception as e:
                            logger.warning(f"Failed to save control poses image: {e}")

                    report_path = os.path.join(output_dir, gif_name.replace(".gif", "_report.html"))
                    try:
                        with open(report_path, "w") as f:
                            f.write("<html><body>\n")
                            f.write(f"<h3>Text: {batch['text'][i]}</h3>\n")
                            f.write(f"<p>Length: {length[i]} | Checkpoint: {ckpt_tag} | Control scale: {scale_tag}</p>\n")
                            if pose_img_path and os.path.exists(pose_img_path):
                                f.write(f"<div><h4>Control Poses</h4><img src='{os.path.basename(pose_img_path)}' width='400'></div>\n")
                            f.write(f"<div><h4>Generated</h4><img src='{os.path.basename(gif_path)}' width='400'></div>\n")
                            f.write("</body></html>")
                        logger.info(f"Report saved to {report_path}")
                    except Exception as e:
                        logger.warning(f"Failed to save report {report_path}: {e}")

                if cfg.DEMO.OUTALL:
                    rep_lst.append(joints)
                    texts_lst.append(batch["text"])

            if cfg.DEMO.OUTALL:
                grouped_lst = []
                for n in range(nsample):
                    grouped_lst.append(
                        torch.cat([r[n][None] for r in rep_lst], dim=0)[None]
                    )
                combinedOut = torch.cat(grouped_lst, dim=0)
                try:
                    # save all motions
                    npypath = str(output_dir / f"{task}_{length[i]}_all.npy")

                    np.save(npypath, combinedOut.detach().cpu().numpy())
                    with open(npypath.replace("npy", "txt"), "w") as text_file:
                        for texts in texts_lst:
                            for text in texts:
                                text_file.write(text)
                                text_file.write("\n")
                    logger.info(
                        f"All reconstructed motions are generated here:\n{npypath}"
                    )
                except:
                    raise ValueError(
                        "Lengths of motions are different, so we cannot save all motions in one file."
                    )

        # random samlping
        if not text:
            if task == "random_sampling":
                # default text
                text = "random sampling"
                length = 196
                nsample, latent_dim = 500, 256
                batch = {
                    "latent": torch.randn(1, nsample, latent_dim, device=model.device),
                    "length": [int(length)] * nsample,
                }
                # vae random sampling
                joints = model.gen_from_latent(batch)

                # latent diffusion random sampling
                # for i in range(100):
                #     model.condition = 'text_uncond'
                #     joints = model(batch)

                num_batch, num_all_frame, num_ave_frame = 100, 100 * 196, 196
                infer_time = time.time() - mld_time

                # joints = joints.cpu().numpy()

                # upscaling to compare with other methods
                # joints = upsample(joints, cfg.DATASET.KIT.FRAME_RATE, cfg.DEMO.FRAME_RATE)
                for i in range(nsample):
                    npypath = output_dir / f"{text.split(' ')[0]}_{length}_{i}.npy"
                    np.save(npypath, joints[i].detach().cpu().numpy())
                    logger.info(f"Motions are generated here:\n{npypath}")

            elif task in ["reconstrucion", "text_motion"]:
                for rep in range(cfg.DEMO.REPLICATION):
                    logger.info(f"Replication {rep}")
                    joints_lst = []
                    ref_lst = []
                    for id, batch in enumerate(dataset.test_dataloader()):
                        if task == "reconstrucion":
                            # batch = dataset.collate_fn(batch)
                            batch["motion"] = batch["motion"].to(device)
                            length = batch["length"]
                            joints, joints_ref = model.recon_from_motion(batch)
                        elif task == "text_motion":
                            # del batch["motion"]
                            batch["motion"] = batch["motion"].to(device)
                            joints, joints_ref = model(batch, return_ref=True)

                        nsample = len(joints)
                        length = batch["length"]
                        for i in range(nsample):
                            npypath = str(
                                output_dir
                                / f"{task}_{length[i]}_batch{id}_{i}_{rep}.npy"
                            )
                            np.save(npypath, joints[i].detach().cpu().numpy())
                            # if exps == "text-motion":
                            np.save(
                                npypath.replace(".npy", "_ref.npy"),
                                joints_ref[i].detach().cpu().numpy(),
                            )
                            with open(
                                npypath.replace(".npy", ".txt"), "w"
                            ) as text_file:
                                text_file.write(batch["text"][i])
                            logger.info(
                                f"Reconstructed motions are generated here:\n{npypath}"
                            )

            else:
                raise ValueError(
                    f"Not support task {task}, only support random_sampling, reconstrucion, text_motion"
                )

        # ToDo fix time counting
        total_time = time.time() - total_time
        print(f"MLD Infer time - This/Ave batch: {infer_time/num_batch:.2f}")
        print(f"MLD Infer FPS - Total batch: {num_all_frame/infer_time:.2f}")
        print(f"MLD Infer time - This/Ave batch: {infer_time/num_batch:.2f}")
        print(f"MLD Infer FPS - Total batch: {num_all_frame/infer_time:.2f}")
        print(
            f"MLD Infer FPS - Running Poses Per Second: {num_ave_frame*infer_time/num_batch:.2f}"
        )
        print(f"MLD Infer FPS - {num_all_frame/infer_time:.2f}s")
        print(
            f"MLD Infer FPS - Running Poses Per Second: {num_ave_frame*infer_time/num_batch:.2f}"
        )

        # todo no num_batch!!!
        # num_batch=> num_forward
        print(
            f"MLD Infer FPS - time for 100 Poses: {infer_time/(num_batch*num_ave_frame)*100:.2f}"
        )
        print(
            f"Total time spent: {total_time:.2f} seconds (including model loading time and exporting time)."
        )

    if cfg.DEMO.RENDER:
        # plot with lines
        # from mld.data.humanml.utils.plot_script import plot_3d_motion
        # fig_path = Path(str(npypath).replace(".npy",".mp4"))
        # plot_3d_motion(fig_path, joints, title=text, fps=cfg.DEMO.FRAME_RATE)

        # single render
        # from mld.utils.demo_utils import render
        # figpath = render(npypath, cfg.DATASET.JOINT_TYPE,
        #                  cfg_path="./configs/render_cx.yaml")
        # logger.info(f"Motions are rendered here:\n{figpath}")

        from mld.utils.demo_utils import render_batch

        blenderpath = cfg.RENDER.BLENDER_PATH
        render_batch(
            os.path.dirname(npypath), execute_python=blenderpath, mode="sequence"
        )  # sequence
        logger.info(f"Motions are rendered here:\n{os.path.dirname(npypath)}")


if __name__ == "__main__":
    main()


# python demo.py --cfg configs/config_mld_controlnet_humanml3d.yaml
# python demo.py \
#   --cfg configs/config_mld_controlnet_humanml3d.yaml \
#   --example ./demo/example_input.txt \
#   --control_pose ./demo/control_pose.npy \
#   --out_dir ./results/demo_run
