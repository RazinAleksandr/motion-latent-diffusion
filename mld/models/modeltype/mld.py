import inspect
import os
from os.path import join as pjoin
import time
import math

import numpy as np
import imageio
import torch
from torch import Tensor
from torch.optim import AdamW
from torchmetrics import MetricCollection

from mld.config import instantiate_from_config
from mld.data.humanml.utils.plot_script import plot_3d_motion
from mld.models.architectures import (
    mld_denoiser,
    mld_vae,
    vposert_vae,
    t2m_motionenc,
    t2m_textenc,
    vposert_vae,
)
from mld.models.losses.mld import MLDLosses
from mld.models.modeltype.base import BaseModel
from mld.utils.temos_utils import remove_padding

from .base import BaseModel


class MLD(BaseModel):
    """
    Stage 1 vae
    Stage 2 diffusion
    """

    def __init__(self, cfg, datamodule, **kwargs):
        super().__init__()

        self.cfg = cfg

        self.stage = cfg.TRAIN.STAGE
        self.condition = cfg.model.condition
        self.is_vae = cfg.model.vae
        self.predict_epsilon = cfg.TRAIN.ABLATION.PREDICT_EPSILON
        self.nfeats = cfg.DATASET.NFEATS
        self.njoints = cfg.DATASET.NJOINTS
        self.debug = cfg.DEBUG
        self.latent_dim = cfg.model.latent_dim
        self.guidance_scale = cfg.model.guidance_scale
        self.guidance_uncodp = cfg.model.guidance_uncondp
        self.datamodule = datamodule

        try:
            self.vae_type = cfg.model.vae_type
        except:
            self.vae_type = (
                cfg.model.motion_vae.target.split(".")[-1].lower().replace("vae", "")
            )

        self.text_encoder = instantiate_from_config(cfg.model.text_encoder)

        if self.vae_type != "no":
            self.vae = instantiate_from_config(cfg.model.motion_vae)

        # Don't train the motion encoder and decoder
        if self.stage == "diffusion":
            if self.vae_type in ["mld", "vposert", "actor"]:
                self.vae.training = False
                for p in self.vae.parameters():
                    p.requires_grad = False
            elif self.vae_type == "no":
                pass
            else:
                self.motion_encoder.training = False
                for p in self.motion_encoder.parameters():
                    p.requires_grad = False
                self.motion_decoder.training = False
                for p in self.motion_decoder.parameters():
                    p.requires_grad = False

        self.denoiser = instantiate_from_config(cfg.model.denoiser)
        self.is_controlnet = getattr(cfg.model, "is_controlnet", False)
        if self.is_controlnet:
            self.controlnet = instantiate_from_config(cfg.model.controlnet)
            self.controlnet_cond_encoder = instantiate_from_config(
                cfg.model.controlnet_cond_encoder
            )
            if getattr(cfg.model, "controlnet_train_only", False):
                for p in self.denoiser.parameters():
                    p.requires_grad = False
                for p in self.text_encoder.parameters():
                    p.requires_grad = False
        if not self.predict_epsilon:
            cfg.model.scheduler.params["prediction_type"] = "sample"
            cfg.model.noise_scheduler.params["prediction_type"] = "sample"
        self.scheduler = instantiate_from_config(cfg.model.scheduler)
        self.noise_scheduler = instantiate_from_config(cfg.model.noise_scheduler)

        if self.condition in ["text", "text_uncond"]:
            self._get_t2m_evaluator(cfg)

        if cfg.TRAIN.OPTIM.TYPE.lower() == "adamw":
            self.optimizer = AdamW(lr=cfg.TRAIN.OPTIM.LR, params=self.parameters())
        else:
            raise NotImplementedError("Do not support other optimizer for now.")

        if cfg.LOSS.TYPE == "mld":
            self._losses = MetricCollection(
                {
                    split: MLDLosses(vae=self.is_vae, mode="xyz", cfg=cfg)
                    for split in ["losses_train", "losses_test", "losses_val"]
                }
            )
        else:
            raise NotImplementedError("MotionCross model only supports mld losses.")

        self.losses = {
            key: self._losses["losses_" + key] for key in ["train", "test", "val"]
        }

        self.metrics_dict = cfg.METRIC.TYPE
        self.configure_metrics()
        self._vis_cache = {}
        self._has_moviepy = None

        # If we want to override it at testing time
        self.sample_mean = False
        self.fact = None
        self.do_classifier_free_guidance = self.guidance_scale > 1.0
        if self.condition in ["text", "text_uncond"]:
            self.feats2joints = datamodule.feats2joints
        elif self.condition == "action":
            self.rot2xyz = Rotation2xyz(smpl_path=cfg.DATASET.SMPL_PATH)
            self.feats2joints_eval = lambda sample, mask: self.rot2xyz(
                sample.view(*sample.shape[:-1], 6, 25).permute(0, 3, 2, 1),
                mask=mask,
                pose_rep="rot6d",
                glob=True,
                translation=True,
                jointstype="smpl",
                vertstrans=True,
                betas=None,
                beta=0,
                glob_rot=None,
                get_rotations_back=False,
            )
            self.feats2joints = lambda sample, mask: self.rot2xyz(
                sample.view(*sample.shape[:-1], 6, 25).permute(0, 3, 2, 1),
                mask=mask,
                pose_rep="rot6d",
                glob=True,
                translation=True,
                jointstype="vertices",
                vertstrans=False,
                betas=None,
                beta=0,
                glob_rot=None,
                get_rotations_back=False,
            )

    def _draw_pose_to_axis(self, ax, joints, color=None):
        from mld.data.humanml.utils import paramUtil

        chains = paramUtil.t2m_kinematic_chain
        colors = (
            [color] * len(chains)
            if color is not None
            else ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"] * 3
        )

        ax.view_init(elev=20, azim=-90)
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
        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], s=5, c="black")
        # dynamic bounds for better visibility
        extent = np.max(np.abs(joints)) if joints.size else 1.0
        radius = max(1.0, extent + 0.1)
        ax.set_xlim3d([-radius, radius])
        ax.set_ylim3d([-radius, radius])
        ax.set_zlim3d([-radius, radius])

    def _save_motion_grid(self, motion_pred, motion_ref, title, save_path, cols=8):
        import matplotlib.pyplot as plt

        total_frames = motion_pred.shape[0]
        if total_frames == 0:
            return None
        indices = np.linspace(0, total_frames - 1, cols, dtype=int)
        sampled_pred = motion_pred[indices]
        sampled_ref = motion_ref[indices] if motion_ref is not None else None

        # Error-based color for predicted row
        pred_color = None
        if sampled_ref is not None:
            err = np.linalg.norm(sampled_pred - sampled_ref, axis=-1).mean()
            if err < 0.05:
                pred_color = "#228B22"  # green
            elif err < 0.1:
                pred_color = "#FFD700"  # yellow
            else:
                pred_color = "#B22222"  # red

        fig = plt.figure(figsize=(cols * 2, 4))
        for idx in range(cols):
            # Row 0: GT (dark green)
            ax_gt = fig.add_subplot(2, cols, idx + 1, projection="3d")
            if sampled_ref is not None:
                self._draw_pose_to_axis(ax_gt, sampled_ref[idx], color="#006400")
            # Row 1: Pred (color-coded by error)
            ax_pd = fig.add_subplot(2, cols, cols + idx + 1, projection="3d")
            self._draw_pose_to_axis(ax_pd, sampled_pred[idx], color=pred_color)

        plt.suptitle(title, fontsize=10)
        plt.tight_layout()
        plt.savefig(save_path, dpi=100)
        fig.canvas.draw()
        # Use RGBA buffer to avoid size mismatch
        buf = np.asarray(fig.canvas.buffer_rgba())
        # Convert RGBA to RGB
        data = buf[..., :3].copy()
        plt.close(fig)
        return data

    def _save_pose_axes(
        self, motion_pred, motion_ref, frames_idx, title, save_path=None, rows=1
    ):
        """
        Log a small set of frames as 3D stick figures on separate axes.
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        from mld.data.humanml.utils import paramUtil

        if motion_pred is None or len(frames_idx) == 0:
            return None

        chains = paramUtil.t2m_kinematic_chain
        chain_colors = ["magenta", "green", "black", "blue", "red"]
        rows = max(1, rows)
        cols = int(np.ceil(len(frames_idx) / rows))

        # collect bounds across all frames for consistent axes
        all_pts = []
        for f in frames_idx:
            all_pts.append(motion_pred[f])
            if motion_ref is not None:
                all_pts.append(motion_ref[f])
        all_pts = np.concatenate(all_pts, axis=0) if all_pts else None
        global_extent = np.max(np.abs(all_pts)) if all_pts is not None else 1.0
        radius = max(1.0, float(global_extent) + 0.1)

        fig = plt.figure(figsize=(4 * cols, 4 * rows))
        for idx, f in enumerate(frames_idx):
            ax = fig.add_subplot(rows, cols, idx + 1, projection="3d")
            frame_pred = motion_pred[f]
            root = frame_pred[0]
            frame_pred = frame_pred.copy()
            frame_pred[:, 0] -= root[0]
            frame_pred[:, 2] -= root[2]

            if motion_ref is not None:
                frame_ref = motion_ref[f].copy()
                frame_ref[:, 0] -= frame_ref[0, 0]
                frame_ref[:, 2] -= frame_ref[0, 2]
            else:
                frame_ref = None

            for chain, color in zip(chains, chain_colors):
                ax.plot(
                    frame_pred[chain, 0],
                    frame_pred[chain, 2],
                    frame_pred[chain, 1],
                    color=color,
                    linewidth=3,
                    marker="o",
                    markersize=4,
                    label="pred" if idx == 0 else None,
                )
                if frame_ref is not None:
                    ax.plot(
                        frame_ref[chain, 0],
                        frame_ref[chain, 2],
                        frame_ref[chain, 1],
                        color="gray",
                        linewidth=2,
                        linestyle="--",
                        marker="o",
                        markersize=3,
                        alpha=0.6,
                        label="gt" if idx == 0 else None,
                    )

            ax.text2D(
                0.02,
                0.9,
                f"f={f}",
                transform=ax.transAxes,
                fontsize=8,
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
            )
            ax.set_xlim3d([-radius, radius])
            ax.set_ylim3d([-radius, radius])
            ax.set_zlim3d([-radius, radius])
            ax.view_init(elev=20, azim=-45)
            ax.grid(True, linestyle="--", alpha=0.3)
        if len(frames_idx) > 0:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc="upper right")
        fig.suptitle(title, fontsize=11)
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, dpi=120)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        data = buf[..., :3].copy()
        plt.close(fig)
        return data

    def _render_pose_video(
        self,
        motion_pred,
        motion_ref=None,
        title="",
        max_frames=64,
        fps=12,
        save_path=None,
    ):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        from mld.data.humanml.utils import paramUtil

        if motion_pred is None or len(motion_pred) == 0:
            return None

        T = len(motion_pred)
        frames = list(range(min(T, max_frames)))
        chains = paramUtil.t2m_kinematic_chain
        chain_colors = ["magenta", "green", "black", "blue", "red"]

        # bounds over all frames (pred + gt if present)
        pts = [motion_pred]
        if motion_ref is not None:
            pts.append(motion_ref)
        all_pts = np.concatenate(pts, axis=0)
        extent = np.max(np.abs(all_pts)) if all_pts.size else 1.0
        radius = max(1.0, float(extent) + 0.1)

        frame_images = []
        for f in frames:
            fig = plt.figure(figsize=(4, 4))
            ax = fig.add_subplot(111, projection="3d")

            frame_pred = motion_pred[f].copy()
            root = frame_pred[0]
            frame_pred[:, 0] -= root[0]
            frame_pred[:, 2] -= root[2]

            frame_ref = None
            if motion_ref is not None:
                frame_ref = motion_ref[f].copy()
                frame_ref[:, 0] -= frame_ref[0, 0]
                frame_ref[:, 2] -= frame_ref[0, 2]

            for chain, color in zip(chains, chain_colors):
                ax.plot(
                    frame_pred[chain, 0],
                    frame_pred[chain, 2],
                    frame_pred[chain, 1],
                    color=color,
                    linewidth=3,
                    marker="o",
                    markersize=4,
                )
                if frame_ref is not None:
                    ax.plot(
                        frame_ref[chain, 0],
                        frame_ref[chain, 2],
                        frame_ref[chain, 1],
                        color="gray",
                        linewidth=2,
                        linestyle="--",
                        marker="o",
                        markersize=3,
                        alpha=0.6,
                    )

            ax.text2D(
                0.02,
                0.9,
                f"f={f}",
                transform=ax.transAxes,
                fontsize=8,
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
            )
            ax.set_xlim3d([-radius, radius])
            ax.set_ylim3d([-radius, radius])
            ax.set_zlim3d([-radius, radius])
            ax.view_init(elev=20, azim=-45)
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.set_title(title, fontsize=9)
            plt.tight_layout()
            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            frame_images.append(buf)
            plt.close(fig)

        if len(frame_images) == 0:
            return None
        video_np = np.stack(frame_images)  # [T, H, W, 3]
        if save_path is not None:
            try:
                imageio.mimsave(save_path, video_np, fps=fps)
            except Exception:
                pass
        video_tensor = (
            torch.from_numpy(video_np).permute(0, 3, 1, 2).unsqueeze(0).float()
            / 255.0
        )  # [1, T, C, H, W]
        return video_tensor, video_np

    def _build_mesh_cloud(self, motion, motion_ref=None, max_frames=4):
        """
        Prepare a multi-frame mesh cloud for TensorBoard Mesh logging.
        Returns (verts, faces, colors).
        """
        from mld.data.humanml.utils import paramUtil

        def _normalize(arr):
            if arr is None:
                return None
            arr = np.asarray(arr)
            if arr.ndim == 3 and arr.shape[-1] == 3:
                return arr
            if arr.ndim == 3 and arr.shape[1] == 3:
                return arr.transpose(0, 2, 1)
            if arr.ndim == 2 and arr.shape[-1] % 3 == 0:
                j = arr.shape[-1] // 3
                return arr.reshape(arr.shape[0], j, 3)
            return None

        motion = _normalize(motion)
        motion_ref = _normalize(motion_ref)

        if motion is None or motion.ndim != 3 or motion.shape[-1] != 3:
            return None
        T, J, _ = motion.shape
        if T == 0 or J == 0:
            return None

        frames_idx = np.linspace(0, T - 1, min(max_frames, T), dtype=int)
        chains = paramUtil.t2m_kinematic_chain
        verts_list = []
        faces = []
        colors_list = []

        for f_local, f_idx in enumerate(frames_idx):
            frame = motion[f_idx].copy()
            # center on root in X/Z for cleaner display
            frame[:, 0] -= frame[0, 0]
            frame[:, 2] -= frame[0, 2]
            base = len(verts_list)
            verts_list.extend(frame.tolist())

            # optional error colors per joint
            err = None
            if motion_ref is not None and motion_ref.shape == motion.shape:
                ref_frame = motion_ref[f_idx].copy()
                ref_frame[:, 0] -= ref_frame[0, 0]
                ref_frame[:, 2] -= ref_frame[0, 2]
                err = np.linalg.norm(frame - ref_frame, axis=-1)
                if np.any(err):
                    err = err / (err.max() + 1e-8)
            if err is not None:
                colors_frame = np.repeat(err.reshape(-1, 1), 3, axis=1)
            else:
                colors_frame = np.ones((J, 3)) * 0.3
            colors_list.extend(colors_frame.tolist())

            for chain in chains:
                for a, b in zip(chain[:-1], chain[1:]):
                    va = base + a
                    vb = base + b
                    mid = (frame[a] + frame[b]) / 2.0
                    mid[2] += 5e-3  # lift a bit so edges are visible
                    verts_list.append(mid.tolist())
                    vc = len(verts_list) - 1
                    faces.append([va, vb, vc])

                    # color for mid-point
                    if err is not None:
                        mean_err = (err[a] + err[b]) / 2.0
                        colors_list.append([mean_err, mean_err, mean_err])
                    else:
                        colors_list.append([0.3, 0.3, 0.3])

        verts = torch.tensor(verts_list, dtype=torch.float32)
        faces = torch.tensor(faces, dtype=torch.int64)
        colors = torch.tensor(colors_list, dtype=torch.float32)
        return verts, faces, colors

    def _get_t2m_evaluator(self, cfg):
        """
        load T2M text encoder and motion encoder for evaluating
        """
        # init module
        self.t2m_textencoder = t2m_textenc.TextEncoderBiGRUCo(
            word_size=cfg.model.t2m_textencoder.dim_word,
            pos_size=cfg.model.t2m_textencoder.dim_pos_ohot,
            hidden_size=cfg.model.t2m_textencoder.dim_text_hidden,
            output_size=cfg.model.t2m_textencoder.dim_coemb_hidden,
        )

        self.t2m_moveencoder = t2m_motionenc.MovementConvEncoder(
            input_size=cfg.DATASET.NFEATS - 4,
            hidden_size=cfg.model.t2m_motionencoder.dim_move_hidden,
            output_size=cfg.model.t2m_motionencoder.dim_move_latent,
        )

        self.t2m_motionencoder = t2m_motionenc.MotionEncoderBiGRUCo(
            input_size=cfg.model.t2m_motionencoder.dim_move_latent,
            hidden_size=cfg.model.t2m_motionencoder.dim_motion_hidden,
            output_size=cfg.model.t2m_motionencoder.dim_motion_latent,
        )
        # load pretrianed
        dataname = cfg.TEST.DATASETS[0]
        dataname = "t2m" if dataname == "humanml3d" else dataname
        t2m_checkpoint = torch.load(
            os.path.join(
                cfg.model.t2m_path, dataname, "text_mot_match/model/finest.tar"
            ),
            map_location=torch.device("cpu"),
            weights_only=False,  # TODO added
        )
        self.t2m_textencoder.load_state_dict(t2m_checkpoint["text_encoder"])
        self.t2m_moveencoder.load_state_dict(t2m_checkpoint["movement_encoder"])
        self.t2m_motionencoder.load_state_dict(t2m_checkpoint["motion_encoder"])

        # freeze params
        self.t2m_textencoder.eval()
        self.t2m_moveencoder.eval()
        self.t2m_motionencoder.eval()
        for p in self.t2m_textencoder.parameters():
            p.requires_grad = False
        for p in self.t2m_moveencoder.parameters():
            p.requires_grad = False
        for p in self.t2m_motionencoder.parameters():
            p.requires_grad = False

    def sample_from_distribution(
        self,
        dist,
        *,
        fact=None,
        sample_mean=False,
    ) -> Tensor:
        fact = fact if fact is not None else self.fact
        sample_mean = sample_mean if sample_mean is not None else self.sample_mean

        if sample_mean:
            return dist.loc.unsqueeze(0)

        # Reparameterization trick
        if fact is None:
            return dist.rsample().unsqueeze(0)

        # Resclale the eps
        eps = dist.rsample() - dist.loc
        z = dist.loc + fact * eps

        # add latent size
        z = z.unsqueeze(0)
        return z

    def forward(self, batch):
        texts = batch["text"]
        lengths = batch["length"]
        if self.cfg.TEST.COUNT_TIME:
            self.starttime = time.time()

        if self.stage in ["diffusion", "vae_diffusion"]:
            # diffusion reverse
            if self.do_classifier_free_guidance:
                uncond_tokens = [""] * len(texts)
                if self.condition == "text":
                    uncond_tokens.extend(texts)
                elif self.condition == "text_uncond":
                    uncond_tokens.extend(uncond_tokens)
                texts = uncond_tokens
            text_emb = self.text_encoder(texts)
            controlnet_cond = self._get_control_cond(batch, lengths)
            z = self._diffusion_reverse(
                text_emb, lengths, controlnet_cond=controlnet_cond
            )
        elif self.stage in ["vae"]:
            motions = batch["motion"]
            z, dist_m = self.vae.encode(motions, lengths)

        with torch.no_grad():
            # ToDo change mcross actor to same api
            if self.vae_type in ["mld", "actor"]:
                feats_rst = self.vae.decode(z, lengths)
            elif self.vae_type == "no":
                feats_rst = z.permute(1, 0, 2)

        if self.cfg.TEST.COUNT_TIME:
            self.endtime = time.time()
            elapsed = self.endtime - self.starttime
            self.times.append(elapsed)
            if len(self.times) % 100 == 0:
                meantime = np.mean(self.times[-100:]) / self.cfg.TEST.BATCH_SIZE
                print(
                    f"100 iter mean Time (batch_size: {self.cfg.TEST.BATCH_SIZE}): {meantime}",
                )
            if len(self.times) % 1000 == 0:
                meantime = np.mean(self.times[-1000:]) / self.cfg.TEST.BATCH_SIZE
                print(
                    f"1000 iter mean Time (batch_size: {self.cfg.TEST.BATCH_SIZE}): {meantime}",
                )
                with open(pjoin(self.cfg.FOLDER_EXP, "times.txt"), "w") as f:
                    for line in self.times:
                        f.write(str(line))
                        f.write("\n")
        joints = self.feats2joints(feats_rst.detach().cpu())
        return remove_padding(joints, lengths)

    def gen_from_latent(self, batch):
        z = batch["latent"]
        lengths = batch["length"]

        feats_rst = self.vae.decode(z, lengths)

        # feats => joints
        joints = self.feats2joints(feats_rst.detach().cpu())
        return remove_padding(joints, lengths)

    def recon_from_motion(self, batch):
        feats_ref = batch["motion"]
        length = batch["length"]

        z, dist = self.vae.encode(feats_ref, length)
        feats_rst = self.vae.decode(z, length)

        # feats => joints
        joints = self.feats2joints(feats_rst.detach().cpu())
        joints_ref = self.feats2joints(feats_ref.detach().cpu())
        return remove_padding(joints, length), remove_padding(joints_ref, length)

    def _diffusion_reverse(
        self, encoder_hidden_states, lengths=None, controlnet_cond=None, generator=None
    ):
        # init latents
        bsz = encoder_hidden_states.shape[0]
        if self.do_classifier_free_guidance:
            bsz = bsz // 2
        if self.vae_type == "no":
            assert (
                lengths is not None
            ), "no vae (diffusion only) need lengths for diffusion"
            latents = torch.randn(
                (bsz, max(lengths), self.cfg.DATASET.NFEATS),
                device=encoder_hidden_states.device,
                dtype=torch.float,
                generator=generator,
            )
        else:
            latents = torch.randn(
                (bsz, self.latent_dim[0], self.latent_dim[-1]),
                device=encoder_hidden_states.device,
                dtype=torch.float,
                generator=generator,
            )

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        # set timesteps
        self.scheduler.set_timesteps(self.cfg.model.scheduler.num_inference_timesteps)
        timesteps = self.scheduler.timesteps.to(encoder_hidden_states.device)
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, and between [0, 1]
        extra_step_kwargs = {}
        if "eta" in set(inspect.signature(self.scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = self.cfg.model.scheduler.eta

        # reverse
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2)
                if self.do_classifier_free_guidance
                else latents
            )
            lengths_reverse = (
                lengths * 2 if self.do_classifier_free_guidance else lengths
            )
            # latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            # predict the noise residual
            controlnet_residuals = None
            if self.is_controlnet and controlnet_cond is not None:
                if self.do_classifier_free_guidance:
                    controlnet_prompt_embeds = encoder_hidden_states.chunk(2)[1]
                else:
                    controlnet_prompt_embeds = encoder_hidden_states
                controlnet_residuals = self.controlnet(
                    sample=latents,
                    timestep=t,
                    encoder_hidden_states=controlnet_prompt_embeds,
                    controlnet_cond=controlnet_cond,
                )
                if self.do_classifier_free_guidance:
                    controlnet_residuals = [
                        torch.cat(
                            [torch.zeros_like(r), r * self.cfg.model.controlnet_scale],
                            dim=1,
                        )
                        for r in controlnet_residuals
                    ]
                else:
                    controlnet_residuals = [
                        r * self.cfg.model.controlnet_scale
                        for r in controlnet_residuals
                    ]

            noise_pred = self.denoiser(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=encoder_hidden_states,
                lengths=lengths_reverse,
                controlnet_residuals=controlnet_residuals,
            )[0]
            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
                # text_embeddings_for_guidance = encoder_hidden_states.chunk(
                #     2)[1] if self.do_classifier_free_guidance else encoder_hidden_states
            latents = self.scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs
            ).prev_sample
            # if self.predict_epsilon:
            #     latents = self.scheduler.step(noise_pred, t, latents,
            #                                   **extra_step_kwargs).prev_sample
            # else:
            #     # predict x for standard diffusion model
            #     # compute the previous noisy sample x_t -> x_t-1
            #     latents = self.scheduler.step(noise_pred,
            #                                   t,
            #                                   latents,
            #                                   **extra_step_kwargs).prev_sample

        # [batch_size, 1, latent_dim] -> [1, batch_size, latent_dim]
        latents = latents.permute(1, 0, 2)
        return latents

    def _diffusion_reverse_tsne(
        self, encoder_hidden_states, lengths=None, controlnet_cond=None
    ):
        # init latents
        bsz = encoder_hidden_states.shape[0]
        if self.do_classifier_free_guidance:
            bsz = bsz // 2
        if self.vae_type == "no":
            assert (
                lengths is not None
            ), "no vae (diffusion only) need lengths for diffusion"
            latents = torch.randn(
                (bsz, max(lengths), self.cfg.DATASET.NFEATS),
                device=encoder_hidden_states.device,
                dtype=torch.float,
            )
        else:
            latents = torch.randn(
                (bsz, self.latent_dim[0], self.latent_dim[-1]),
                device=encoder_hidden_states.device,
                dtype=torch.float,
            )

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        # set timesteps
        self.scheduler.set_timesteps(self.cfg.model.scheduler.num_inference_timesteps)
        timesteps = self.scheduler.timesteps.to(encoder_hidden_states.device)
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, and between [0, 1]
        extra_step_kwargs = {}
        if "eta" in set(inspect.signature(self.scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = self.cfg.model.scheduler.eta

        # reverse
        latents_t = []
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2)
                if self.do_classifier_free_guidance
                else latents
            )
            lengths_reverse = (
                lengths * 2 if self.do_classifier_free_guidance else lengths
            )
            # latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            # predict the noise residual
            controlnet_residuals = None
            if self.is_controlnet and controlnet_cond is not None:
                if self.do_classifier_free_guidance:
                    controlnet_prompt_embeds = encoder_hidden_states.chunk(2)[1]
                else:
                    controlnet_prompt_embeds = encoder_hidden_states
                controlnet_residuals = self.controlnet(
                    sample=latents,
                    timestep=t,
                    encoder_hidden_states=controlnet_prompt_embeds,
                    controlnet_cond=controlnet_cond,
                )
                if self.do_classifier_free_guidance:
                    controlnet_residuals = [
                        torch.cat(
                            [torch.zeros_like(r), r * self.cfg.model.controlnet_scale],
                            dim=1,
                        )
                        for r in controlnet_residuals
                    ]
                else:
                    controlnet_residuals = [
                        r * self.cfg.model.controlnet_scale
                        for r in controlnet_residuals
                    ]

            noise_pred = self.denoiser(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=encoder_hidden_states,
                lengths=lengths_reverse,
                controlnet_residuals=controlnet_residuals,
            )[0]
            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
                # text_embeddings_for_guidance = encoder_hidden_states.chunk(
                #     2)[1] if self.do_classifier_free_guidance else encoder_hidden_states
            latents = self.scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs
            ).prev_sample
            # [batch_size, 1, latent_dim] -> [1, batch_size, latent_dim]
            latents_t.append(latents.permute(1, 0, 2))
        # [1, batch_size, latent_dim] -> [t, batch_size, latent_dim]
        latents_t = torch.cat(latents_t)
        return latents_t

    def _diffusion_process(
        self, latents, encoder_hidden_states, lengths=None, controlnet_cond=None
    ):
        """
        heavily from https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py
        """
        # our latent   [batch_size, n_token=1 or 5 or 10, latent_dim=256]
        # sd  latent   [batch_size, [n_token0=64,n_token1=64], latent_dim=4]
        # [n_token, batch_size, latent_dim] -> [batch_size, n_token, latent_dim]
        latents = latents.permute(1, 0, 2)

        # Sample noise that we'll add to the latents
        # [batch_size, n_token, latent_dim]
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each motion
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=latents.device,
        )
        timesteps = timesteps.long()
        # Add noise to the latents according to the noise magnitude at each timestep
        noisy_latents = self.noise_scheduler.add_noise(
            latents.clone(), noise, timesteps
        )
        # Predict the noise residual
        controlnet_residuals = None
        if self.is_controlnet and controlnet_cond is not None:
            controlnet_residuals = self.controlnet(
                sample=latents,
                timestep=timesteps,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=controlnet_cond,
            )
            controlnet_residuals = [
                r * self.cfg.model.controlnet_scale for r in controlnet_residuals
            ]

        noise_pred = self.denoiser(
            sample=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            lengths=lengths,
            controlnet_residuals=controlnet_residuals,
            return_dict=False,
        )[0]
        # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
        if self.cfg.LOSS.LAMBDA_PRIOR != 0.0:
            noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
            noise, noise_prior = torch.chunk(noise, 2, dim=0)
        else:
            noise_pred_prior = 0
            noise_prior = 0
        n_set = {
            "noise": noise,
            "noise_prior": noise_prior,
            "noise_pred": noise_pred,
            "noise_pred_prior": noise_pred_prior,
        }
        if not self.predict_epsilon:
            n_set["pred"] = noise_pred
            n_set["latent"] = latents
        return n_set

    def train_vae_forward(self, batch):
        feats_ref = batch["motion"]
        lengths = batch["length"]

        if self.vae_type in ["mld", "vposert", "actor"]:
            motion_z, dist_m = self.vae.encode(feats_ref, lengths)
            feats_rst = self.vae.decode(motion_z, lengths)
        else:
            raise TypeError("vae_type must be mcross or actor")

        # prepare for metric
        recons_z, dist_rm = self.vae.encode(feats_rst, lengths)

        # joints recover
        if self.condition == "text":
            joints_rst = self.feats2joints(feats_rst)
            joints_ref = self.feats2joints(feats_ref)
        elif self.condition == "action":
            mask = batch["mask"]
            joints_rst = self.feats2joints(feats_rst, mask)
            joints_ref = self.feats2joints(feats_ref, mask)

        if dist_m is not None:
            if self.is_vae:
                # Create a centred normal distribution to compare with
                mu_ref = torch.zeros_like(dist_m.loc)
                scale_ref = torch.ones_like(dist_m.scale)
                dist_ref = torch.distributions.Normal(mu_ref, scale_ref)
            else:
                dist_ref = dist_m

        # cut longer part over max length
        min_len = min(feats_ref.shape[1], feats_rst.shape[1])
        rs_set = {
            "m_ref": feats_ref[:, :min_len, :],
            "m_rst": feats_rst[:, :min_len, :],
            # [bs, ntoken, nfeats]<= [ntoken, bs, nfeats]
            "lat_m": motion_z.permute(1, 0, 2),
            "lat_rm": recons_z.permute(1, 0, 2),
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
            "dist_m": dist_m,
            "dist_ref": dist_ref,
        }
        return rs_set

    def _get_control_hint(self, batch, lengths, feats_ref=None):
        if not self.is_controlnet:
            return None
        if "control_hint" in batch:
            hint = batch["control_hint"]
            if not torch.is_tensor(hint):
                hint = torch.tensor(hint)
            return hint
        if feats_ref is None:
            return None
        if torch.is_tensor(lengths):
            lengths = lengths.tolist()
        joints = self.feats2joints(feats_ref)
        first = joints[:, 0]
        last = torch.stack(
            [joints[i, int(lengths[i]) - 1] for i in range(len(lengths))], dim=0
        )
        return torch.stack((first, last), dim=1)

    def _get_control_cond(self, batch, lengths, feats_ref=None):
        if not self.is_controlnet:
            return None
        hint = self._get_control_hint(batch, lengths, feats_ref=feats_ref)
        if hint is None:
            return None
        if feats_ref is not None and hint.device != feats_ref.device:
            hint = hint.to(feats_ref.device)
        elif feats_ref is None:
            hint = hint.to(next(self.controlnet_cond_encoder.parameters()).device)
        control_cond = self.controlnet_cond_encoder(hint)
        return control_cond

    def train_diffusion_forward(self, batch):
        feats_ref = batch["motion"]
        lengths = batch["length"]
        # motion encode
        with torch.no_grad():
            if self.vae_type in ["mld", "vposert", "actor"]:
                z, dist = self.vae.encode(feats_ref, lengths)
            elif self.vae_type == "no":
                z = feats_ref.permute(1, 0, 2)
            else:
                raise TypeError("vae_type must be mcross or actor")

        if self.condition in ["text", "text_uncond"]:
            text = batch["text"]
            # classifier free guidance: randomly drop text during training
            text = ["" if np.random.rand(1) < self.guidance_uncodp else i for i in text]
            # text encode
            cond_emb = self.text_encoder(text)
        elif self.condition in ["action"]:
            action = batch["action"]
            # text encode
            cond_emb = action
        else:
            raise TypeError(f"condition type {self.condition} not supported")

        controlnet_cond = self._get_control_cond(batch, lengths, feats_ref=feats_ref)
        # diffusion process return with noise and noise_pred
        n_set = self._diffusion_process(
            z, cond_emb, lengths, controlnet_cond=controlnet_cond
        )
        return {**n_set}

    def test_diffusion_forward(self, batch, finetune_decoder=False):
        lengths = batch["length"]

        if self.condition in ["text", "text_uncond"]:
            # get text embeddings
            if self.do_classifier_free_guidance:
                uncond_tokens = [""] * len(lengths)
                if self.condition == "text":
                    texts = batch["text"]
                    uncond_tokens.extend(texts)
                elif self.condition == "text_uncond":
                    uncond_tokens.extend(uncond_tokens)
                texts = uncond_tokens
            cond_emb = self.text_encoder(texts)
        elif self.condition in ["action"]:
            cond_emb = batch["action"]
            if self.do_classifier_free_guidance:
                cond_emb = torch.cat(
                    cond_emb,
                    torch.zeros_like(batch["action"], dtype=batch["action"].dtype),
                )
        else:
            raise TypeError(f"condition type {self.condition} not supported")

        controlnet_cond = self._get_control_cond(batch, lengths)
        # diffusion reverse
        with torch.no_grad():
            z = self._diffusion_reverse(
                cond_emb, lengths, controlnet_cond=controlnet_cond
            )

        with torch.no_grad():
            if self.vae_type in ["mld", "vposert", "actor"]:
                feats_rst = self.vae.decode(z, lengths)
            elif self.vae_type == "no":
                feats_rst = z.permute(1, 0, 2)
            else:
                raise TypeError("vae_type must be mcross or actor or mld")

        joints_rst = self.feats2joints(feats_rst)

        rs_set = {
            "m_rst": feats_rst,
            # [bs, ntoken, nfeats]<= [ntoken, bs, nfeats]
            "lat_t": z.permute(1, 0, 2),
            "joints_rst": joints_rst,
        }

        # prepare gt/refer for metric
        if "motion" in batch.keys() and not finetune_decoder:
            feats_ref = batch["motion"].detach()
            with torch.no_grad():
                if self.vae_type in ["mld", "vposert", "actor"]:
                    motion_z, dist_m = self.vae.encode(feats_ref, lengths)
                    recons_z, dist_rm = self.vae.encode(feats_rst, lengths)
                elif self.vae_type == "no":
                    motion_z = feats_ref
                    recons_z = feats_rst

            joints_ref = self.feats2joints(feats_ref)

            rs_set["m_ref"] = feats_ref
            rs_set["lat_m"] = motion_z.permute(1, 0, 2)
            rs_set["lat_rm"] = recons_z.permute(1, 0, 2)
            rs_set["joints_ref"] = joints_ref
        return rs_set

    def _prepare_vis_cache(self, split, batch, vis_n):
        if split in self._vis_cache:
            return self._vis_cache[split]

        lengths = batch.get("length", [])
        if torch.is_tensor(lengths):
            lengths = lengths.tolist()
        lengths = list(lengths)
        if len(lengths) == 0:
            return None

        num = min(vis_n, len(lengths))
        texts_full = batch.get("text", [""] * len(lengths))
        cache = {
            "lengths": [int(lengths[i]) for i in range(num)],
            "texts": [
                texts_full[i] if i < len(texts_full) else f"{split}_{i}"
                for i in range(num)
            ],
            "titles": [
                (texts_full[i][:80] if i < len(texts_full) else f"{split}_{i}")
                for i in range(num)
            ],
        }

        if "motion" in batch:
            cache["motion"] = batch["motion"][:num].detach().cpu()
        if "action" in batch:
            cache["action"] = batch["action"][:num].detach().cpu()
        if "control_hint" in batch:
            cache["control_hint"] = batch["control_hint"][:num].detach().cpu()

        self._vis_cache[split] = cache
        return cache

    def _run_vis_inference(self, vis_state):
        device = next(self.parameters()).device
        generator = torch.Generator(device=device)
        generator.manual_seed(getattr(self.cfg, "SEED_VALUE", 0))

        lengths = vis_state["lengths"]
        texts = vis_state.get("texts", [""] * len(lengths))
        batch_stub = {"length": lengths}
        feats_ref = None
        if "control_hint" in vis_state:
            batch_stub["control_hint"] = vis_state["control_hint"].to(device)
        if "motion" in vis_state:
            feats_ref = vis_state["motion"].to(device)

        if self.condition in ["text", "text_uncond"]:
            if self.do_classifier_free_guidance:
                uncond_tokens = [""] * len(texts)
                if self.condition == "text":
                    texts = uncond_tokens + texts
                else:
                    texts = uncond_tokens + uncond_tokens
            cond_emb = self.text_encoder(texts)
        elif self.condition == "action":
            if "action" not in vis_state:
                return None
            cond_emb = vis_state["action"].to(device)
            if self.do_classifier_free_guidance:
                cond_emb = torch.cat(
                    (torch.zeros_like(cond_emb), cond_emb), dim=0
                )
        else:
            raise TypeError(f"Unsupported condition {self.condition} for visualization")
        controlnet_cond = self._get_control_cond(
            batch_stub, lengths, feats_ref=feats_ref
        )

        with torch.no_grad():
            z = self._diffusion_reverse(
                cond_emb,
                lengths,
                controlnet_cond=controlnet_cond,
                generator=generator,
            )
            if self.vae_type in ["mld", "vposert", "actor"]:
                feats_rst = self.vae.decode(z, lengths)
            elif self.vae_type == "no":
                feats_rst = z.permute(1, 0, 2)
            else:
                raise TypeError("vae_type must be mcross or actor or mld")

        joints_rst = self.feats2joints(feats_rst.detach().cpu())
        joints_ref = None
        if feats_ref is not None:
            joints_ref = self.feats2joints(feats_ref.detach().cpu())
        return joints_rst, joints_ref, texts, lengths

    def _maybe_visualize(self, split, batch, rs_set, batch_idx):
        vis_cfg = getattr(self.cfg, "LOGGER", self.cfg)
        vis_n = getattr(vis_cfg, "VAL_VIS_SAMPLES", 0)
        if (
            split not in ["val", "test"]
            or vis_n <= 0
            or batch_idx > 0
            or self.trainer.sanity_checking
        ):
            return

        vis_state = self._prepare_vis_cache(split, batch, vis_n)
        if vis_state is None:
            return

        vis_outputs = self._run_vis_inference(vis_state)
        if vis_outputs is None:
            return
        joints, joints_ref, texts_full, lengths = vis_outputs
        titles = vis_state.get("titles", texts_full)

        out_dir = os.path.join(
            self.cfg.FOLDER_EXP, "vis", f"epoch_{self.trainer.current_epoch}"
        )
        os.makedirs(out_dir, exist_ok=True)

        exp = None
        if isinstance(self.logger, (list, tuple)):
            for lg in self.logger:
                if hasattr(lg, "experiment"):
                    exp = lg.experiment
                    break
        else:
            exp = getattr(self.logger, "experiment", None)

        for idx_vis in range(min(vis_n, len(joints))):
            length = lengths[idx_vis]
            motion = joints[idx_vis][:length].detach().cpu().numpy()
            motion_ref = None
            if joints_ref is not None:
                motion_ref = joints_ref[idx_vis][:length].detach().cpu().numpy()

            if motion.shape[1] > 22:
                motion = motion[:, :22, :]
            if motion_ref is not None and motion_ref.shape[1] > 22:
                motion_ref = motion_ref[:, :22, :]

            title = titles[idx_vis] if idx_vis < len(titles) else f"{split}_{idx_vis}"
            save_path = os.path.join(out_dir, f"{split}_sample_{idx_vis}.png")

            img_array = self._save_motion_grid(motion, motion_ref, title, save_path)
            if img_array is None:
                continue
            frames_idx = np.linspace(0, len(motion) - 1, min(4, len(motion)), dtype=int)
            pose_path = os.path.join(out_dir, f"{split}_pose_{idx_vis}.png")
            pose_img = self._save_pose_axes(
                motion, motion_ref, frames_idx, title, save_path=pose_path, rows=1
            )
            video_pred, video_pred_np = self._render_pose_video(
                motion,
                motion_ref,
                title=title,
                max_frames=min(len(motion), 64),
                fps=12,
                save_path=os.path.join(out_dir, f"{split}_video_pred_{idx_vis}.gif"),
            )
            video_gt = None
            video_gt_np = None
            if motion_ref is not None:
                video_gt, video_gt_np = self._render_pose_video(
                    motion_ref,
                    None,
                    title=f"{title} | GT",
                    max_frames=min(len(motion_ref), 64),
                    fps=12,
                    save_path=os.path.join(out_dir, f"{split}_video_gt_{idx_vis}.gif"),
                )

            if exp is not None and hasattr(exp, "add_image"):
                try:
                    img_tensor = (
                        torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
                    )
                    exp.add_image(
                        f"{split}_motion/sample_{idx_vis}",
                        img_tensor,
                        global_step=self.global_step,
                        dataformats="CHW",
                    )
                except Exception:
                    pass
                if pose_img is not None:
                    try:
                        pose_tensor = (
                            torch.from_numpy(pose_img).permute(2, 0, 1).float() / 255.0
                        )
                        exp.add_image(
                            f"{split}_motion_pose3d/sample_{idx_vis}",
                            pose_tensor,
                            global_step=self.global_step,
                            dataformats="CHW",
                        )
                    except Exception:
                        pass

            if exp is not None and hasattr(exp, "add_video"):
                if self._has_moviepy is None:
                    try:
                        import moviepy.editor  # noqa: F401
                        self._has_moviepy = True
                    except Exception:
                        self._has_moviepy = False
                if self._has_moviepy:
                    if video_pred is not None:
                        try:
                            exp.add_video(
                                f"{split}_motion_video/sample_{idx_vis}",
                                video_pred,
                                global_step=self.global_step,
                                fps=12,
                            )
                        except Exception:
                            pass
                if video_gt is not None:
                    try:
                        exp.add_video(
                            f"{split}_motion_video_gt/sample_{idx_vis}",
                            video_gt,
                            global_step=self.global_step,
                            fps=12,
                        )
                    except Exception:
                        pass
            # Mesh logging removed to avoid unclear visualization; 3D pose images are kept.

    def t2m_eval(self, batch):
        texts = batch["text"]
        motions = batch["motion"].detach().clone()
        lengths = batch["length"]
        word_embs = batch["word_embs"].detach().clone()
        pos_ohot = batch["pos_ohot"].detach().clone()
        text_lengths = batch["text_len"].detach().clone()

        # start
        start = time.time()

        if self.trainer.datamodule.is_mm:
            texts = texts * self.cfg.TEST.MM_NUM_REPEATS
            motions = motions.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            lengths = lengths * self.cfg.TEST.MM_NUM_REPEATS
            word_embs = word_embs.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            pos_ohot = pos_ohot.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            text_lengths = text_lengths.repeat_interleave(
                self.cfg.TEST.MM_NUM_REPEATS, dim=0
            )

        if self.stage in ["diffusion", "vae_diffusion"]:
            # diffusion reverse
            if self.do_classifier_free_guidance:
                uncond_tokens = [""] * len(texts)
                if self.condition == "text":
                    uncond_tokens.extend(texts)
                elif self.condition == "text_uncond":
                    uncond_tokens.extend(uncond_tokens)
                texts = uncond_tokens
            text_emb = self.text_encoder(texts)
            controlnet_cond = self._get_control_cond(batch, lengths, feats_ref=motions)
            z = self._diffusion_reverse(
                text_emb, lengths, controlnet_cond=controlnet_cond
            )
        elif self.stage in ["vae"]:
            if self.vae_type in ["mld", "vposert", "actor"]:
                z, dist_m = self.vae.encode(motions, lengths)
            else:
                raise TypeError("Not supported vae type!")
            if self.condition in ["text_uncond"]:
                # uncond random sample
                z = torch.randn_like(z)

        with torch.no_grad():
            if self.vae_type in ["mld", "vposert", "actor"]:
                feats_rst = self.vae.decode(z, lengths)
            elif self.vae_type == "no":
                feats_rst = z.permute(1, 0, 2)

        # end time
        end = time.time()
        self.times.append(end - start)

        # joints recover
        joints_rst = self.feats2joints(feats_rst)
        joints_ref = self.feats2joints(motions)

        # renorm for t2m evaluators
        feats_rst = self.datamodule.renorm4t2m(feats_rst)
        motions = self.datamodule.renorm4t2m(motions)

        # t2m motion encoder
        m_lens = lengths.copy()
        m_lens = torch.tensor(m_lens, device=motions.device)
        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        motions = motions[align_idx]
        feats_rst = feats_rst[align_idx]
        m_lens = m_lens[align_idx]
        m_lens = torch.div(
            m_lens, self.cfg.DATASET.HUMANML3D.UNIT_LEN, rounding_mode="floor"
        )

        recons_mov = self.t2m_moveencoder(feats_rst[..., :-4]).detach()
        recons_emb = self.t2m_motionencoder(recons_mov, m_lens)
        motion_mov = self.t2m_moveencoder(motions[..., :-4]).detach()
        motion_emb = self.t2m_motionencoder(motion_mov, m_lens)

        # t2m text encoder
        text_emb = self.t2m_textencoder(word_embs, pos_ohot, text_lengths)[align_idx]

        rs_set = {
            "m_ref": motions,
            "m_rst": feats_rst,
            "lat_t": text_emb,
            "lat_m": motion_emb,
            "lat_rm": recons_emb,
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
        }
        return rs_set

    def a2m_eval(self, batch):
        actions = batch["action"]
        actiontexts = batch["action_text"]
        motions = batch["motion"].detach().clone()
        lengths = batch["length"]

        if self.do_classifier_free_guidance:
            cond_emb = torch.cat((torch.zeros_like(actions), actions))

        if self.stage in ["diffusion", "vae_diffusion"]:
            z = self._diffusion_reverse(cond_emb, lengths)
        elif self.stage in ["vae"]:
            if self.vae_type in ["mld", "vposert", "actor"]:
                z, dist_m = self.vae.encode(motions, lengths)
            else:
                raise TypeError("vae_type must be mcross or actor")

        with torch.no_grad():
            if self.vae_type in ["mld", "vposert", "actor"]:
                feats_rst = self.vae.decode(z, lengths)
            elif self.vae_type == "no":
                feats_rst = z.permute(1, 0, 2)
            else:
                raise TypeError("vae_type must be mcross or actor or mld")

        mask = batch["mask"]
        joints_rst = self.feats2joints(feats_rst, mask)
        joints_ref = self.feats2joints(motions, mask)
        joints_eval_rst = self.feats2joints_eval(feats_rst, mask)
        joints_eval_ref = self.feats2joints_eval(motions, mask)

        rs_set = {
            "m_action": actions,
            "m_ref": motions,
            "m_rst": feats_rst,
            "m_lens": lengths,
            "joints_rst": joints_rst,
            "joints_ref": joints_ref,
            "joints_eval_rst": joints_eval_rst,
            "joints_eval_ref": joints_eval_ref,
        }
        return rs_set

    def a2m_gt(self, batch):
        actions = batch["action"]
        actiontexts = batch["action_text"]
        motions = batch["motion"].detach().clone()
        lengths = batch["length"]
        mask = batch["mask"]

        joints_ref = self.feats2joints(motions.to("cuda"), mask.to("cuda"))

        rs_set = {
            "m_action": actions,
            "m_text": actiontexts,
            "m_ref": motions,
            "m_lens": lengths,
            "joints_ref": joints_ref,
        }
        return rs_set

    def eval_gt(self, batch, renoem=True):
        motions = batch["motion"].detach().clone()
        lengths = batch["length"]

        # feats_rst = self.datamodule.renorm4t2m(feats_rst)
        if renoem:
            motions = self.datamodule.renorm4t2m(motions)

        # t2m motion encoder
        m_lens = lengths.copy()
        m_lens = torch.tensor(m_lens, device=motions.device)
        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        motions = motions[align_idx]
        m_lens = m_lens[align_idx]
        m_lens = torch.div(
            m_lens, self.cfg.DATASET.HUMANML3D.UNIT_LEN, rounding_mode="floor"
        )

        word_embs = batch["word_embs"].detach()
        pos_ohot = batch["pos_ohot"].detach()
        text_lengths = batch["text_len"].detach()

        motion_mov = self.t2m_moveencoder(motions[..., :-4]).detach()
        motion_emb = self.t2m_motionencoder(motion_mov, m_lens)

        # t2m text encoder
        text_emb = self.t2m_textencoder(word_embs, pos_ohot, text_lengths)[align_idx]

        # joints recover
        joints_ref = self.feats2joints(motions)

        rs_set = {
            "m_ref": motions,
            "lat_t": text_emb,
            "lat_m": motion_emb,
            "joints_ref": joints_ref,
        }
        return rs_set

    def allsplit_step(self, split: str, batch, batch_idx):
        if split in ["train", "val"]:
            if self.stage == "vae":
                rs_set = self.train_vae_forward(batch)
                rs_set["lat_t"] = rs_set["lat_m"]
            elif self.stage == "diffusion":
                rs_set = self.train_diffusion_forward(batch)
            elif self.stage == "vae_diffusion":
                vae_rs_set = self.train_vae_forward(batch)
                diff_rs_set = self.train_diffusion_forward(batch)
                t2m_rs_set = self.test_diffusion_forward(batch, finetune_decoder=True)
                # merge results
                rs_set = {
                    **vae_rs_set,
                    **diff_rs_set,
                    "gen_m_rst": t2m_rs_set["m_rst"],
                    "gen_joints_rst": t2m_rs_set["joints_rst"],
                    "lat_t": t2m_rs_set["lat_t"],
                }
            else:
                raise ValueError(f"Not support this stage {self.stage}!")

            loss = self.losses[split].update(rs_set)
            if loss is None:
                raise ValueError("Loss is None, this happend with torchmetrics > 0.7")

        # Compute the metrics - currently evaluate results from text to motion
        if split in ["val", "test"]:
            if self.condition in ["text", "text_uncond"]:
                # use t2m evaluators
                rs_set = self.t2m_eval(batch)
            elif self.condition == "action":
                # use a2m evaluators
                rs_set = self.a2m_eval(batch)
            # MultiModality evaluation sperately
            if self.trainer.datamodule.is_mm:
                metrics_dicts = ["MMMetrics"]
            else:
                metrics_dicts = self.metrics_dict

            for metric in metrics_dicts:
                if metric == "TemosMetric":
                    phase = split if split != "val" else "eval"
                    if eval(f"self.cfg.{phase.upper()}.DATASETS")[0].lower() not in [
                        "humanml3d",
                        "kit",
                    ]:
                        raise TypeError(
                            "APE and AVE metrics only support humanml3d and kit datasets now"
                        )

                    getattr(self, metric).update(
                        rs_set["joints_rst"], rs_set["joints_ref"], batch["length"]
                    )
                elif metric == "TM2TMetrics":
                    getattr(self, metric).update(
                        # lat_t, latent encoded from diffusion-based text
                        # lat_rm, latent encoded from reconstructed motion
                        # lat_m, latent encoded from gt motion
                        # rs_set['lat_t'], rs_set['lat_rm'], rs_set['lat_m'], batch["length"])
                        rs_set["lat_t"],
                        rs_set["lat_rm"],
                        rs_set["lat_m"],
                        batch["length"],
                    )
                elif metric == "UncondMetrics":
                    getattr(self, metric).update(
                        recmotion_embeddings=rs_set["lat_rm"],
                        gtmotion_embeddings=rs_set["lat_m"],
                        lengths=batch["length"],
                    )
                elif metric == "MRMetrics":
                    getattr(self, metric).update(
                        rs_set["joints_rst"], rs_set["joints_ref"], batch["length"]
                    )
                elif metric == "MMMetrics":
                    getattr(self, metric).update(
                        rs_set["lat_rm"].unsqueeze(0), batch["length"]
                    )
                elif metric == "HUMANACTMetrics":
                    getattr(self, metric).update(
                        rs_set["m_action"],
                        rs_set["joints_eval_rst"],
                        rs_set["joints_eval_ref"],
                        rs_set["m_lens"],
                    )
                elif metric == "UESTCMetrics":
                    # the stgcn model expects rotations only
                    getattr(self, metric).update(
                        rs_set["m_action"],
                        rs_set["m_rst"]
                        .view(*rs_set["m_rst"].shape[:-1], 6, 25)
                        .permute(0, 3, 2, 1)[:, :-1],
                        rs_set["m_ref"]
                        .view(*rs_set["m_ref"].shape[:-1], 6, 25)
                        .permute(0, 3, 2, 1)[:, :-1],
                        rs_set["m_lens"],
                    )
                else:
                    raise TypeError(f"Not support this metric {metric}")

            self._maybe_visualize(split, batch, rs_set, batch_idx)

        # return forward output rather than loss during test
        if split in ["test"]:
            return rs_set["joints_rst"], batch["length"]
        return loss
