import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from tqdm import tqdm
from torchvision.transforms import v2
from torchvision.utils import make_grid, save_image
from einops import rearrange

from utils.zero123plus_train_util import instantiate_from_config
from diffusers import (
    DiffusionPipeline,
    EulerAncestralDiscreteScheduler,
    DDPMScheduler,
    UNet2DConditionModel,
)
from .pipeline import RefOnlyNoisedUNet
from huggingface_hub import hf_hub_download
from torchvision import transforms
import lpips
from utils.losses import (
    angular_loss,
    align_depth_least_square,
    align_depth_least_square_torch,
)
from scripts.ccm_to_depth import depth_to_ccm
from utils.visualization import apply_turbo_colormap
import copy


def rearrange_grid(tensor, mode=[3, 2]):
    rows = torch.chunk(tensor, mode[0], dim=-2)  # [3, 320, 640]
    patches = []
    for row in rows:
        patches.extend(torch.chunk(row, mode[1], dim=-1))  # [3, 320, 320]
    return torch.cat(patches, dim=-2)  # [3, 1920, 320]


def clone_tensor_dict(tensor_dict):
    cloned_dict = {}
    for key, tensor in tensor_dict.items():
        cloned_dict[key] = tensor.detach().clone()
    return cloned_dict


def scale_latents(latents):
    latents = (latents - 0.22) * 0.75
    return latents


def unscale_latents(latents):
    latents = latents / 0.75 + 0.22
    return latents


def scale_image(image):
    image = image * 0.5 / 0.8
    return image


def unscale_image(image):
    image = image / 0.5 * 0.8
    return image


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class MVDiffusion(pl.LightningModule):
    def __init__(
        self,
        stable_diffusion_config,
        white_back_unet_path,
        drop_cond_prob=0.1,
        train_controlnet=False,
        one_step=False,
        front_guided=False,
        image_loss=False,
        render_loss=False,
        finetune_vae_decoder=False,
        vae_path=None,
        domains=None,
        weights_domains=None,
        weight_lpips=None,
        weight_global_mse=None,
        weight_render_loss=None,
    ):
        super(MVDiffusion, self).__init__()

        self.drop_cond_prob = drop_cond_prob
        self.train_controlnet = train_controlnet
        self.one_step = one_step
        self.front_guided = front_guided
        self.image_loss = image_loss
        self.render_loss = render_loss
        self.finetune_vae_decoder = finetune_vae_decoder
        self.domains = domains
        self.weights_domains = weights_domains
        self.weight_lpips = weight_lpips
        self.weight_global_mse = weight_global_mse
        self.weight_render_loss = weight_render_loss
        if self.image_loss:
            self.lpips_loss = lpips.LPIPS(net="vgg", eval_mode=True, verbose=False)

        # assert self.one_step
        self.register_schedule()

        # init modules
        # if self.front_guided or self.one_step:
        #     unet = UNet2DConditionModel.from_pretrained(subfolder="unet", in_channels=8, low_cpu_mem_usage=False, ignore_mismatched_sizes=True, **stable_diffusion_config)
        #     pipeline = DiffusionPipeline.from_pretrained(unet=unet, **stable_diffusion_config)
        # else:
        #     pipeline = DiffusionPipeline.from_pretrained(**stable_diffusion_config)

        pipeline = DiffusionPipeline.from_pretrained(
            **stable_diffusion_config
        )  # for pure unet

        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipeline.scheduler.config, timestep_spacing="trailing"
        )
        self.pipeline = pipeline

        # # load custom white-background UNet
        # print("Loading custom white-background unet ...")
        # if os.path.exists(white_back_unet_path):
        #     unet_ckpt_path = white_back_unet_path
        # else:
        #     unet_ckpt_path = hf_hub_download(
        #         repo_id="TencentARC/InstantMesh",
        #         filename="diffusion_pytorch_model.bin",
        #         repo_type="model",
        #     )
        # state_dict = torch.load(unet_ckpt_path, map_location="cpu")
        # self.pipeline.unet.load_state_dict(state_dict, strict=True)
        # load custom white-background UNet
        if white_back_unet_path is not None:
            print(f"Loading custom white-background unet...")
            if os.path.exists(white_back_unet_path):
                unet_ckpt_path = white_back_unet_path
            else:
                print(f"{white_back_unet_path} not exists")
                unet_ckpt_path = hf_hub_download(
                    repo_id="TencentARC/InstantMesh",
                    filename="diffusion_pytorch_model.bin",
                    repo_type="model",
                )
            print(f"unet_ckpt_path:{unet_ckpt_path}")
            if unet_ckpt_path.endswith(".safetensors"):
                from safetensors.torch import load_file

                state_dict = load_file(unet_ckpt_path)
            else:
                state_dict = torch.load(unet_ckpt_path, map_location="cpu")

            # if (self.front_guided or self.one_step) and state_dict['conv_in.weight'].shape[1]==4:
            #     # double unet input channel from 4 to 8
            #     ori_conv_in_weight = state_dict.pop('conv_in.weight') /  2.0
            #     ori_conv_in_bias = state_dict.pop('conv_in.bias') /  2.0
            #     self.pipeline.unet.conv_in.weight.data = ori_conv_in_weight.repeat(1, 2, 1, 1).contiguous().to(self.pipeline.unet.conv_in.weight.data.dtype)
            #     self.pipeline.unet.conv_in.bias.data = ori_conv_in_bias.to(self.pipeline.unet.conv_in.bias.data.dtype)

            self.pipeline.unet.load_state_dict(state_dict, strict=False)

        if vae_path is not None:
            print(f"Loading custom vae...")
            assert os.path.exists(vae_path)
            print(f"vae_path:{vae_path}")
            if vae_path.endswith(".safetensors"):
                from safetensors.torch import load_file

                state_dict = load_file(vae_path)
            else:
                state_dict = torch.load(vae_path, map_location="cpu")

            self.pipeline.vae.load_state_dict(state_dict, strict=False)

        train_sched = DDPMScheduler.from_config(self.pipeline.scheduler.config)
        if isinstance(self.pipeline.unet, UNet2DConditionModel):
            self.pipeline.unet = RefOnlyNoisedUNet(
                self.pipeline.unet, train_sched, self.pipeline.scheduler
            )

        self.train_scheduler = train_sched  # use ddpm scheduler during training

        if self.train_controlnet:
            self.pipeline.add_controlnet()

            self.pipeline.vae.requires_grad_(False)
            self.pipeline.unet.unet.requires_grad_(False)
            self.pipeline.text_encoder.requires_grad_(False)
            self.pipeline.unet.controlnet.train()

        self.unet = self.pipeline.unet
        if self.finetune_vae_decoder:
            self.vae_post_quant_conv = self.pipeline.vae.post_quant_conv
            self.vae_decoder = self.pipeline.vae.decoder

        # validation output buffer
        self.validation_step_outputs = []

    def register_schedule(self):
        self.num_timesteps = 1000

        # replace scaled_linear schedule with linear schedule as Zero123++
        beta_start = 0.00085
        beta_end = 0.0120
        betas = torch.linspace(
            beta_start, beta_end, self.num_timesteps, dtype=torch.float32
        )

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, dtype=torch.float64), alphas_cumprod[:-1]], 0
        )

        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas_cumprod", alphas_cumprod.float())
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev.float())

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod).float())
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod).float()
        )

        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod).float()
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1).float()
        )

    def on_fit_start(self):
        device = torch.device(f"cuda:{self.global_rank}")
        self.pipeline.to(device)

        if self.image_loss:
            self.lpips_loss.to(device)
        if self.render_loss:
            from render_loss import RenderLoss

            self.image_renderer = RenderLoss(device=device)

        if self.global_rank == 0:
            os.makedirs(os.path.join(self.logdir, "images"), exist_ok=True)
            os.makedirs(os.path.join(self.logdir, "images_val"), exist_ok=True)

    def prepare_batch_data(self, batch):
        # prepare stable diffusion input
        cond_imgs = batch["cond_imgs"]  # (B, C, H, W)
        cond_imgs = cond_imgs.to(self.device)

        # random resize the condition image
        cond_size = np.random.randint(128, 513)
        cond_imgs = v2.functional.resize(
            cond_imgs, cond_size, interpolation=3, antialias=True
        ).clamp(0, 1)

        if self.render_loss:
            # prepare black bg cond imgs for render loss
            gt_rendered_rgb = batch["gt_rendered_rgb"]  # (B, C, H, W)
            gt_rendered_rgb = gt_rendered_rgb.to(self.device)

            gt_rendered_rgb = v2.functional.resize(
                gt_rendered_rgb, 320, interpolation=3, antialias=True
            ).clamp(0, 1)
            batch["gt_rendered_rgb"] = gt_rendered_rgb

        # tiled cond image 3x2
        # to (B, 6, C, H, W)
        cond_imgs_tiled = v2.functional.resize(
            batch["cond_imgs"].to(self.device), 320, interpolation=3, antialias=True
        ).clamp(0, 1)
        cond_imgs_tiled = cond_imgs_tiled.unsqueeze(1).expand(-1, 6, -1, -1, -1)
        # to (B, C, 3H, 2W)
        cond_imgs_tiled = rearrange(
            cond_imgs_tiled, "b (x y) c h w -> b c (x h) (y w)", x=3, y=2
        )

        target_imgs = batch["target_imgs"]  # (B, 6, C, H, W)
        target_imgs = v2.functional.resize(
            target_imgs, 320, interpolation=3, antialias=True
        ).clamp(0, 1)
        target_imgs = rearrange(
            target_imgs, "b (x y) c h w -> b c (x h) (y w)", x=3, y=2
        )  # (B, C, 3H, 2W)
        target_imgs = target_imgs.to(self.device)

        if self.train_controlnet:
            ref_imgs = batch["ref_imgs"]  # (B, 6, C, H, W)
            ref_imgs = v2.functional.resize(
                ref_imgs, 320, interpolation=3, antialias=True
            ).clamp(0, 1)
            ref_imgs = rearrange(
                ref_imgs, "b (x y) c h w -> b c (x h) (y w)", x=3, y=2
            )  # (B, C, 3H, 2W)
            ref_imgs = ref_imgs.to(self.device)

            return cond_imgs, target_imgs, ref_imgs

        if self.front_guided:
            front_imgs = batch["front_imgs"]  # (B, C, H, W)
            front_imgs = front_imgs.to(self.device)

            # random resize the front image to have the same size with the condition image
            front_imgs = v2.functional.resize(
                front_imgs, cond_size, interpolation=3, antialias=True
            ).clamp(0, 1)

            # NOTE: mask loss
            target_masks = batch["target_masks"]  # (B, 1, H, W)
            target_masks = v2.functional.resize(
                target_masks.to(self.device), 320, interpolation=3, antialias=True
            ).clamp(0, 1)
            target_masks = target_masks.unsqueeze(1).expand(-1, 6, -1, -1, -1)
            # to (B, C, 3H, 2W)
            target_masks = rearrange(
                target_masks, "b (x y) c h w -> b c (x h) (y w)", x=3, y=2
            )

            return (
                cond_imgs,
                target_imgs,
                front_imgs,
                cond_imgs_tiled,
                target_masks,
                batch,
            )

        return cond_imgs, target_imgs

    @torch.no_grad()
    def forward_vision_encoder(self, images):
        dtype = next(self.pipeline.vision_encoder.parameters()).dtype
        image_pil = [
            v2.functional.to_pil_image(images[i]) for i in range(images.shape[0])
        ]
        image_pt = self.pipeline.feature_extractor_clip(
            images=image_pil, return_tensors="pt"
        ).pixel_values
        image_pt = image_pt.to(device=self.device, dtype=dtype)
        global_embeds = self.pipeline.vision_encoder(
            image_pt, output_hidden_states=False
        ).image_embeds
        global_embeds = global_embeds.unsqueeze(-2)

        encoder_hidden_states = self.pipeline._encode_prompt("", self.device, 1, False)[
            0
        ]
        ramp = global_embeds.new_tensor(
            self.pipeline.config.ramping_coefficients
        ).unsqueeze(-1)
        encoder_hidden_states = encoder_hidden_states + global_embeds * ramp

        return encoder_hidden_states

    @torch.no_grad()
    def encode_condition_image(self, images):
        dtype = next(self.pipeline.vae.parameters()).dtype
        image_pil = [
            v2.functional.to_pil_image(images[i]) for i in range(images.shape[0])
        ]
        image_pt = self.pipeline.feature_extractor_vae(
            images=image_pil, return_tensors="pt"
        ).pixel_values
        image_pt = image_pt.to(device=self.device, dtype=dtype)
        latents = self.pipeline.vae.encode(image_pt).latent_dist.sample()
        return latents

    @torch.no_grad()
    def encode_target_images(self, images):
        dtype = next(self.pipeline.vae.parameters()).dtype
        # equals to scaling images to [-1, 1] first and then call scale_image
        images = (images - 0.5) / 0.8  # [-0.625, 0.625]
        posterior = self.pipeline.vae.encode(images.to(dtype)).latent_dist
        latents = posterior.sample() * self.pipeline.vae.config.scaling_factor
        latents = scale_latents(latents)
        return latents

    @torch.no_grad()
    def encode_ref_image(self, images):
        images = transforms.Normalize([0.5], [0.5])(images).to(
            device=self.pipeline.unet.controlnet.device,
            dtype=self.pipeline.unet.controlnet.dtype,
        )
        return images

    def forward_unet(self, latents, t, prompt_embeds, cond_latents, control_depth=None):
        dtype = next(self.pipeline.unet.parameters()).dtype
        latents = latents.to(dtype)
        prompt_embeds = prompt_embeds.to(dtype)
        cond_latents = cond_latents.to(dtype)
        cross_attention_kwargs = dict(cond_lat=cond_latents)
        # cross_attention_kwargs = dict()

        if control_depth is not None:
            cross_attention_kwargs["control_depth"] = control_depth

        pred_noise = self.pipeline.unet(
            latents,
            t,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
            return_dict=False,
        )[0]
        return pred_noise

    def predict_start_from_z_and_v(self, x_t, t, v):
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def get_v(self, x, noise, t):
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x.shape) * noise
            - extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * x
        )

    def training_step(self, batch, batch_idx):
        # get input
        if self.train_controlnet:
            cond_imgs, target_imgs, ref_imgs = self.prepare_batch_data(batch)
        elif self.front_guided:
            (
                cond_imgs,
                target_imgs,
                front_imgs,
                cond_imgs_tiled,
                target_masks,
                extra_dict,
            ) = self.prepare_batch_data(batch)
        else:
            cond_imgs, target_imgs = self.prepare_batch_data(batch)

        # sample random timestep
        B = cond_imgs.shape[0]

        if not self.one_step:
            t = torch.randint(0, self.num_timesteps, size=(B,)).long().to(self.device)
        else:
            t = (
                torch.randint(self.num_timesteps - 1, self.num_timesteps, size=(B,))
                .long()
                .to(self.device)
            )

        # classifier-free guidance
        if np.random.rand() < self.drop_cond_prob:
            prompt_embeds = self.pipeline._encode_prompt(
                [""] * B, self.device, 1, False
            )
            cond_latents = self.encode_condition_image(torch.zeros_like(cond_imgs))
            if self.front_guided:
                front_latents = self.encode_condition_image(
                    torch.zeros_like(front_imgs)
                )
        else:
            prompt_embeds = self.forward_vision_encoder(cond_imgs)
            cond_latents = self.encode_condition_image(cond_imgs)
            if self.front_guided:
                front_latents = self.encode_condition_image(front_imgs)

        # prepare latents
        latents = self.encode_target_images(target_imgs)
        if (
            self.front_guided
        ):  # TODO: this is not for front guidance, but for using tiled cond image
            tiled_cond_latents = self.encode_target_images(cond_imgs_tiled)

        if not self.one_step:
            noise = torch.randn_like(latents)
            latents_noisy = self.train_scheduler.add_noise(latents, noise, t)
        else:
            latents_noisy = torch.zeros_like(latents)

            # noised tiled_cond_latents
            # noise_2 = torch.randn_like(tiled_cond_latents)
            # tiled_cond_latents_noisy = self.train_scheduler.add_noise(tiled_cond_latents, noise_2, torch.abs(self.num_timesteps-t))

        # front_guided
        if self.front_guided:
            # unet_input = torch.cat([tiled_cond_latents, latents_noisy], dim = 1)
            # cond_latents = torch.cat([cond_latents, torch.zeros_like(cond_latents)], dim = 1)

            unet_input = tiled_cond_latents  # [b, 4, w, h]
            cond_latents = cond_latents  # [b, 4, w, h]
        else:
            unet_input = latents_noisy

        if self.train_controlnet:
            control_depth = self.encode_ref_image(ref_imgs)
            v_pred = self.forward_unet(
                unet_input,
                t,
                prompt_embeds,
                cond_latents,
                control_depth=control_depth,
            )
        else:
            v_pred = self.forward_unet(unet_input, t, prompt_embeds, cond_latents)

        if not self.one_step:
            v_target = self.get_v(latents, noise, t)
            loss, loss_dict = self.compute_loss(v_pred, v_target)
        else:
            # latents_pred = self.predict_start_from_z_and_v(latents_noisy, t, v_pred)
            latents_pred = -v_pred
            # residue prediction
            # latents_pred = latents_pred + tiled_cond_latents
            if self.image_loss:
                latents_pred = unscale_latents(latents_pred)
                pred_images = unscale_image(
                    self.pipeline.vae.decode(
                        latents_pred / self.pipeline.vae.config.scaling_factor,
                        return_dict=False,
                    )[0]
                )  # [-1, 1]

                pred_images = (pred_images * 0.5 + 0.5).clamp(0, 1)

                if self.render_loss:
                    loss, loss_dict, rendered_rgb = self.compute_loss_image(
                        pred_images, target_imgs, target_masks, extra_dict
                    )
                else:
                    loss, loss_dict = self.compute_loss_image(
                        pred_images, target_imgs, target_masks, extra_dict
                    )
            else:
                loss, loss_dict = self.compute_loss(latents_pred, latents)

        # logging
        self.log_dict(
            loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True
        )
        self.log(
            "global_step",
            self.global_step,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )
        lr = self.optimizers().param_groups[0]["lr"]
        self.log("lr_abs", lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.global_step % 100 == 0 and self.global_rank == 0:
            with torch.no_grad():
                if self.one_step and self.image_loss:
                    images = pred_images
                else:
                    if not self.one_step:
                        latents_pred = self.predict_start_from_z_and_v(
                            latents_noisy, t, v_pred
                        )

                    latents = unscale_latents(latents_pred)
                    images = unscale_image(
                        self.pipeline.vae.decode(
                            latents / self.pipeline.vae.config.scaling_factor,
                            return_dict=False,
                        )[0]
                    )  # [-1, 1]
                    images = (images * 0.5 + 0.5).clamp(0, 1)

                # depth, roughness and metallic use mean as output and apply turbo colormap
                row_col = [[1, 1], [2, 0], [2, 1]]
                images = images.detach().clone()
                target_imgs = target_imgs.detach().clone()
                for row, col in row_col:
                    images[
                        :, :, 320 * row : 320 * (row + 1), 320 * col : 320 * (col + 1)
                    ] = images[
                        :, :, 320 * row : 320 * (row + 1), 320 * col : 320 * (col + 1)
                    ][
                        :, 0:1
                    ].repeat(
                        1, 3, 1, 1
                    )
                    target_imgs[
                        :, :, 320 * row : 320 * (row + 1), 320 * col : 320 * (col + 1)
                    ] = target_imgs[
                        :, :, 320 * row : 320 * (row + 1), 320 * col : 320 * (col + 1)
                    ][
                        :, 0:1
                    ].repeat(
                        1, 3, 1, 1
                    )

                    # images[:, :, 320 * row:320*(row+1), 320*col:320*(col+1)] = images[:, :, 320 * row:320*(row+1), 320*col:320*(col+1)][:,0:1].repeat(1,3,1,1)

                if self.train_controlnet:
                    images = torch.cat([ref_imgs, target_imgs, images], dim=-2)
                else:
                    if not self.render_loss:
                        images = torch.cat([target_imgs, images], dim=-2)
                    else:  # vis rendered rgb
                        rendered_vis = torch.cat(
                            [rendered_rgb, extra_dict["gt_rendered_rgb"]], dim=-1
                        )
                        images = torch.cat([target_imgs, images, rendered_vis], dim=-2)

                grid = make_grid(
                    images, nrow=images.shape[0], normalize=True, value_range=(0, 1)
                )
                save_image(
                    grid,
                    os.path.join(
                        self.logdir, "images", f"train_{self.global_step:07d}.png"
                    ),
                )

        return loss

    def compute_loss(self, noise_pred, noise_gt):
        loss = F.mse_loss(noise_pred, noise_gt)

        prefix = "train"
        loss_dict = {}
        loss_dict.update({f"{prefix}/loss": loss})

        return loss, loss_dict

    def compute_loss_image(self, image_pred, image_gt, mask_gt, extra_dict):
        print("--------------image loss----------------")
        prefix = "train"
        loss_dict = {}
        # mse_loss = F.mse_loss(image_pred, image_gt)
        # lpips_loss = self.lpips_loss(image_pred,image_gt,normalize=True).mean()

        # loss = mse_loss + 0.1 * lpips_loss

        # split preds
        domains = self.domains
        # weights=[1.0, 1.0, 0.5, 0.5, 3.0, 3.0]
        weights = self.weights_domains
        weight_lpips = self.weight_lpips
        weight_global_mse = self.weight_global_mse
        weight_render_loss = self.weight_render_loss
        _pred = dict()
        _gt = dict()
        _mask = dict()
        _weight = dict()
        _loss = dict()
        domain_i = 0
        for row in range(3):
            for col in range(2):
                _pred[domains[domain_i]] = image_pred[
                    :, :, 320 * row : 320 * (row + 1), 320 * col : 320 * (col + 1)
                ]
                _gt[domains[domain_i]] = image_gt[
                    :, :, 320 * row : 320 * (row + 1), 320 * col : 320 * (col + 1)
                ]
                _mask[domains[domain_i]] = (
                    mask_gt[
                        :, :, 320 * row : 320 * (row + 1), 320 * col : 320 * (col + 1)
                    ]
                    > 0.5
                )
                _weight[domains[domain_i]] = weights[domain_i]
                # v2.functional.to_pil_image(_gt[domains[domain_i]][1]).save(f'tmp/{domains[domain_i]}_gt.png')
                # v2.functional.to_pil_image(_pred[domains[domain_i]][1]).save(f'tmp/{domains[domain_i]}_pred.png')
                # v2.functional.to_pil_image(_mask[domains[domain_i]][1]).save(f'tmp/{domains[domain_i]}_mask.png')
                domain_i += 1

        # -----------compute loss for each domain ----------
        # -----MSE loss and LPIPS loss for rgb and albedo-----
        for d in ["rgb", "albedo"]:
            mse_loss = F.mse_loss(_pred[d], _gt[d])
            lpips_loss = (
                self.lpips_loss(_pred[d], _gt[d], normalize=True).mean() * weight_lpips
            )
            _loss[d] = (mse_loss + lpips_loss) * _weight[d]
            # print(f"{d} lpips: {lpips_loss}")

        # -----angular loss and MSE loss for normal-----
        d = "normal"
        ang_loss = angular_loss(_pred[d], _gt[d], None)
        lpips_loss = (
            self.lpips_loss(_pred[d], _gt[d], normalize=True).mean() * weight_lpips
        )
        # print(f"normal lpips: {lpips_loss}")
        _loss[d] = (ang_loss + lpips_loss) * _weight[d]

        # # -----affine invariant loss for 1 channel depth-----
        # d = "depth"
        # # convert 3 channel depth to 1 channel
        # pred_depth = _pred[d][:, 0]  # [B, H, W]
        # gt_depth = _gt[d][:, 0]  # [B, H, W]
        # gt_mask = _mask[d][:, 0] > 0.5  # [B, H, W]

        # aligned_pred_depth, _, _ = align_depth_least_square(
        #     gt_depth.cpu().detach().numpy(),
        #     pred_depth.cpu().detach().numpy(),
        #     gt_mask.cpu().detach().numpy(),
        #     pred_depth,
        # )
        # aligned_pred_depth = torch.clamp(aligned_pred_depth, 0, 1)
        # affine_invariant_depth_loss = F.l1_loss(aligned_pred_depth, gt_depth)
        # _loss[d] = affine_invariant_depth_loss * _weight[d]

        # -----MSE loss for 1 channel material maps-----
        for d in ["depth", "roughness", "metallic"]:
            # convert 3 channel depth to 1 channel
            pred_material = _pred[d][:, 0]  # [B, H, W]
            gt_material = _gt[d][:, 0]  # [B, H, W]
            _loss[d] = F.l1_loss(pred_material, gt_material) * _weight[d]

        loss = 0.0
        for d in domains:
            loss += _loss[d]
            # print(f"{d}_loss: {_loss[d]}")

        # global mse loss
        global_mse = F.mse_loss(image_pred, image_gt) * weight_global_mse
        loss += global_mse
        # print(f"global_mse_loss: {global_mse}")

        # render loss
        if self.render_loss:
            render_loss_mse = 0.0
            render_loss_lpips = 0.0
            pred_rendered_rgb_batch = []
            for i in range(image_pred.shape[0]):
                root = "/3d/WonderMaterial/files/hdr/"
                light_name = extra_dict["light_name"][i]
                c2w = extra_dict["c2w"][i]
                data_gt = dict()  # gt_domain_maps
                data_pred = dict()  # gt_domain_maps
                mask = _mask["rgb"][i]

                # ------prepare pred domains for rendering
                for d in domains:
                    data_gt[d] = torch.clamp(_gt[d][i], 0, 1)
                    data_pred[d] = torch.clamp(_pred[d][i], 0, 1)

                for d in domains[-2:]:  # material use 1 channel
                    data_gt[d] = data_gt[d][0:1]
                    data_pred[d] = data_pred[d][0:1]

                # depth to ccm
                if extra_dict["depth_bg_color"][i] == 0:
                    data_gt["depth"] = 1 - data_gt["depth"][0:1]

                    data_pred["depth"] = 1 - data_pred["depth"][0:1]

                data_gt["depth"] = depth_to_ccm(
                    data_gt["depth"], c2w, 2.6 - 1.414, 2.6 + 1.414
                )[
                    0
                ]  # black bg
                data_pred["depth"] = depth_to_ccm(
                    data_pred["depth"], c2w, 2.6 - 1.414, 2.6 + 1.414
                )[
                    0
                ]  # black bg

                # ------prepare gt rendered image
                gt_rendered_rgb = self.image_renderer.render(
                    root + light_name, clone_tensor_dict(data_gt), mask, c2w
                )
                gt_rendered_rgb = gt_rendered_rgb[0].permute(-1, 0, 1).contiguous()
                gt_rendered_rgb = torch.clamp(gt_rendered_rgb, 0, 1)
                extra_dict["gt_rendered_rgb"][i] = gt_rendered_rgb

                # ------render using all pred domains for visualization
                pred_rendered_rgb = self.image_renderer.render(
                    root + light_name, clone_tensor_dict(data_pred), mask, c2w
                )
                pred_rendered_rgb = pred_rendered_rgb[0].permute(-1, 0, 1).contiguous()
                pred_rendered_rgb = torch.clamp(pred_rendered_rgb, 0, 1)
                pred_rendered_rgb_batch.append(pred_rendered_rgb)

                # ------iterate render loss over pred domains
                for d_pred in domains[1:]:
                    if d_pred == "depth":  # skip depth for render loss
                        continue

                    # for domain d_pred, use pred as input
                    data_gt_input = clone_tensor_dict(data_gt)
                    data_gt_input[d_pred] = data_pred[d_pred]

                    # render engine
                    rendered_rgb = self.image_renderer.render(
                        root + light_name, data_gt_input, mask, c2w
                    )
                    rendered_rgb = rendered_rgb[0].permute(-1, 0, 1).contiguous()
                    rendered_rgb = torch.clamp(rendered_rgb, 0, 1)

                    # render loss
                    render_loss_mse += F.mse_loss(rendered_rgb, gt_rendered_rgb)
                    # render_loss_lpips += self.lpips_loss(rendered_rgb, gt_rendered_rgb,normalize=True).mean()

            # batch mean
            render_loss_mse = render_loss_mse / image_pred.shape[0]
            # render_loss_lpips = render_loss_lpips / image_pred.shape[0] * weight_lpips

            # render loss
            render_loss = (render_loss_mse) * weight_render_loss
            # render_loss = (render_loss_mse + render_loss_lpips) * weight_render_loss
            loss += render_loss
            # print(f"render_loss: {render_loss}")
            # import ipdb; ipdb.set_trace()

            loss_dict.update({f"{prefix}/render_loss": render_loss})

        # image loss for each domain
        for d in domains:
            loss_dict.update({f"{prefix}/{d}_loss": _loss[d]})

        if self.render_loss:
            return loss, loss_dict, torch.stack(pred_rendered_rgb_batch, dim=0)
        else:
            return loss, loss_dict

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        # get input
        if self.train_controlnet:
            cond_imgs, target_imgs, ref_imgs = self.prepare_batch_data(batch)
            refs_pil = [
                v2.functional.to_pil_image(ref_imgs[i])
                for i in range(ref_imgs.shape[0])
            ]
        elif self.front_guided:
            (
                cond_imgs,
                target_imgs,
                front_imgs,
                cond_imgs_tiled,
                target_masks,
                extra_dict,
            ) = self.prepare_batch_data(batch)
            front_images_pil = [
                v2.functional.to_pil_image(front_imgs[i])
                for i in range(front_imgs.shape[0])
            ]
            cond_images_tiled_pil = [
                v2.functional.to_pil_image(cond_imgs_tiled[i])
                for i in range(cond_imgs_tiled.shape[0])
            ]
        else:
            cond_imgs, target_imgs = self.prepare_batch_data(batch)

        images_pil = [
            v2.functional.to_pil_image(cond_imgs[i]) for i in range(cond_imgs.shape[0])
        ]
        # target_images_pil = [
        #     v2.functional.to_pil_image(target_imgs[i]) for i in range(target_imgs.shape[0])
        # ]
        # for i in range(len(images_pil)):
        #     images_pil[i].save(f'tmp/cond_img_{i}.png')
        #     target_images_pil[i].save(f'tmp/target_img_{i}.png')
        #     front_images_pil[i].save(f'tmp/front_img_{i}.png')
        #     cond_images_tiled_pil[i].save(f'tmp/cond_tiled_img_{i}.png')
        # import pdb; pdb.set_trace()

        outputs = []
        num_inference_steps = 75
        if self.one_step:
            num_inference_steps = 1
        for i in range(len(images_pil)):
            cond_img = images_pil[i]

            if self.train_controlnet:
                ref_img = refs_pil[i]
                latent = self.pipeline(
                    cond_img,
                    depth_image=ref_img,
                    num_inference_steps=num_inference_steps,
                    output_type="latent",
                ).images
            elif self.front_guided:
                front_img = front_images_pil[i]
                cond_img_tiled = cond_images_tiled_pil[i]
                cond_img_tiled = cond_imgs_tiled[i]  # NOTE:DEBUG
                latent = self.pipeline(
                    cond_img,
                    front_img=front_img,
                    cond_img_tiled=cond_img_tiled,
                    num_inference_steps=num_inference_steps,
                    output_type="latent",
                    bg_color=extra_dict["depth_bg_color"][i],
                    guidance_scale=0.0,
                ).images
            else:
                latent = self.pipeline(
                    cond_img,
                    num_inference_steps=num_inference_steps,
                    output_type="latent",
                ).images

            image = unscale_image(
                self.pipeline.vae.decode(
                    latent / self.pipeline.vae.config.scaling_factor, return_dict=False
                )[0]
            )  # [-1, 1]

            image = (image * 0.5 + 0.5).clamp(0, 1)

            # vis rendered rgb
            if self.render_loss:
                image_pred = image[0]  # [3, 960, 640]
                image_gt = target_imgs[i].clone()  # [3, 960, 640]
                mask_gt = target_masks[i]  # [1, 960, 640]
                # split preds
                domains = self.domains
                _pred = dict()
                _gt = dict()
                _mask = dict()
                domain_i = 0
                for row in range(3):
                    for col in range(2):
                        _pred[domains[domain_i]] = image_pred[
                            :, 320 * row : 320 * (row + 1), 320 * col : 320 * (col + 1)
                        ]
                        _gt[domains[domain_i]] = image_gt[
                            :, 320 * row : 320 * (row + 1), 320 * col : 320 * (col + 1)
                        ]
                        _mask[domains[domain_i]] = (
                            mask_gt[
                                :,
                                320 * row : 320 * (row + 1),
                                320 * col : 320 * (col + 1),
                            ]
                            > 0.5
                        )
                        domain_i += 1

                root = "/3d/WonderMaterial/files/hdr/"
                light_name = extra_dict["light_name"][i]
                c2w = extra_dict["c2w"][i]
                data = dict()
                data_gt = dict()
                mask = _mask["rgb"]
                for d in domains:
                    data[d] = torch.clamp(_pred[d], 0, 1)
                    data_gt[d] = torch.clamp(_gt[d], 0, 1)

                for d in domains[-2:]:  # material use 1 channel
                    data[d] = data[d][0:1]
                    data_gt[d] = data_gt[d][0:1]

                data["depth"] = depth_to_ccm(
                    1 - data["depth"][0:1], c2w, 2.6 - 1.414, 2.6 + 1.414
                )[0]
                rendered_rgb = self.image_renderer.render(
                    root + light_name, data, mask, c2w
                )
                rendered_rgb = rendered_rgb[0].permute(-1, 0, 1).contiguous()
                rendered_rgb = torch.clamp(rendered_rgb, 0, 1)

                # gt
                data_gt["depth"] = depth_to_ccm(
                    1 - data_gt["depth"][0:1], c2w, 2.6 - 1.414, 2.6 + 1.414
                )[0]
                rendered_rgb_gt = self.image_renderer.render(
                    root + light_name, data_gt, mask, c2w
                )
                rendered_rgb_gt = rendered_rgb_gt[0].permute(-1, 0, 1).contiguous()
                rendered_rgb_gt = torch.clamp(rendered_rgb_gt, 0, 1)

                image = torch.cat(
                    [
                        image,
                        torch.cat(
                            [rendered_rgb[None, :], rendered_rgb_gt[None, :]], dim=-1
                        ),
                    ],
                    dim=-2,
                )
            
            # apply turbo colormap on predicted depth, roughness and metallic
            row_col = [[1, 1], [2, 0], [2, 1]]
            for row, col in row_col:
                image[
                    :, :, 320 * row : 320 * (row + 1), 320 * col : 320 * (col + 1)
                ] = image[
                    :, :, 320 * row : 320 * (row + 1), 320 * col : 320 * (col + 1)
                ][
                    :, 0:1
                ].repeat(
                    1, 3, 1, 1
                )

            outputs.append(image)
        outputs = torch.cat(outputs, dim=0).to(self.device)

        # apply turbo colormap on gt depth, roughness and metallic
        row_col = [[1, 1], [2, 0], [2, 1]]
        for row, col in row_col:
            target_imgs[
                :, :, 320 * row : 320 * (row + 1), 320 * col : 320 * (col + 1)
            ] = target_imgs[
                :, :, 320 * row : 320 * (row + 1), 320 * col : 320 * (col + 1)
            ][
                :, 0:1
            ].repeat(
                1, 3, 1, 1
            )

        if self.train_controlnet:
            images = torch.cat([ref_imgs, target_imgs, outputs], dim=-2)
        else:
            images = torch.cat([target_imgs, outputs], dim=-2)

        # target_imgs: [3, 960, 640]
        # outputs: [3, 960+320, 640]
        output_col_num = 4 if self.render_loss else 3
        target_imgs = rearrange_grid(target_imgs, [3, 2])
        outputs = rearrange_grid(outputs, [output_col_num, 2])

        # get the last two sub-images from outputs
        if self.render_loss:
            render_res_list = outputs[:, :, -640:, :].chunk(2, dim=-2)

            # add each to the target_imgs and outputs
            target_imgs = torch.cat([target_imgs, render_res_list[1]], dim=-2)
            outputs = torch.cat([outputs[:, :, :-640, :], render_res_list[0]], dim=-2)

        images = torch.cat([target_imgs, outputs], dim=-1)

        self.validation_step_outputs.append(images)

    @torch.no_grad()
    def on_validation_epoch_end(self):
        images = torch.cat(self.validation_step_outputs, dim=0)

        all_images = self.all_gather(images)
        all_images = rearrange(all_images, "r b c h w -> (r b) c h w")

        if self.global_rank == 0:
            grid = make_grid(all_images, nrow=8, normalize=True, value_range=(0, 1))
            save_image(
                grid,
                os.path.join(
                    self.logdir, "images_val", f"val_{self.global_step:07d}.png"
                ),
            )

        self.validation_step_outputs.clear()  # free memory
        if not self.train_controlnet:
            self.pipeline.unet.unet.save_pretrained(
                os.path.join(
                    # self.pipeline.unet.save_pretrained(os.path.join(
                    self.logdir,
                    f"unet_{self.global_step:07d}",
                )
            )
        else:
            self.pipeline.unet.controlnet.save_pretrained(
                os.path.join(self.logdir, f"controlnet_{self.global_step:07d}")
            )

        if self.finetune_vae_decoder:
            self.pipeline.vae.save_pretrained(
                os.path.join(self.logdir, f"vae_{self.global_step:07d}")
            )

    def configure_optimizers(self):
        lr = self.learning_rate

        if self.finetune_vae_decoder:
            # import ipdb; ipdb.set_trace()
            optimized_params = [
                {"params": self.unet.parameters()},
                {"params": self.vae_post_quant_conv.parameters()},
                {"params": self.vae_decoder.parameters()},
            ]
            optimizer = torch.optim.AdamW(optimized_params, lr=lr)
        else:
            optimizer = torch.optim.AdamW(self.unet.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 3000, eta_min=lr / 4
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
