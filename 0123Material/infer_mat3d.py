import os
import tyro
import glob
import imageio
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from safetensors.torch import load_file
import rembg

import kiui
# from kiui.op import recenter
# from kiui.cam import orbit_camera
# from core.utils import spherical_camera_pose

from core.options import Options
# from core.models import LGM

# from mvdream.pipeline_mvdream import MVDreamPipeline

from PIL import Image
from einops import rearrange, repeat

from diffusers import (
    DiffusionPipeline,
    EulerAncestralDiscreteScheduler,
    ControlNetModel,
    UNet2DConditionModel,
)
from huggingface_hub import hf_hub_download
from core.utils import remove_background, resize_foreground
from torchvision import transforms
from torchvision.transforms import v2

import json
from core.utils import _locate_datadir
from safetensors.torch import load_file


opt = tyro.cli(Options)

# set seed
kiui.utils.seed_everything(opt.seed)


# # device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# load multi-view diffusion model
print("Loading multi-view diffusion model ...")
if opt.mv_model_name == "zero123plus" and opt.pred_mv:
    pipeline = DiffusionPipeline.from_pretrained(
        "sudo-ai/zero123plus-v1.2",
        custom_pipeline=opt.custom_pipeline,
        torch_dtype=torch.float16,
    )

    # load custom background unet
    if opt.mv_unet_path is not None:
        # assert os.path.exists(opt.mv_unet_path)
        print(f"Loading custom mv unet from {opt.mv_unet_path}...")
        if os.path.exists(opt.mv_unet_path):
            opt.mv_unet_path = opt.mv_unet_path
            # import ipdb; ipdb.set_trace()
        else:
            opt.mv_unet_path = hf_hub_download(
                repo_id="TencentARC/InstantMesh",
                filename="diffusion_pytorch_model.bin",
                repo_type="model",
            )

        if opt.mv_unet_path.endswith(".safetensors"):
            state_dict = load_file(opt.mv_unet_path)
        elif opt.mv_unet_path.endswith(".bin"):
            state_dict = torch.load(opt.mv_unet_path)
        else:
            raise NotImplementedError
        pipeline.unet.load_state_dict(state_dict, strict=True)

    pipe = pipeline
    pipe = pipe.to(device)

# load domain transfer unet
if opt.domain_transfer_unet_path is not None:
    pipe2 = DiffusionPipeline.from_pretrained(
        "sudo-ai/zero123plus-v1.2",
        custom_pipeline=opt.custom_pipeline,
        torch_dtype=torch.float16,
    )
    print(f"Loading domain transfer unet from {opt.domain_transfer_unet_path}...")
    assert os.path.exists(opt.domain_transfer_unet_path)

    if opt.domain_transfer_unet_path.endswith(".safetensors"):
        state_dict = load_file(opt.domain_transfer_unet_path)
    elif opt.domain_transfer_unet_path.endswith(".bin"):
        state_dict = torch.load(opt.domain_transfer_unet_path)
    else:
        raise NotImplementedError

    pipe2.unet.load_state_dict(state_dict, strict=True)
    pipe2.to(device)

# load domain transfer vae
if opt.domain_transfer_vae_path is not None:
    print(f"Loading domain transfer vae from {opt.domain_transfer_vae_path}...")
    assert os.path.exists(opt.domain_transfer_vae_path)

    if opt.domain_transfer_vae_path.endswith(".safetensors"):
        state_dict = load_file(opt.domain_transfer_vae_path)
    elif opt.domain_transfer_vae_path.endswith(".bin"):
        state_dict = torch.load(opt.domain_transfer_vae_path)
    else:
        raise NotImplementedError

    pipe2.vae.load_state_dict(state_dict, strict=True)
    pipe2.to(device)


    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipeline.scheduler.config, timestep_spacing="trailing"
    )


# else:
#     raise ValueError(f"Unsupported mv_model_name: {opt.mv_model_name}")


# load rembg
bg_remover = None if opt.no_rembg else rembg.new_session()


def _load_rgba_image(
    file_path,
    bg_color: float = 1.0,
    ref_map_name=None,
    resize_fg_ratio=None,
    use_objaverse_retrived_ref_maps=False,
):
    """Load and blend RGBA image to RGB with certain background, 0-1 scaled"""
    img = Image.open(file_path)
    if resize_fg_ratio is not None:
        img = resize_foreground(img, resize_fg_ratio)
    rgba = np.array(img)
    rgba = torch.from_numpy(rgba).float() / 255.0
    rgba = rgba.permute(2, 0, 1).contiguous()
    mask = rgba[3:4]  # [1, 512, 512]
    rgb = rgba[:3, :, :]
    rgb = rgb * mask + bg_color * (1 - mask)
    return rgb, mask


# process function
def process(opt: Options, path):
    if isinstance(path, dict):
        uid = path["uid"]
        root_dir = path["root_dir"]
        path = os.path.join(root_dir, uid, "rgba", "000.png")
        name = uid
    else:
        name = os.path.splitext(os.path.basename(path))[0]

    # skip generated
    # if os.path.exists(os.path.join(opt.workspace, name + ".mp4")):
    #     print(f"[INFO] Skip processing {path} --> {name}")
    #     return

    print(f"[INFO] Processing {path} --> {name}")
    os.makedirs(opt.workspace, exist_ok=True)

    # remove background optionally
    input_image = Image.open(path)
    if not opt.no_rembg:
        print("remove background")
        input_image = remove_background(input_image, bg_remover)
        input_image = resize_foreground(input_image, opt.resize_fg_ratio)

    # 1. pred mv images
    if opt.pred_mv:
        # raise NotImplementedError()
        if opt.use_exist_mv is not None:  # TODO: for zero123plus mv; debug only; remove it
            if os.path.isdir(opt.use_exist_mv):
                mv_image = Image.open(
                    os.path.join(opt.use_exist_mv, name + ".png")
                ).convert("RGB")
            else:
                mv_image = Image.open(opt.use_exist_mv).convert("RGB")

            mv_image = (
                np.asarray(mv_image, dtype=np.float32) / 255.0
            )  # (256, 256*num_views, 3)
            mv_image = rearrange(
                mv_image, "(n h) (m w) c -> (n m) h w c", m=1, n=opt.num_input_views
            )  # (num_views, 256, 256, 3)
            # mv_image = rearrange(
            #     mv_image, "(n h) (m w) c -> (n m) h w c", n=3, m=2
            # )  # (num_views, 256, 256, 3)
        else:
            if opt.mv_model_name == "zero123plus":
                
                mv_image = pipe(
                    input_image,
                    num_inference_steps=opt.diffusion_steps,
                ).images[0]

                mv_image = np.asarray(mv_image, dtype=np.uint8)
                mv_image = rearrange(
                    mv_image, "(n h) (m w) c -> (n m) h w c", n=3, m=2
                )  # (num_views, 320, 320, 3)
                print(mv_image.shape)
            else:
                raise ValueError(f"Unsupported mv_model_name: {opt.mv_model_name}")
    else:
        # not pred mv images, use input image as mv images
        # convert input_image with 4 channel to white background rgb
        # tmp = Image.new("RGB", input_image.size, (255, 255, 255))
        # tmp.paste(input_image, (0, 0), input_image)
        # mv_image = [np.array(tmp.resize([opt.input_size,opt.input_size], resample=Image.LANCZOS).convert("RGB"))]

        mv_image = torchvision.io.read_image(path)

        # resize to 320x320
        mv_image = rearrange(mv_image, "c h (n w) -> n c h w", n=4)
        mv_image = F.interpolate(mv_image, size=(320, 320), mode="bilinear", align_corners=False)
        mv_image = rearrange(mv_image, "n c h w -> n h w c").numpy()

        # use mask to blend with white background
        if mv_image.shape[-1] == 4:
            mask = mv_image[:, :, :, 3:4]
            binary_mask = (mask > 0.5 * mask.max()).astype(np.float32)
            mv_image = mv_image[:, :, :, :3] * binary_mask + 255 * (1 - binary_mask)

        # debug save image
        # tmp = Image.fromarray(mv_image[0].astype(np.uint8))
        # tmp.save('debug.png')

    if opt.two_step_inference:
        if opt.front_guided:
            front_img = input_image
        else:
            front_img = None
        # domain transfer for each of the 6 views, using pipe2
        count = 0
        tranfered_views = []
        for view_i in mv_image:
            cond_img = torch.from_numpy(view_i).float() / 255.0
            cond_img = cond_img.permute(2, 0, 1).contiguous()
            cond_img_tiled = cond_img.unsqueeze(0).expand(6, -1, -1, -1)
            # to (C, 3H, 2W)
            cond_img_tiled = rearrange(
                cond_img_tiled, "(x y) c h w -> c (x h) (y w)", x=3, y=2
            )

            view_i = Image.fromarray((view_i).astype(np.uint8))
            view_i = view_i.convert("RGB")
            count += 1
            tranfered_view_i = pipe2(
                view_i,
                front_img=front_img,
                cond_img_tiled=cond_img_tiled,
                num_inference_steps=opt.second_step_diffusion_steps,
                bg_color=opt.bg_color,
                guidance_scale=0.0
            ).images[0]
            # import ipdb; ipdb.set_trace()

            tranfered_view_i = (
                np.asarray(tranfered_view_i, dtype=np.float32) / 255.0
            )
            tranfered_view_i = rearrange(
                tranfered_view_i, "(n h) (m w) c -> (h) (n m w) c", n=3, m=2
            )  # (320, num_domains * 320, 3)
            for col in range(3,6):
                tranfered_view_i[:, col * 320 : (col + 1) * 320, :] =  np.tile(tranfered_view_i[:, col * 320 : (col + 1) * 320, :][:,:,0:1], (1,1,3))
            tranfered_views.append(tranfered_view_i)

        tranfered_views = np.stack(
            tranfered_views, axis=0
        )  # [num_views, 320, num_domains * 320, 3]
        tranfered_views = rearrange(
            tranfered_views, "n h w c -> (n h) w c", n=len(mv_image)
        )  # [num_views * 320, num_domains * 320, 3]

    if opt.pred_mv:
        kiui.write_image(
            os.path.join(opt.workspace, name + ".png"),
            mv_image.reshape(-1, mv_image.shape[1], 3),
        )
        print(f"MV image saved to {os.path.join(opt.workspace, name + '.png')}")

    if opt.two_step_inference:
        kiui.write_image(
            os.path.join(opt.workspace, name + "_transfered.png"), tranfered_views
        )
        print(
            f"Transfered MV image saved to {os.path.join(opt.workspace, name + '_transfered.png')}"
        )


assert opt.test_path is not None

if opt.test_path.endswith(".json"):  # objaverse uids
    test_uids = json.load(open(opt.test_path))
    file_paths = []
    for uid in test_uids:
        root_dir = _locate_datadir(opt.root_dirs, uid, locator="intrinsics.npy")
        file_paths.append({"uid": uid, "root_dir": root_dir})
else:
    if os.path.isdir(opt.test_path):
        file_paths = glob.glob(os.path.join(opt.test_path, "*"))
    
    elif os.path.isfile(opt.test_path):
        file_paths = [opt.test_path]

    else:
        # is glob
        file_paths = glob.glob(opt.test_path.replace("'", ""))

print(f"Processing {len(file_paths)} files: {opt.test_path}")

for path in file_paths:
    process(opt, path)
