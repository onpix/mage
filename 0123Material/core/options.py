import tyro
from dataclasses import dataclass
from typing import Tuple, Literal, Dict, Optional


@dataclass
class Options:
    # ### LGM model
    # # Unet input channel, image 3, ray 6, exclude ref maps
    # unet_in_channel: int = 9
    # # Unet image input size
    input_size: int = 320 # 256
    # # Unet definition
    # down_channels: Tuple[int, ...] = (64, 128, 256, 512, 1024, 1024)
    # down_attention: Tuple[bool, ...] = (False, False, False, True, True, True)
    # mid_attention: bool = True
    # up_channels: Tuple[int, ...] = (1024, 1024, 512, 256)
    # up_attention: Tuple[bool, ...] = (True, True, True, False)
    # # Unet output size, dependent on the input_size and U-Net structure!
    # splat_size: int = 160 # 64
    # # gaussian render size
    # output_size: int = 256

    # ### dataset
    # # train image path
    # train_path: str = (
    #     "/mnt/petrelfs/wangzhenwei/projects/ThemeStationPro/InstantGTA3/data_train/Objaverse/rendered/pv6_ft_poor_empty1_med5_thin15_noscan_lgmv1/train.json"
    # )
    # # val image path
    # val_path: str = (
    #     "/mnt/petrelfs/wangzhenwei/projects/ThemeStationPro/InstantGTA3/data_train/Objaverse/rendered/pv6_ft_poor_empty1_med5_thin15_noscan_lgmv1/val.json"
    # )
    # # rendering root dir
    # root_dirs: Tuple[str, ...] = (
    #     "/mnt/hwfile/3dobject_aigc/wangzhenwei/datasets/objaverse-render-lgm-div2/beta",
    #     "/mnt/hwfile/3dobject_aigc/wangzhenwei/datasets/objaverse-render-lgm/beta/",
    # )
    # # data mode (only support s3 now)
    # data_mode: Literal["s3"] = "s3"
    # # fovy of the dataset
    # fovy: float = 30  # 49.1 # 67.38013718106679
    # # camera near plane 
    # znear: float = 0.5
    # # camera far plane
    # zfar: float = 2.5
    # # number of all views (input + output)
    # num_views: int = 12
    # # number of views
    # num_input_views: int = 6
    # # camera radius
    # cam_radius: float = 1.86603  # to better use [-1, 1]^3 space
    # # num workers
    # num_workers: int = 8
    # # reference maps type, in 'ccm', 'normal', None
    # ref_maps: Optional[Tuple[str, ...]] = None # ("ccm", "normal")

    # # reference maps downsample size
    # ref_map_downsample_size: int = 24

    # ### training
    # # seed
    seed: int = 42
    # workspace
    workspace: str = "./workspace"
    # # resume
    # resume: Optional[str] = None
    # # batch size (per-GPU)
    # batch_size: int = 8
    # # gradient accumulation
    # gradient_accumulation_steps: int = 1
    # # training epochs
    # num_epochs: int = 60
    # # lpips loss weight
    # lambda_lpips: float = 1.0
    # # gradient clip
    # gradient_clip: float = 1.0
    # # mixed precision
    # mixed_precision: str = "bf16"
    # # learning rate
    # lr: float = 4e-4
    # # augmentation 
    # prob_z123_views: float = 0.2
    # # augmentation prob for resize
    # prob_resize: float = 0
    # # augmentation prob for shift
    # prob_shift: float = 0
    # # augmentation prob for grid distortion
    # prob_grid_distortion: float = 0.5
    # # augmentation prob for camera jitter
    # prob_cam_jitter: float = 0.5
    # # resize ratio
    # resize_ratio: float = 0.95
    # # shift offset
    # shift_offset: int = 8
    # # camera jitter skip first view
    # cam_jitter_skip_1st_view: bool = False
    # # normalized camera feature
    # normalize_cam_feat: bool = False
    # # checkpoint interval (step)
    # checkpoint_interval: int = 120
    # # log interval
    # log_interval: int = 100
    # # log image interval
    # log_image_interval: int = 100

    ### testing
    # multi-view model name
    mv_model_name: str = "zero123plus" # "zero123plus" | "mvdream"
    # custom pipeline
    custom_pipeline: str = "zero123plus_material" # "zero123plus" | "z123p_meta_ccm_dt"
    # multi-view unet path -> zero123plus
    # mv_unet_path: str = "/mnt/petrelfs/wangzhenwei/projects/ThemeStationPro/InstantMesh/logs/zero123plus-finetune-unit_sphere_no_ccm/checkpoints/step=00014000.ckpt"
    mv_unet_path: Optional[str] = "None"
    domain_transfer_unet_path: Optional[str] = None
    domain_transfer_vae_path: Optional[str] = None
    # # controlnet path
    # mv_controlnet_path:  Optional[str] = None
    # # reference controlnet
    # mv_controlnet_ref_maps: Optional[Tuple[str, ...]] = None # ("ccm", "rgba")
    # # controlnet_conditioning_scale
    # controlnet_conditioning_scale: float = 0.75
    # # use meta-control or not
    # use_meta_control: bool = True

    # not remove background
    no_rembg: bool = False
    # resize foreground ratio
    resize_fg_ratio: float = 0.7
    # use retrieval
    # use_retrieval: bool = False

    # diffusion steps
    diffusion_steps: int = 75
    # test image path
    test_path: Optional[str] = None
    # using existing multi-view images
    use_exist_mv: Optional[str] = None

    # predict type: rgb/ccm/normal
    # pred_type: Optional[str] = None # rgb|normal|ccm|None
    # # dynamic-timestep-aware reference resolution
    # downsample_size: Optional[Tuple[int, ...]] = (16,32,48)
    # downsample_size_rgba: Optional[Tuple[int, ...]] = (16,32,48)

    # two step inference: pred mv rgb -> domain transfer
    two_step_inference: bool = True
    # use front rgb to guide the domain transfer stage
    front_guided: bool = True
    bg_color: float = 1.0
    second_step_diffusion_steps: int = 1
    # whether to perform multi-view prediction to generate target views or use the input image as target view
    pred_mv: bool = True 

    ### misc
    # nvdiffrast backend setting
    # force_cuda_rast: bool = False
    # render fancy video with gaussian scaling effect
    # fancy_video: bool = False


# # all the default settings
# config_defaults: Dict[str, Options] = {}
# config_doc: Dict[str, str] = {}

# config_doc["mv"] = "the default settings for LGM"
# config_defaults["mv"] = Options()

# config_doc["sv"] = "small model with lower resolution Gaussians"
# config_defaults["sv"] = Options(
#     pred_mv=False,
# )

# config_doc["small"] = "small model with lower resolution Gaussians"
# config_defaults["small"] = Options(
#     input_size=256,
#     splat_size=64,
#     output_size=256,
#     batch_size=8,
#     gradient_accumulation_steps=1,
#     mixed_precision="bf16",
# )

# config_doc["big"] = "big model with higher resolution Gaussians"
# config_defaults["big"] = Options(
#     input_size=256,
#     up_channels=(1024, 1024, 512, 256, 128),  # one more decoder
#     up_attention=(True, True, True, False, False),
#     splat_size=128,
#     output_size=512,  # render & supervise Gaussians at a higher resolution.
#     batch_size=8,
#     num_views=12, # 8
#     num_input_views=6,
#     gradient_accumulation_steps=1,
#     mixed_precision="bf16",
# )

# config_doc["tiny"] = "tiny model for ablation"
# config_defaults["tiny"] = Options(
#     input_size=256,
#     down_channels=(32, 64, 128, 256, 512),
#     down_attention=(False, False, False, False, True),
#     up_channels=(512, 256, 128),
#     up_attention=(True, False, False, False),
#     splat_size=64,
#     output_size=256,
#     batch_size=24,
#     num_views=8,
#     gradient_accumulation_steps=1,
#     mixed_precision="bf16",
# )

# AllConfigs = tyro.extras.subcommand_type_from_defaults(config_defaults, config_doc)
# AllConfigs = Options
