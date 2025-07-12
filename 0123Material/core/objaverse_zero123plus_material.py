import os
import json
import numpy as np
import webdataset as wds
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
from pathlib import Path

from utils.zero123plus_train_util import instantiate_from_config

from megfile import smart_open, smart_path_join, smart_exists, smart_listdir
from megfile.errors import S3UnknownError
import time
import torch.nn.functional as F
import random
from core.utils import grid_distortion, MyRandomPhotometricDistort
from utils.proxy import no_proxy
from scripts.ccm_to_depth import ccm_to_depth
import yaml
class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(
        self,
        batch_size=8,
        num_workers=4,
        train=None,
        validation=None,
        test=None,
        **kwargs,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dataset_configs = dict()
        if train is not None:
            self.dataset_configs["train"] = train
        if validation is not None:
            self.dataset_configs["validation"] = validation
        if test is not None:
            self.dataset_configs["test"] = test

    def setup(self, stage):

        if stage in ["fit"]:
            self.datasets = dict(
                (k, instantiate_from_config(self.dataset_configs[k]))
                for k in self.dataset_configs
            )
        else:
            raise NotImplementedError

    def train_dataloader(self):

        sampler = DistributedSampler(self.datasets["train"])
        return wds.WebLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            sampler=sampler,
        )

    def val_dataloader(self):

        sampler = DistributedSampler(self.datasets["validation"])
        return wds.WebLoader(
            self.datasets["validation"],
            batch_size=4,
            num_workers=self.num_workers,
            shuffle=False,
            sampler=sampler,
        )

    def test_dataloader(self):

        return wds.WebLoader(
            self.datasets["test"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )


class ObjaverseData(Dataset):
    def __init__(
        self,
        root_dir="objaverse/",
        meta_fname="valid_paths.json",
        ref_maps=None,
        cond_map=None,
        target_map=None,
        downsample_size=None,
        upsample_size=320,
        prob_predict_ref_map=None,
        prob_grid_distortion=None,
        prob_resize=None,
        prob_flip=None,
        prob_rand_downsample_size=None,
        prob_shift=None,
        prob_view_dropout=None,
        prob_color_jitter=None,
        rand_downsample_offset=4,
        shift_offset=16,
        brightness=1.0,
        contrast=1.0,
        saturation=1.0,
        hue=0,
        prob_channel_permutation=0.5,
        front_guided=False,
        bg_color=1.0,
        rand_view=False,
    ):
        self.root_dir = root_dir
        self.ref_maps = ref_maps
        self.cond_map = cond_map
        self.target_map = target_map
        self.downsample_size = downsample_size
        self.upsample_size = upsample_size
        self.prob_predict_ref_map = prob_predict_ref_map
        self.prob_grid_distortion = prob_grid_distortion
        self.prob_resize = prob_resize
        self.prob_flip = prob_flip
        self.prob_rand_downsample_size = prob_rand_downsample_size
        self.prob_shift = prob_shift
        self.prob_view_dropout = prob_view_dropout
        self.prob_color_jitter = prob_color_jitter
        self.front_guided = front_guided
        self.bg_color = bg_color
        self.rand_view = rand_view

        self.rand_downsample_offset = rand_downsample_offset
        self.shift_offset = shift_offset
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.prob_channel_permutation = prob_channel_permutation
        self.c2w_dict=json.load(open('./scripts/poses.json'))

        self.items = json.load(open(meta_fname, "r"))

        print("============= length of dataset %d =============" % len(self.items))

    def __len__(self):
        return len(self.items)

    @staticmethod
    def _locate_datadir(root_dirs, uid, locator: str = None):
        while True:
            try:
                for root_dir in root_dirs:
                    datadir = smart_path_join(root_dir, uid)
                    if smart_exists(datadir) and len(smart_listdir(datadir))>=36:
                        return root_dir
                raise FileNotFoundError(
                    f"Cannot find valid data directory for uid {uid}"
                )
            #  megfile.errors.S3UnknownError
            except S3UnknownError:
                print(f"S3UnknownError: {datadir}")
                time.sleep(5)

    @staticmethod
    def _load_rgba_image(
        file_path,
        bg_color: float = 1.0,
        cond_mask=None,
        ref_map_name=None,
        ref_size=None,
        flip_flag=False,
        drop_view=False,
        shift=None,
        color_jitter_dict=None,
    ):
        ori_img = Image.open(smart_open(file_path, "rb"))
        img = ori_img.copy()

        # random dropout the image content
        if drop_view:
            img = Image.new("RGBA", img.size, (0, 0, 0, 0))
        else:
            """Load and blend RGBA image to RGB with certain background, 0-1 scaled"""
            if flip_flag:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            # random resize to adapt to different reference size
            if ref_map_name is not None and ref_size is not None:
                original_size = img.size
                tmp = Image.new(
                    "RGBA", (original_size[0], original_size[1]), (0, 0, 0, 0)
                )
                # resize
                img = img.resize((ref_size, ref_size), Image.BILINEAR)
                offset = (original_size[0] - ref_size) // 2
                tmp.paste(img, (offset, offset))
                img = tmp

            # random shift
            if shift is not None:
                img = img.transform(
                    img.size,
                    Image.AFFINE,
                    (1, 0, shift[0], 0, 1, shift[1]),
                    resample=Image.BILINEAR,
                )

        def rgba_to_rgb(rgba, bg_color):
            rgba = np.array(rgba)
            rgba = torch.from_numpy(rgba).float() / 255.0
            rgba = rgba.permute(2, 0, 1).contiguous()
            if cond_mask is not None:
                mask=cond_mask
            else:
                mask = rgba[3:4]  # [1, 512, 512]
            rgb = rgba[:3, :, :]

            # color jitter on rgb
            if color_jitter_dict is not None and ref_map_name == "rgba":
                params = color_jitter_dict["color_jitter_params"]
                color_jitter = color_jitter_dict["color_jitter"]
                rgb, params = color_jitter(inputs=rgb, params=params)
                color_jitter_dict["color_jitter_params"] = params

            # TODO: remove lines rescale ccm value after re-rendering ccm correctly
            if ref_map_name is not None and ref_map_name == "ccm":
                rgb = rgb * 2.0 - 0.5  # rescale [0.25, 0.75] to [0, 1]

            rgb = rgb * mask + bg_color * (1 - mask)
            return rgb, mask

        ori_rgb, ori_mask = rgba_to_rgb(ori_img, bg_color)
        rgb, mask = rgba_to_rgb(img, bg_color)
        return rgb, mask, ori_rgb, ori_mask

    # def load_im(self, path, color):
    #     pil_img = Image.open(path)

    #     image = np.asarray(pil_img, dtype=np.float32) / 255.
    #     alpha = image[:, :, 3:]
    #     image = image[:, :, :3] * alpha + color * (1 - alpha)

    #     image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
    #     alpha = torch.from_numpy(alpha).permute(2, 0, 1).contiguous().float()
    #     return image, alpha

    @no_proxy
    def __getitem__(self, index):
        uid = self.items[index]
        root_dir = self._locate_datadir(self.root_dir, uid)

        # view_i.rgb  -> view_i.[rgb, albedo, normal, roughness, metallic, CCM]
        if self.target_map == 'tailed_multi_domain':
            bg_color = self.bg_color  # white

            # random view id
            if self.rand_view:
                vid=random.randint(1,6)
            else:
                vid=1

            # read mask
            mask_path = smart_path_join(root_dir, uid, f"mask_000_{vid:1d}.png")
            rgb_mask = np.array(Image.open(smart_open(mask_path, "rb")))
            rgb_mask = torch.from_numpy(rgb_mask).float() / 255.0
            rgb_mask = rgb_mask[None,:]

            # read cond img (rgb)
            cond_img_list=[]
            rgba_path = smart_path_join(root_dir, uid, f"rgb_000_{vid:1d}.png")
            rgb_image, _, _, _ = self._load_rgba_image(
                rgba_path, bg_color=bg_color, cond_mask=rgb_mask
            )
            cond_img_list.append(rgb_image)
                
            cond_imgs = torch.stack(cond_img_list, dim=0).float()
           
            # read target img (6 domains)
            target_img_list=[]
            # 1.rgb
            target_img_list.append(rgb_image)
            # 2.albedo
            path = smart_path_join(root_dir, uid, f"albedo_000_{vid:1d}.png")
            image, _, _, _ = self._load_rgba_image(
                path, bg_color=bg_color, cond_mask=rgb_mask
            )
            target_img_list.append(image)
            # 3.normal
            path = smart_path_join(root_dir, uid, f"normals_000_{vid:1d}.png")
            image, _, _, _ = self._load_rgba_image(
                path, bg_color=bg_color, cond_mask=rgb_mask
            )
            target_img_list.append(image)
            # 4.ccm (depth)
            path = smart_path_join(root_dir, uid, f"ccm_000_{vid:1d}0001.png")
            ccm, _, _, _ = self._load_rgba_image(
                path, bg_color=bg_color, cond_mask=rgb_mask
            )
            # x z -y to x y z & [0,1] to [-1,1]
            ccm = ccm * 2 - 1
            tmp = ccm[1].clone()
            ccm[1] = -ccm[2]
            ccm[2] = tmp
            # ccm to depth
            depth = ccm_to_depth(ccm, torch.tensor(self.c2w_dict[str(vid)]), 2.6-1.414, 2.6+1.414)
            # white bg
            if bg_color == 0:
                depth = 1 - depth
            depth = depth * rgb_mask + bg_color * (1 - rgb_mask)
            depth = depth.repeat(3,1,1).contiguous()
            target_img_list.append(depth)
            # 5-6.roughness and meltallic
            path = smart_path_join(root_dir, uid, f"ks_000_{vid:1d}.png")
            ks_map, _, _, _ = self._load_rgba_image(
                path, bg_color=bg_color, cond_mask=rgb_mask
            )
            roughness = ks_map[1:2,:,:].repeat(3,1,1).contiguous()
            meltallic = ks_map[2:3,:,:].repeat(3,1,1).contiguous()
            target_img_list.append(roughness)
            target_img_list.append(meltallic)

            target_imgs = torch.stack(target_img_list, dim=0).float()

            # print(f' cond_imgs[0]:{ cond_imgs[0].shape}')
            # print(f' target_imgs:{ target_imgs.shape}')
            # exit()
            data = {
                "cond_imgs": cond_imgs[0],  # (3, H, W)
                "target_imgs": target_imgs,  # (6, 3, H, W)
                "target_masks": rgb_mask, # [1, 512, 512]
            }

            # NOTE: light name & c2w for render loss
            # read lignt_name
            path = smart_path_join(root_dir, uid, f"metadata.yaml")
            data['light_name']=yaml.load(open(path), Loader=yaml.FullLoader)['envmap']
            # data['light_name']=yaml.load(open(path), Loader=yaml.FullLoader)['envmap']
            data['c2w']=torch.tensor(self.c2w_dict[str(vid)])
            data['uid']=uid
            # black background rgb for render loss computation
            data['gt_rendered_rgb'] = rgb_image * rgb_mask + 0.0 * (1 - rgb_mask) # (3, H, W)
            data['depth_bg_color'] = bg_color

            if self.front_guided:
                # read front img (rgb)
                front_img_list=[]
                front_path = smart_path_join(root_dir, uid, f"rgb_000_0.png")
                front_image, _, _, _ = self._load_rgba_image(
                    front_path, bg_color=bg_color
                )
                front_img_list.append(front_image)
                front_imgs = torch.stack(front_img_list, dim=0).float()

                data['front_imgs'] = front_imgs[0]  # (3, H, W)

        else:
            # read cond img
            cond_img_list = []
            cond_mask_list = []
            bg_color = self.bg_color  # white
            if len(self.cond_map)==1:
                cond_map=self.cond_map[0]
            else:
                # random
                cond_map = random.choice(self.cond_map)
            for vid in [0,1,2,3,4,5,6]:
                if cond_map=='normals':
                    rgba_path = smart_path_join(root_dir, uid, f"rgb_000_{vid:1d}.png")
                    _, cond_mask, _, _ = self._load_rgba_image(
                        rgba_path
                    )
                else:
                    cond_mask=None
                
                cond_path = smart_path_join(root_dir, uid, f"{cond_map}_000_{vid:1d}.png")
                image, mask, ori_image, ori_mask = self._load_rgba_image(
                    cond_path, bg_color=bg_color,cond_mask=cond_mask
                )
                cond_img_list.append(image)
                cond_mask_list.append(mask)
                
            cond_imgs = torch.stack(cond_img_list, dim=0).float()

            # read target imgs
            if self.target_map == 'same_as_cond':
                target_imgs = cond_imgs[1:]  # (6, 3, H, W)
            else:
                img_list = []
                bg_color = self.bg_color  # white
                for vid in [1,2,3,4,5,6]:
                    rgba_path = smart_path_join(root_dir, uid, f"{self.target_map}_000_{vid:1d}.png")
                    if self.target_map == 'normals':
                        cond_mask=cond_mask_list[vid]
                    else:
                        cond_mask=None
                    image, mask, ori_image, ori_mask = self._load_rgba_image(
                        rgba_path, bg_color=bg_color, cond_mask=cond_mask
                    )
                    img_list.append(image)

                imgs = torch.stack(img_list, dim=0).float()

                target_imgs=imgs

            data = {
                "cond_imgs": cond_imgs[0],  # (3, H, W)
                "target_imgs": target_imgs,  # (6, 3, H, W)
            }
        # read reference maps (controlnet conditions)
        if self.ref_maps is not None:
            # predefine augmentation for each view
            # random resize
            # if self.prob_resize is not None and random.random() < self.prob_resize:
            #     ref_size = np.random.randint(256, 513)
            # else:
            #     ref_size = None
            # drop_view = []
            # shift = []
            vid_list = [1,2,3,4,5,6]
            # for vid in vid_list:
                # dropout view
                # if vid == 0:  # keep front ref
                #     drop_view.append(False)
                # else:
                #     if (
                #         self.prob_view_dropout is not None
                #         and random.random() < self.prob_view_dropout
                #     ):
                #         drop_view.append(True)
                #     else:
                #         drop_view.append(False)

                # # random shift
                # if self.prob_shift is not None and random.random() < self.prob_shift:
                #     shift.append(
                #         np.random.randint(-self.shift_offset, self.shift_offset, size=2)
                #     )
                # else:
                #     shift.append(None)

            ref_imgs = []
            # ori_ref_imgs = []
            for ref_map in self.ref_maps:
                if ref_map == 'rgb':
                    refs = cond_imgs[1:]  # (6, 3, H, W)
                    ref_imgs.append(refs)
                else:
                    # 6 surrounding views of reference maps (ccm/normal) for controlnet zero123plus
                    refs = []
                    ori_refs = []
                    bg_color = self.bg_color # white
                    # random color jitter
                    # if ref_map == "rgba":
                    #     if (
                    #         self.prob_color_jitter is not None
                    #         and random.random() < self.prob_color_jitter
                    #     ):
                    #         color_jitter_dict = dict()
                    #         color_jitter_dict["color_jitter"] = MyRandomPhotometricDistort(
                    #             brightness=self.brightness,
                    #             contrast=self.contrast,
                    #             saturation=self.saturation,
                    #             hue=self.hue,
                    #             p=self.prob_channel_permutation,
                    #         )
                    #         color_jitter_dict["color_jitter_params"] = None
                    # else:
                    #     color_jitter_dict = None

                    for v_ix in range(len(vid_list)):
                        vid = vid_list[v_ix]
                        ref_dir = root_dir

                        ref_path = smart_path_join(ref_dir, uid, f"{ref_map}_000_{vid:1d}.png")

                        if ref_map == 'normals':
                            cond_mask=cond_mask_list[vid]
                        else:
                            cond_mask=None
                        image, mask, ori_image, ori_mask = self._load_rgba_image(
                            ref_path,
                            bg_color=bg_color,
                            cond_mask=cond_mask,
                            # ref_map_name=ref_map,
                            # ref_size=ref_size,
                            # drop_view=drop_view[v_ix],
                            # shift=shift[v_ix],
                            # color_jitter_dict=color_jitter_dict,
                        )
                        refs.append(image)  # 21 images
                        ori_refs.append(ori_image)  # 21 images
                    refs = torch.stack(refs, dim=0).float()  # (7, 3, H, W)
                    # ori_refs = torch.stack(ori_refs, dim=0).float()  # (7, 3, H, W)
                    ref_imgs.append(refs)
                    # ori_ref_imgs.append(ori_refs)

            refs = torch.cat([x for x in ref_imgs], dim=1)  # (7, 3*num_ref_maps, H, W)
            # ori_refs = torch.cat(
                # [x for x in ori_ref_imgs], dim=1
            # )  # (7, 3*num_ref_maps, H, W)

            # save target reference maps for joint prediction with rgb images
            # data["target_ref_imgs"] = ori_refs # (6, 3*num_ref_maps, H, W)


            # grid distortion
            # if (
            #     self.prob_grid_distortion is not None
            #     and random.random() < self.prob_grid_distortion
            # ):
            #     # strength = 0.5 + random.random() * 0.5  # [0.5, 1.0]
            #     strength = 0.5
            #     refs = grid_distortion(refs, strength)

            # downsample
            # random
            # downsample_offset = 0
            # if (
            #     self.prob_rand_downsample_size is not None
            #     and random.random() < self.prob_rand_downsample_size
            # ):
            #     downsample_offset = np.random.randint(
            #         -self.rand_downsample_offset, self.rand_downsample_offset
            #     )

            # if 'rgba' in self.ref_maps:
            #     # downsample rgba
            #     downsampled_ref_rgba = F.interpolate(
            #         refs[:,-3:],
            #         size=(
            #             self.downsample_size-12 + downsample_offset,
            #             self.downsample_size-12 + downsample_offset,
            #         ),
            #         mode="nearest",
            #     )
            #     # upsample rgba
            #     upsampled_ref_rgba = F.interpolate(
            #         downsampled_ref_rgba,
            #         size=(
            #             self.upsample_size,
            #             self.upsample_size,
            #         ),
            #         mode="nearest",
            #     )  # [V, C, H, W]
            #     # downsample other
            #     downsampled_ref_other = F.interpolate(
            #         refs[:,:-3],
            #         size=(
            #             self.downsample_size + downsample_offset,
            #             self.downsample_size + downsample_offset,
            #         ),
            #         mode="nearest",
            #     )
            #     # upsample other
            #     upsampled_ref_other = F.interpolate(
            #         downsampled_ref_other,
            #         size=(
            #             self.upsample_size,
            #             self.upsample_size,
            #         ),
            #         mode="nearest",
            #     )  # [V, C, H, W]
            #     upsampled_ref=torch.cat([upsampled_ref_other, upsampled_ref_rgba], dim=1)
            # else:
            #     downsampled_ref = F.interpolate(
            #         refs,
            #         size=(
            #             self.downsample_size + downsample_offset,
            #             self.downsample_size + downsample_offset,
            #         ),
            #         mode="nearest",
            #     )  # [V, C, H, W]
            #     # upsample
            #     upsampled_ref = F.interpolate(
            #         downsampled_ref,
            #         size=(
            #             self.upsample_size,
            #             self.upsample_size,
            #         ),
            #         mode="nearest",
            #     )  # [V, C, H, W]

            # assert self.downsample_size is None
            # resized_refs = F.interpolate(
            #         refs,
            #         size=(
            #             self.upsample_size,self.upsample_size
            #         ),
            #         mode="bilinear",
            # )  # [V, C, H, W]
                
            # refs = resized_refs.clamp(0, 1)

            data["ref_imgs"] = refs  # (6, 3*num_ref_maps, H, W)

            # TODO: current only predict refmaps when flip is not applied, because we didn't render flipped refmaps
            # if self.prob_predict_ref_map is not None:
            #     data["out_type_mask"] = torch.ones_like(
            #         data["ref_imgs"][:, :3]
            #     )  # 1 is rgb, (6, 3, H, W)
            #     if not flip_flag:
            #         # random predict rgb or refmaps for each view
            #         for i in range(len(data["out_type_mask"])):
            #             if random.random() < self.prob_predict_ref_map:
            #                 if "ccm" in self.ref_maps and "normal" in self.ref_maps:
            #                     if random.random() < 0.5:
            #                         data["out_type_mask"][i] = 0  # 0 is ccm, (3, H, W)
            #                     else:
            #                         data["out_type_mask"][
            #                             i
            #                         ] = 0.5  # 0.5 is normal, (3, H, W)
            #                 else:
            #                     data["out_type_mask"][
            #                         i
            #                     ] = 0  # 0 is refmap, either ccm or normal, (3, H, W)

        return data
