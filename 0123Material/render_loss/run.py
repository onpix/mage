import argparse
import os
import torch
import json
import numpy as np
from torch import nn
from PIL import Image
import fire
from pathlib import Path
import torchvision as tv
import nvdiffrast.torch as dr

try:
    from . import light
except:
    import light

glctx = dr.RasterizeCudaContext()


def get_transformation_matrix(mode):
    axes = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}
    new_axes = mode.lower().split()
    T_list = [[0, 0, 0, 0] for _ in range(4)]

    for i, axis in enumerate(new_axes):
        sign = 1

        if axis.startswith("-"):
            sign = -1
            axis = axis[1:]

        T_list[0][i] = sign * axes[axis][0]
        T_list[1][i] = sign * axes[axis][1]
        T_list[2][i] = sign * axes[axis][2]

    T_list[3][3] = 1

    return T_list


def load_image(path):
    img = Image.open(path)
    # resize to 320x320
    # img = img.resize((320, 320), Image.ANTIALIAS)
    return torch.from_numpy(np.array(img) / 255.0).float().cuda()


def normalize_vectors(vectors):
    return vectors / torch.norm(vectors, dim=-1, keepdim=True)


class RenderLoss(nn.Module):
    def __init__(self, env_root=None, device='cuda'):
        super(RenderLoss, self).__init__()
        # read all lights under env_root
        if env_root:
            self.lights = {
                fname: light.load_env(os.path.join(env_root, fname))
                for fname in os.listdir(env_root)
            }
        else:
            self.lights = None

        mode = "-z -x y"
        T_list = get_transformation_matrix(mode)
        self.T = torch.tensor(T_list, device=device, dtype=torch.float32)

    def render(self, light_name, data, mask, c2w):
        """
        Pytorch loss version of function main. Passing tensors instead of paths.
        data: prediction of the model

        Note: c2w should be rendered using y-up in blender!! (ie, the up direction of the object is world y)
        """
        for key in data.keys():
            data[key] = data[key] * mask

            # reshape to [1, h, w, 3]
            if (len(data[key].shape) == 3):
                _, h, w = data[key].shape
                data[key] = data[key].permute(1, 2, 0)[None, ...]
            else:
                raise NotImplementedError
                # _, h, w, _ = data[key].shape

        if self.lights:
            _light = self.lights[light_name]
        else:
            _light = light.load_env(light_name)

        xyz = data["depth"]
        albedo = data["albedo"]
        normals = data["normal"]
        roughness = data["roughness"]
        metallic = data["metallic"]

        # merge roughness and metallic into ks
        # roughness shape: [1, h, w]
        # metallic shape: [1, h, w]
        ks = torch.cat([torch.zeros_like(roughness), roughness, metallic], dim=-1)
        # import ipdb; ipdb.set_trace()

        # transform c2w to match GT
        c2w = self.T @ c2w

        # transform ccm and normals to match GT
        xyz = torch.matmul(self.T[:3, :3], xyz.reshape(-1, 3)[:, :, None]).reshape([1, h, w, 3])

        normals = normals * 2 - 1
        normals = torch.matmul(c2w[:3, :3][None, :, :], normals.reshape(-1, 3)[:, :, None]).reshape(
            [1, h, w, 3]
        )

        # normalize normals
        normals = normalize_vectors(normals)

        cam_xyz = c2w[:3, 3].view(1, 1, 1, 3)

        # import ipdb; ipdb.set_trace()
        shaded = _light.shade(xyz, normals, albedo, ks, cam_xyz)

        # Apply mask to shaded result
        shaded = shaded * mask.view(1, h, w, 1)

        # gamma
        # NOTE: very important!!! to add 1e-8
        shaded = torch.pow(shaded+1e-8, 1.0 / 2.2)

        return shaded


def main(env_path, root_path, mode="x y z", idx=1):
    # Load environment map
    env_light = light.load_env(env_path)

    # Load input images
    suffix = f"_000_{idx}"
    xyz = load_image(os.path.join(root_path, f"ccm_000_{idx}0001.png"))
    normals = load_image(os.path.join(root_path, f"normals{suffix}.png"))

    albedo = load_image(os.path.join(root_path, f"albedo{suffix}.png"))
    ks = load_image(os.path.join(root_path, f"ks{suffix}.png"))
    mask = load_image(os.path.join(root_path, f"mask{suffix}.png"))

    # Create binary mask
    binary_mask = (mask[:, :] >= 0.5).float().unsqueeze(-1)

    # Apply mask and set background to black
    xyz = xyz * binary_mask
    normals = normals * binary_mask
    albedo = albedo * binary_mask
    ks = ks * binary_mask

    xyz = xyz[:, :, :3]
    normals = normals[:, :, :3]
    albedo = albedo[:, :, :3]
    ks = ks[:, :, :3]

    # Reshape inputs to match expected dimensions
    #
    # save all images as tensor grid
    mask = mask.unsqueeze(-1).repeat(1, 1, 3)

    h, w, _ = xyz.shape
    # CCM is in [0, 1], convert it to [-1, 1]
    xyz = xyz.view(1, h, w, 3) * 2 - 1

    albedo = albedo.view(1, h, w, 3)
    ks = ks.view(1, h, w, 3)

    # Set fixed view position

    # view_pos = torch.tensor([0.0, 0.0, 5.0], dtype=torch.float32, device="cuda").view(1, 1, 1, 3)
    transform_json_path = os.path.join(root_path, "transforms.json")
    matrix = json.load(open(transform_json_path))["frames"][idx]["transform_matrix"]
    matrix = torch.tensor(matrix, dtype=torch.float32, device="cuda")

    # y = xyz[..., 1].clone()
    # z = xyz[..., 2].clone()
    # xyz[..., 1] = -z
    # xyz[..., 2] = y

    # switch yz
    T_list = get_transformation_matrix(mode)
    T = torch.tensor(T_list, device="cuda", dtype=torch.float32)
    matrix = T @ matrix

    # convert xyz to the new coordinate system
    xyz = torch.matmul(T[:3, :3], xyz.reshape(-1, 3)[:, :, None]).reshape([1, h, w, 3])
    normals = normals * 2 - 1
    normals = torch.matmul(matrix[:3, :3][None, :, :], normals.reshape(-1, 3)[:, :, None]).reshape(
        [1, h, w, 3]
    )

    # Normalize normals
    normals = normalize_vectors(normals)

    # t vec in Rt matrix
    view_pos = matrix[:3, 3].view(1, 1, 1, 3)

    # make it like mirror
    # ks = torch.zeros_like(ks)
    # ks[..., -1] = 1

    # import ipdb; ipdb.set_trace()
    # img_list = [x[:, :, :3].permute(-1, 0, 1) for x in [xyz, normals, albedo, ks, mask]]
    # tv.utils.save_image(list(img_list), os.path.join(root_path, "input_images.png"))

    # Perform shading
    shaded = env_light.shade(xyz, normals, albedo, ks, view_pos)

    # Apply mask to shaded result
    shaded = shaded * binary_mask.view(1, h, w, 1)

    # gamma correction
    shaded = torch.pow(shaded, 1.0 / 2.2)

    # img_list = [(xyz + 1) / 2, (normals + 1) / 2, albedo, ks, shaded]
    # tv.utils.save_image([x[0].permute(-1, 0, 1) for x in img_list], f'tmp_{mode}.png')

    tv.utils.save_image(
        shaded[0].permute(-1, 0, 1), os.path.join(root_path, f"shaded_result_{mode}_view_{idx}.png")
    )
    return shaded

    # # Convert shaded result to image
    # result = (shaded.squeeze().detach().cpu().numpy() * 255).astype(np.uint8)
    # # Save result
    # Image.fromarray(result).save(os.path.join(root_path, "shaded_result.png"))


def debug_torch(path):
    # path is a torch.save file's path, a dict
    render = RenderLoss()
    data = torch.load(path)
    # import ipdb; ipdb.set_trace()

    # idx = 1
    # root_path = '../files/bear/bear_z_up_realCCM'
    # suffix = f"_000_{idx}"
    # c2w = json.load(open(os.path.join(root_path, "transforms.json")))["frames"][idx]["transform_matrix"]
    # c2w = torch.tensor(c2w).cuda()

    c2w = data['c2w']
    mask = data['mask']

    res = render.render('../files/hdr/railway_bridges_4k.hdr', data['data'], mask, c2w)
    tv.utils.save_image(res[0].permute(-1, 0, 1), 'tmp.png')

def run(env, root):
    axes = ["x", "y", "z"]
    signs = ["", "-"]

    all_modes = [
        f"{s1}{a1} {s2}{a2} {s3}{a3}"
        for a1 in axes
        for a2 in axes
        for a3 in axes
        for s1 in signs
        for s2 in signs
        for s3 in signs
        if len({a1, a2, a3}) == 3
    ]
    all_modes = [
        # 'z x y', 'z -x y', 'x z y', 'x -z y', '-z x y', '-z -x y', '-x z y', '-x -z y'
        "-z -x y"
    ]

    for mode in all_modes:
        for idx in range(1, 7):
            main(env, root, mode, idx)
    # main(env, root, "x y z")
    os._exit(0)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Run environment map rendering")
    # parser.add_argument("env", type=str, help="Path to the HDR environment map")

    # parser.add_argument("root", type=str, help="Root folder containing input images")

    # args = parser.parse_args()
    fire.Fire()

        
    # render_loss = RenderLoss()
    # idx = 1
    # root_path = '../files/bear/bear_z_up_realCCM'
    # suffix = f"_000_{idx}"
    # xyz = load_image(os.path.join(root_path, f"ccm_000_{idx}0001.png"))
    # xyz = xyz * 2 - 1
    # normals = load_image(os.path.join(root_path, f"normals{suffix}.png"))
    # albedo = load_image(os.path.join(root_path, f"albedo{suffix}.png"))
    # ks = load_image(os.path.join(root_path, f"ks{suffix}.png"))
    # mask = load_image(os.path.join(root_path, f"mask{suffix}.png"))[None, ..., None]
    # data = {
    #     "depth": xyz[None, :, :, :3],
    #     "normal": normals[None, :, :, :3],
    #     "albedo": albedo[None, :, :, :3],
    #     "roughness": ks[None, :, :, 0:1],
    #     "metallic": ks[None, :, :, 1:2]
    # }
    # c2w = json.load(open(os.path.join(root_path, "transforms.json")))["frames"][idx]["transform_matrix"]
    # res = render_loss.render('../files/hdr/railway_bridges_4k.hdr', data, mask, torch.tensor(c2w).cuda())
    # tv.utils.save_image(res[0].permute(-1, 0, 1), 'tmp.png')
