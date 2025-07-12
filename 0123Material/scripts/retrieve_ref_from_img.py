# Copyright (c) 2023-2024, Zexin He
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import sys
sys.path.append('.')

from PIL import Image
import numpy as np
import torch

from utils.retrieval import RetrievalImageToPointCloud


retrieval = RetrievalImageToPointCloud(device="cuda")


def retrieve_image_to_pcd(image_path: str, n_candidates: int = 5):
    img = torch.from_numpy(np.array(Image.open(image_path))).cuda().permute(2,0,1).unsqueeze(0) / 255.0
    if img.shape[1] == 4:
        img = img[:, :3, ...] * img[:, 3:, ...] + (1 - img[:, 3:, ...])
    img = torch.nn.functional.interpolate(img, size=(224,224), mode='bicubic', align_corners=True).clamp(0,1)
    global retrieval
    outs = retrieval.retrieve(img, n_candidates=n_candidates)
    return outs


def demo():
    while True:
        image_path = input("Input the image path: ")
        print(retrieve_image_to_pcd(image_path, n_candidates=3))


if __name__ == '__main__':
    # image_path = "/mnt/petrelfs/hezexin/remote/LRM/data/inference_images/release_inputs/lrm-inputs/redlight.png"
    demo()