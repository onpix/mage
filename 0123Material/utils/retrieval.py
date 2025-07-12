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


import numpy as np
import torch
import open_clip

from .proxy import no_proxy


class RetrievalImageToPointCloud:

    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        self.query_encoder, self.query_transform = self._build_query_encoder()
        self.database_keys, self.database_values = self._build_database()

    def _build_query_encoder(self):
        import torchvision.transforms as transforms
        clip, _, _ = open_clip.create_model_and_transforms("EVA02-E-14-plus", pretrained="laion2b_s9b_b144k", device=self.device)
        clip.eval()
        transform = transforms.Compose([
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        return clip, transform

    @no_proxy
    def _build_database(self):
        from megfile import smart_open
        N_splits = 32
        split_path_template = "/mnt/hwfile/3dobject_aigc/hezexin/datasets/objaverse-pcd-ours-feats-norgb/{}.npz"
        splits = [np.load(smart_open(split_path_template.format(idx), 'rb')) for idx in range(N_splits)]

        data = {}
        for split in splits:
            data.update(dict(split))

        uids = list(data.keys())
        feats = torch.from_numpy(np.stack(list(data.values()), axis=0)).to(self.device)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats, uids

    def encode_query(self, query: torch.Tensor):
        # query: (B, 3, H, W), 0-1 scaled
        query = self.query_transform(query)
        feats = self.query_encoder.encode_image(query)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    @torch.no_grad()
    def retrieve(self, query: torch.Tensor, n_candidates: int = 1):
        query = self.encode_query(query)
        similarity_matrix = query @ self.database_keys.T
        topk = similarity_matrix.topk(n_candidates, dim=-1)
        topk_probs = torch.softmax(topk.values, dim=-1)
        topk_inds = topk.indices
        sampled_inds = topk_inds.gather(dim=-1, index=torch.multinomial(topk_probs, num_samples=n_candidates))[0]  # [n_candidates]

        retrive_result = []
        for sample_ind in sampled_inds.cpu():
            retrive_result.append(self.database_values[sample_ind.item()])
        return retrive_result