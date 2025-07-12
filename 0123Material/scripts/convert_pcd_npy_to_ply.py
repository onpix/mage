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

import numpy as np

from utils.point_cloud import save_pcd


def make_array(data: np.ndarray) -> np.ndarray:
    if data.dtype == np.dtype("O"):
        data = np.concatenate([data.item()['xyz'], data.item()['rgb']], axis=1)
    return data


if __name__ == '__main__':
    import sys
    npy_file = sys.argv[1]
    ply_file = sys.argv[2]
    point_cloud_data = np.load(npy_file, allow_pickle=True)
    point_cloud_data = make_array(point_cloud_data)
    save_pcd(ply_file, point_cloud_data)