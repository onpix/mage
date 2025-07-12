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


def save_pcd(dump_path: str, points: np.ndarray):
    """
    Save point cloud data to file with automatic extension detection.

    Parameters:
    - dump_path: Point cloud path.
    - points: Nx6 NumPy array with XYZRGB data.
    """
    assert points.shape[1] == 6, "Input array must have 6 columns (XYZRGB)."
    ext = dump_path.split('.')[-1].lower()
    if ext == 'ply':
        xyz = points[:, :3]
        rgb = (points[:, 3:] * 255).astype(np.uint8)
        with open(dump_path, 'w') as ply_file:
            ply_file.write("ply\n")
            ply_file.write("format ascii 1.0\n")
            ply_file.write(f"element vertex {len(points)}\n")
            ply_file.write("property float x\n")
            ply_file.write("property float y\n")
            ply_file.write("property float z\n")
            ply_file.write("property uchar red\n")
            ply_file.write("property uchar green\n")
            ply_file.write("property uchar blue\n")
            ply_file.write("end_header\n")
            for i in range(len(points)):
                ply_file.write(f"{xyz[i, 0]} {xyz[i, 1]} {xyz[i, 2]} {rgb[i, 0]} {rgb[i, 1]} {rgb[i, 2]}\n")
    elif ext == 'npy':
        np.save(dump_path, points)