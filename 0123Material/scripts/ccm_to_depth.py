import torch
import torchvision.transforms.functional as TF
from PIL import Image
import json
import numpy as np
from torchvision.transforms import v2

def ccm_to_depth(ccm, c2w, znear, zfar):
    # Invert the c2w matrix to get w2c (world to camera coordinates)
    w2c = torch.inverse(c2w)

    # Convert ccm to homogeneous coordinates
    ones = torch.ones((1, 512, 512), device=ccm.device)  # Add 1s for the homogeneous coordinate
    ccm_homogeneous = torch.cat([ccm, ones], dim=0)  # Shape [4, 512, 512]

    # Reshape ccm to [4, N] where N is the number of pixels
    ccm_homogeneous = ccm_homogeneous.view(4, -1)  # Shape [4, 512*512]

    # Transform the points from world to camera coordinates
    ccm_camera = torch.matmul(w2c, ccm_homogeneous)  # Shape [4, N]

    # Reshape back to [4, 512, 512]
    ccm_camera = ccm_camera.view(4, 512, 512)

    # Extract the Z component (depth) from camera coordinates
    depth_camera = -ccm_camera[2:3, :, :]  # Z component, shape [1, 512, 512]

    # Normalize the depth values to the range [0, 1] using znear and zfar
    depth_normalized = (depth_camera - znear) / (zfar - znear)
    
    # Print min and max of depth_camera and depth_normalized for debugging
    # print(depth_camera.min().item(), depth_camera.max().item())
    # print(depth_normalized.min().item(), depth_normalized.max().item())
    
    # Ensure the depth values are within the range [0, 1]
    depth_normalized = torch.clamp(depth_normalized, 0, 1)
    
    return depth_normalized



def depth_to_ccm(depth, c2w, znear, zfar, image_size=(512, 512)):
    """
    Converts a batch of depth maps back into a batch of Canonical Coordinate Maps (CCMs) 
    using the intrinsic matrix K.
    
    Args:
    - depth (torch.Tensor): Normalized depth maps of shape [B, H, W] in the range [0, 1].
    - c2w (torch.Tensor): Camera-to-world transformation matrix of shape [B, 4, 4].
    - K (torch.Tensor): Intrinsic matrix of shape [3, 3] defining camera parameters.
    - znear (float): Near clipping plane.
    - zfar (float): Far clipping plane.
    - image_size (tuple): The size of the depth image (H, W). Defaults to (512, 512).
    
    Returns:
    - ccm (torch.Tensor): Canonical Coordinate Maps (XYZ positions) of shape [B, 3, H, W].
    """

    # Example parameters
    camera_angle_x = 0.6911112070083618  # radians

    B, H, W = depth.shape  # Get batch size, height, and width

    K = get_intrinsic_matrix(camera_angle_x, camera_angle_x, W, H)


    # Unnormalize the depth values to get actual depth in camera space for each batch
    depth_camera = (depth * (zfar - znear) + znear)  # Shape [B, H, W]

    # Create a meshgrid for pixel coordinates (in pixel space)
    i = torch.arange(0, W, device=depth.device).float()  # X pixel coordinates
    j = torch.arange(0, H, device=depth.device).float()  # Y pixel coordinates
    ii, jj = torch.meshgrid(i, j, indexing='ij')  # Shape [W, H]
    ii = W - 1 - ii
    # jj = H -1 - jj
    # import pdb; pdb.set_trace()

    # Normalize pixel coordinates using the intrinsic matrix K
    fx = K[0, 0]  # Focal length in x
    fy = K[1, 1]  # Focal length in y
    cx = K[0, 2]  # Principal point in x
    cy = K[1, 2]  # Principal point in y
    # Compute the camera coordinates (x, y, z) from pixel coordinates and depth
    x_camera = (jj - cx) * depth_camera / fx  # Shape [B, H, W]
    y_camera = (ii - cy) * depth_camera / fy  # Shape [B, H, W]
    z_camera = depth_camera  # Negative depth because camera looks in negative Z direction

    # Stack to form [B, 4, H, W], where the 4 channels represent (x, y, z, 1) in camera space
    ones = torch.ones_like(z_camera, device=depth.device)  # Shape [B, H, W]
    ccm_camera = torch.stack([x_camera, y_camera, -z_camera, ones], dim=1)  # Shape [B, 4, H, W]

    # Reshape for matrix multiplication (flatten H*W per batch and transpose)
    ccm_camera = ccm_camera.view(B, 4, -1) # Shape [B, H*W, 4]

    # Apply the camera-to-world transformation for each batch
    ccm_world = torch.bmm(c2w.repeat(B, 1, 1), ccm_camera)  # Shape [B, 4, H*W]

    # Reshape back to [B, 4, H, W] and drop the homogeneous coordinate
    ccm_world = ccm_world.view(B, 4, H, W)

    # Extract the first three components (X, Y, Z) as the CCM in world coordinates
    ccm = ccm_world[:, :3, :, :]  # Shape [B, 3, H, W]

    # xyz -> x z -y
    # ccm = ccm[:,[0,2,1]] # x z y
    # ccm[:, 2] = - ccm[:, 2] # x z -y
    # print(ccm.mean())
    # print(ccm.max())
    # print(ccm.min())


    # ccm = (ccm + 1.0)/2.0 # [-1,1] to [0,1]
    # ccm = torch.clamp(ccm, 0, 1)
    ccm = torch.clamp(ccm, -1, 1)
    
    return ccm


def get_intrinsic_matrix(fov_x, fov_y, image_width, image_height):
    """
    Calculate the camera intrinsic matrix K
    Args:
    - fov_x (float): Horizontal field of view (unit: radians)
    - fov_y (float): Vertical field of view (unit: radians)
    - image_width (int): Image width (unit: pixels)
    - image_height (int): Image height (unit: pixels)

    Returns:
    - K (np.array): Intrinsic matrix 3x3
    """
    # Calculate focal lengths fx and fy
    fx = image_width / (2 * np.tan(fov_x / 2))
    fy = image_height / (2 * np.tan(fov_y / 2))

    # Set the optical center at the image center
    cx = image_width / 2
    cy = image_height / 2

    # Construct the intrinsic matrix K
    K = np.array([[fx, 0,  cx],
                  [0,  fy, cy],
                  [0,  0,  1]])
    
    return K



# Example usage
if __name__ == "__main__":
    for vid in range(0,7):
        # Create a sample CCM (replace with your actual CCM loading code)
        # ccm = Image.open(f'/3d/WonderMaterial/files/render-v8/0a3da81bfa904cafa446ab97e7a62cad/ccm_000_{vid:1d}0001.png')  # Random CCM for demonstration
        ccm = Image.open(f'/3d/WonderMaterial/files/render-v8/9438abf986c7453a9f4df7c34aa2e65b/ccm_000_{vid:1d}0001.png')  # Random CCM for demonstration
        ccm = np.array(ccm)
        ccm = torch.from_numpy(ccm).float()/255.0
        ccm = ccm.permute(2, 0, 1).contiguous()
        mask = ccm[3:4].clone()
        ccm = ccm*2 -1

        ccm = ccm[:3, :, :]
        tmp = ccm[1].clone()
        ccm[1]=-ccm[2]
        ccm[2]=tmp
        ccm = ccm * mask + 1.0 * (1 - mask)

        # Create a sample camera-to-world matrix (replace with your actual matrix)
        c2w_dict=json.load(open('/3d/0123Material/scripts/poses.json'))
        camera_to_world_matrix = torch.tensor(c2w_dict[str(vid)])
        normalized_depth_map = torch.tensor(ccm_to_depth(ccm, camera_to_world_matrix, 2.6-1.414, 2.6+1.414))

        # white bg
        # normalized_depth_map = normalized_depth_map * mask + 1.0 * (1 - mask)

        # black bg
        normalized_depth_map = (1 - normalized_depth_map) * mask + 0.0 * (1 - mask)
        normalized_depth_map = 1-normalized_depth_map



        print(normalized_depth_map.shape)
        print(normalized_depth_map.min())
        print(normalized_depth_map.max())
        image = normalized_depth_map.repeat(3,1,1).permute(1,2,0).contiguous().numpy()
        # # Set near and far planes
        # near = 2.6-1.414
        # far = 2.6+1.414

        # # Convert CCM to normalized depth map
        # normalized_depth_map = ccm_to_normalized_depth(ccm, camera_to_world_matrix, near, far)

        # # Move tensor to CPU and convert to numpy for visualization
        # normalized_depth_map=normalized_depth_map.squeeze(0).repeat(3,1,1).permute(1,2,0).contiguous().cpu().numpy()
        print(image.shape)
        # save to pil image
        image = Image.fromarray((image * 255).astype(np.uint8))
        image.save(f'tmp/normalized_depth_map_{vid}.png')


        # depth to ccm
        # normalized_depth_map = (1 - normalized_depth_map) * mask + 1.0 * (1 - mask)
        # normalized_depth_map = v2.functional.resize(
        #     normalized_depth_map, (320), interpolation=3, antialias=True
        # ).clamp(0, 1)
        # mask = v2.functional.resize(
        #     mask, (320), interpolation=3, antialias=True
        # ).clamp(0, 1)
        ccm_recovered = depth_to_ccm(normalized_depth_map, camera_to_world_matrix, 2.6-1.414, 2.6+1.414)
        print(ccm.shape)
        ccm_recovered = ccm_recovered[0]
        ccm_recovered = ccm_recovered * mask + 1.0 * (1 - mask)
        
        image = ccm_recovered.permute(1,2,0).numpy()
        image = Image.fromarray((image * 255).astype(np.uint8))
        image.save(f'tmp/recovered_ccm_{vid}.png')
