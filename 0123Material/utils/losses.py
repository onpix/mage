import torch
import numpy as np
import torch.nn.functional as F

def angular_loss(norm_out, gt_norm, gt_norm_mask=None): 
            """ norm_out:       (B, 3, ...)
                gt_norm:        (B, 3, ...)
                gt_norm_mask:   (B, 1, ...)   

            """
            pred_norm = norm_out[:, 0:3, ...]
            dot = torch.cosine_similarity(pred_norm, gt_norm, dim=1).unsqueeze(1)    
            if gt_norm_mask is not None:
                valid_mask = torch.logical_and(gt_norm_mask, torch.abs(dot.detach()) < 1-1e-7)
                angle = torch.acos(dot[valid_mask])
            else:
                angle = torch.acos(dot[torch.abs(dot.detach()) < 1-1e-7])
            return torch.mean(angle)

def align_depth_least_square(
    gt: np.ndarray,
    pred: np.ndarray,
    valid_mask: np.ndarray,
    pred_torch: torch.tensor,
    return_scale_shift=True,
):
    ori_shape = pred.shape  # input shape

    assert (
        gt.shape == pred.shape == valid_mask.shape
    ), f"{gt.shape}, {pred.shape}, {valid_mask.shape}"

    batch_size = gt.shape[0]
    aligned_preds = []
    scales = []
    shifts = []
    for i in range(batch_size):
        gt_masked = gt[i][valid_mask[i]].reshape((-1, 1))
        pred_masked = pred[i][valid_mask[i]].reshape((-1, 1))
        # numpy solver
        _ones = np.ones_like(pred_masked)
        A = np.concatenate([pred_masked, _ones], axis=-1)
        X = np.linalg.lstsq(A, gt_masked, rcond=None)[0]
        scale, shift = X

        aligned_pred = pred_torch[i] * scale.item() + shift.item()

        # restore dimensions
        aligned_pred = aligned_pred.view(ori_shape[1:])
        aligned_preds.append(aligned_pred)
        scales.append(torch.from_numpy(scale))
        shifts.append(torch.from_numpy(shift))

    aligned_pred = torch.stack(aligned_preds, dim=0)
    scale = torch.stack(scales, dim=0)
    shift = torch.stack(shifts, dim=0)

    if return_scale_shift:
        return aligned_pred, scale, shift
    else:
        return aligned_pred



# would out of memory
def align_depth_least_square_torch(
    gt: torch.Tensor,
    pred: torch.Tensor,
    valid_mask: torch.Tensor,
    return_scale_shift=True,
):
    ori_shape = pred.shape  # input shape

    assert gt.shape == pred.shape == valid_mask.shape, f"{gt.shape}, {pred.shape}, {valid_mask.shape}"

    batch_size = gt.shape[0]
    aligned_preds = []
    scales = []
    shifts = []
    for i in range(batch_size):
        gt_masked = gt[i][valid_mask[i]].reshape(-1, 1)
        pred_masked = pred[i][valid_mask[i]].reshape(-1, 1)

        # PyTorch solver
        ones = torch.ones_like(pred_masked)
        A = torch.cat([pred_masked, ones], dim=-1)
        result = torch.linalg.lstsq(A, gt_masked, rcond=None)
        scale, shift = result.solution[:2]

        aligned_pred = pred[i] * scale + shift

        # restore dimensions
        aligned_pred = aligned_pred.view(ori_shape[1:])

        aligned_preds.append(aligned_pred)
        scales.append(scale)
        shifts.append(shift)

    aligned_pred = torch.stack(aligned_preds, dim=0)
    scale = torch.stack(scales, dim=0)
    shift = torch.stack(shifts, dim=0)
    if return_scale_shift:
        return aligned_pred, scale, shift
    else:
        return aligned_pred
