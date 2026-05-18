#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.image_utils import lpips_score
import json
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np
import re
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from scipy import ndimage
import faulthandler
faulthandler.enable()

def array2tensor(array, device="cuda", dtype=torch.float32):
    return torch.tensor(array, dtype=dtype, device=device)

def _extract_source_path_from_cfg(scene_dir):
    cfg_path = Path(scene_dir) / "cfg_args"
    if not cfg_path.exists():
        return None
    text = cfg_path.read_text(encoding="utf-8")
    match = re.search(r"source_path='([^']+)'", text)
    if match is None:
        return None
    source_path = match.group(1)
    source_path = os.path.expanduser(source_path)
    if not os.path.isabs(source_path):
        source_path = os.path.abspath(os.path.join(os.getcwd(), source_path))
    return source_path

def _parse_imed_intrinsics(k_path):
    with open(k_path, "r", encoding="utf-8") as f:
        raw_lines = [line.strip() for line in f if line.strip()]
    matrices = {}
    i = 0
    while i < len(raw_lines):
        line = raw_lines[i]
        if line.startswith("#"):
            header = line[1:].strip()
            if not header.startswith("K"):
                i += 1
                continue
            key = header.split()[0]
            rows = []
            for j in range(1, 4):
                vals = [float(v) for v in raw_lines[i + j].split()]
                rows.append(vals)
            matrices[key] = np.array(rows, dtype=np.float32)
            i += 4
            continue
        i += 1
    assert "K1_L" in matrices and "K2_L" in matrices, "K.txt must contain K1_L and K2_L"
    return matrices["K1_L"], matrices["K2_L"]

def _parse_imed_poses(pose_path):
    with open(pose_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    assert len(lines) == 2, f"Expected 2 rows in pose.txt, found {len(lines)}"
    c2w_by_cam = {}
    for line in lines:
        parts = line.split()
        assert len(parts) == 8, f"Invalid pose row: {line}"
        cam_id = int(parts[0])
        t = np.array([float(v) for v in parts[1:4]], dtype=np.float32)
        q = np.array([float(v) for v in parts[4:8]], dtype=np.float32)
        rot = R.from_quat(q).as_matrix().astype(np.float32)
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = rot
        c2w[:3, 3] = t
        c2w_by_cam[cam_id] = c2w
    assert 0 in c2w_by_cam and 1 in c2w_by_cam, "pose.txt must include camera ids 0 and 1"
    return c2w_by_cam[0], c2w_by_cam[1]  # cam2, cam1

def _build_global_imed_overlap_mask(source_path, out_h, out_w):
    source = Path(source_path)
    k1, k2 = _parse_imed_intrinsics(source / "K.txt")
    c2w_cam2, c2w_cam1 = _parse_imed_poses(source / "pose.txt")
    w2c_cam1 = np.linalg.inv(c2w_cam1)

    depth_files = sorted((source / "endoscope2" / "depthL").glob("*.npy"))
    assert len(depth_files) > 0, "No depth files in endoscope2/depthL"
    depth = np.load(depth_files[0]).astype(np.float32)
    h2, w2 = depth.shape

    u, v = np.meshgrid(np.arange(w2, dtype=np.float32), np.arange(h2, dtype=np.float32))
    z = depth
    valid = z > 0
    x = (u - float(k2[0, 2])) * z / float(k2[0, 0])
    y = (v - float(k2[1, 2])) * z / float(k2[1, 1])

    pts_cam2 = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    valid_flat = valid.reshape(-1)
    pts_cam2 = pts_cam2[valid_flat]
    pts_cam2_h = np.concatenate([pts_cam2, np.ones((pts_cam2.shape[0], 1), dtype=np.float32)], axis=1)
    pts_world = (c2w_cam2 @ pts_cam2_h.T).T
    pts_cam1 = (w2c_cam1 @ pts_world.T).T[:, :3]

    z1 = pts_cam1[:, 2]
    front = z1 > 1e-6
    pts_cam1 = pts_cam1[front]
    z1 = z1[front]
    u1 = float(k1[0, 0]) * (pts_cam1[:, 0] / z1) + float(k1[0, 2])
    v1 = float(k1[1, 1]) * (pts_cam1[:, 1] / z1) + float(k1[1, 2])
    u1 = np.round(u1).astype(np.int32)
    v1 = np.round(v1).astype(np.int32)

    h1 = int(round(float(k1[1, 2]) * 2))
    w1 = int(round(float(k1[0, 2]) * 2))
    if h1 <= 0 or w1 <= 0:
        h1, w1 = h2, w2
    in_bounds = (u1 >= 0) & (u1 < w1) & (v1 >= 0) & (v1 < h1)
    u1 = u1[in_bounds]
    v1 = v1[in_bounds]

    mask = np.zeros((h1, w1), dtype=np.uint8)
    mask[v1, u1] = 1
    # Convert sparse reprojections into a contiguous valid support region.
    mask = ndimage.binary_dilation(mask, structure=np.ones((3, 3), dtype=np.uint8), iterations=2)
    mask = ndimage.binary_closing(mask, structure=np.ones((11, 11), dtype=np.uint8), iterations=2)
    mask = ndimage.binary_fill_holes(mask)
    mask = mask.astype(np.float32)
    if (h1, w1) != (out_h, out_w):
        mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
        mask_t = F.interpolate(mask_t, size=(out_h, out_w), mode="nearest")
        mask = mask_t.squeeze(0).squeeze(0).numpy()
    return mask

def _has_imed_reprojection_inputs(source_path):
    if source_path is None:
        return False
    source = Path(source_path)
    return (
        (source / "K.txt").is_file()
        and (source / "pose.txt").is_file()
        and (source / "endoscope2" / "depthL").is_dir()
    )

def _create_window(window_size, channel, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    gauss = torch.exp(-(coords**2) / (2 * (1.5**2)))
    gauss = gauss / gauss.sum()
    kernel_2d = torch.outer(gauss, gauss).unsqueeze(0).unsqueeze(0)
    return kernel_2d.expand(channel, 1, window_size, window_size).contiguous()

def masked_psnr(pred, gt, mask, eps=1e-8):
    # pred/gt: [1,3,H,W], mask: [1,1,H,W]
    mask3 = mask.repeat(1, pred.shape[1], 1, 1)
    diff2 = (pred - gt) ** 2
    denom = mask3.sum().clamp_min(1.0)
    mse = (diff2 * mask3).sum() / denom
    return -10.0 * torch.log10(mse + eps)

def masked_ssim(pred, gt, mask, window_size=11, eps=1e-8):
    # SSIM map + valid-region averaging (strict masked normalization)
    channel = pred.shape[1]
    window = _create_window(window_size, channel, pred.device, pred.dtype)
    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(gt, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(gt * gt, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(pred * gt, window, padding=window_size // 2, groups=channel) - mu1_mu2
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + eps)
    mask3 = mask.repeat(1, channel, 1, 1)
    denom = mask3.sum().clamp_min(1.0)
    return (ssim_map * mask3).sum() / denom

def readImages(renders_dir, gt_dir, masks_dir, overlap_mask_path=None):
    renders = []
    gts = []
    image_names = []
    masks = []
    overlap_mask = None

    if overlap_mask_path is not None and overlap_mask_path.exists():
        overlap_raw = np.array(Image.open(overlap_mask_path))
        overlap_mask = tf.to_tensor(overlap_raw).unsqueeze(0).cuda()
        if overlap_mask.shape[1] > 1:
            overlap_mask = overlap_mask[:, :1, :, :]
        overlap_mask = (overlap_mask > 0.5).float()
    
    for fname in os.listdir(renders_dir):
        render = np.array(Image.open(renders_dir / fname))
        gt = np.array(Image.open(gt_dir / fname))
        mask = np.array(Image.open(masks_dir / fname))
        
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        frame_mask = tf.to_tensor(mask).unsqueeze(0).cuda()
        if frame_mask.shape[1] > 1:
            frame_mask = frame_mask[:, :1, :, :]
        frame_mask = (frame_mask > 0.5).float()
        if overlap_mask is not None:
            frame_mask = overlap_mask
        masks.append(frame_mask)
        image_names.append(fname)
    return renders, gts, masks, image_names


def evaluate(model_paths):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")
    
    with torch.no_grad():
        for scene_dir in model_paths:
            # try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                masks_dir = method_dir / "masks"
                overlap_mask_path = method_dir / "overlap_mask.png"
                
                renders, gts, masks, image_names = readImages(renders_dir, gt_dir, masks_dir, overlap_mask_path)
                if len(renders) > 0:
                    out_h, out_w = int(renders[0].shape[2]), int(renders[0].shape[3])
                    source_path = _extract_source_path_from_cfg(scene_dir)
                    if _has_imed_reprojection_inputs(source_path):
                        global_mask = _build_global_imed_overlap_mask(source_path, out_h, out_w)
                        global_mask_t = torch.from_numpy(global_mask).unsqueeze(0).unsqueeze(0).to(device="cuda", dtype=torch.float32)
                        for i in range(len(masks)):
                            masks[i] = global_mask_t
                        Image.fromarray((global_mask * 255.0).astype(np.uint8)).save(overlap_mask_path)
                        Image.fromarray((global_mask * 255.0).astype(np.uint8)).save(method_dir / "imed_reprojection_mask.png")

                ssims = []
                psnrs = []
                lpipss = []
                                
                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    render, gt, mask = renders[idx], gts[idx], masks[idx]
                    psnrs.append(masked_psnr(render, gt, mask))
                    ssims.append(masked_ssim(render, gt, mask))
                    # Keep LPIPS consistent with previous behavior by masking inputs.
                    lpipss.append(lpips_score(render * mask, gt * mask))

                print("Scene: ", scene_dir,  "SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("Scene: ", scene_dir,  "PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("Scene: ", scene_dir,  "LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("")

                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                                        "LPIPS": torch.tensor(lpipss).mean().item()})
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        # except:
        #     print("Unable to compute metrics for model", scene_dir)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)
