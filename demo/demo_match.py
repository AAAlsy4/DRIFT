"""
RoMa demo for UAV--satellite image registration.

Input:
    1) UAV / drone image
    2) Satellite image

Output directory:
    matches_all.png          RoMa sampled correspondences before RANSAC
    matches_inliers.png      RANSAC inlier correspondences
    warped_uav_to_sat.png    UAV image warped into the satellite image frame
    warped_overlay.png       Overlay of warped UAV and satellite image
    registration_result.png  Paper-friendly summary figure
    homography.txt           Estimated UAV -> satellite homography
    statistics.txt           Match and RANSAC statistics

Example:
    python demo/demo_match.py --uav_path uav.jpg --sat_path sat.jpg
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Optional, Tuple

import re
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import cv2

from romatch import roma_outdoor

import warnings
warnings.filterwarnings("ignore", message="Local correlation is not supported on non-Linux platforms")


PointArray = np.ndarray


def get_device(device_arg: str) -> torch.device:
    """Select running device."""
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda:1")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_rgb(path: str) -> Image.Image:
    """Load image as RGB PIL image."""
    return Image.open(path).convert("RGB")


def pil_resize_keep_height(img: Image.Image, target_h: int) -> Image.Image:
    """Resize image to a fixed height while keeping aspect ratio."""
    resample = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
    w, h = img.size
    new_w = max(1, int(round(w * target_h / h)))
    return img.resize((new_w, target_h), resample)


def to_numpy_points(x) -> np.ndarray:
    """Convert torch/list points to float32 numpy array with shape [N, 2]."""
    if isinstance(x, torch.Tensor):
        x = x.detach().float().cpu().numpy()
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(-1, 2)
    return x


def sample_roma_matches(
    roma_model,
    warp: torch.Tensor,
    certainty: torch.Tensor,
    uav_size: Tuple[int, int],
    sat_size: Tuple[int, int],
    num_matches: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample RoMa dense correspondences and convert them to original image pixels.

    Returns:
        kpts_uav: [N, 2], points in original UAV image coordinates.
        kpts_sat: [N, 2], corresponding points in original satellite coordinates.
        conf    : [N], confidence values.
    """
    uav_w, uav_h = uav_size
    sat_w, sat_h = sat_size

    with torch.no_grad():
        try:
            matches, conf = roma_model.sample(warp, certainty, num=num_matches)
            # # 不使用TopK
            # matches, conf = roma_model.sample(warp, certainty)
        except TypeError:
            # Some RoMa versions do not expose the num argument.
            matches, conf = roma_model.sample(warp, certainty)

        kpts_uav, kpts_sat = roma_model.to_pixel_coordinates(
            matches, uav_h, uav_w, sat_h, sat_w
        )

    kpts_uav = to_numpy_points(kpts_uav)
    kpts_sat = to_numpy_points(kpts_sat)

    if isinstance(conf, torch.Tensor):
        conf = conf.detach().float().cpu().numpy()
    conf = np.asarray(conf, dtype=np.float32).reshape(-1)

    # Be defensive in case a RoMa version returns confidence with a different length.
    n = min(len(kpts_uav), len(kpts_sat), len(conf))
    kpts_uav = kpts_uav[:n]
    kpts_sat = kpts_sat[:n]
    conf = conf[:n]

    if n > num_matches:
        order = np.argsort(-conf)[:num_matches]
        kpts_uav = kpts_uav[order]
        kpts_sat = kpts_sat[order]
        conf = conf[order]

    return kpts_uav, kpts_sat, conf


def filter_matches_by_confidence(
    kpts_uav: np.ndarray,
    kpts_sat: np.ndarray,
    conf: np.ndarray,
    conf_thr: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep correspondences whose confidence is above the threshold."""
    if len(conf) == 0:
        return kpts_uav, kpts_sat, conf
    keep = conf >= conf_thr
    return kpts_uav[keep], kpts_sat[keep], conf[keep]


def estimate_homography(
    kpts_uav: np.ndarray,
    kpts_sat: np.ndarray,
    ransac_thr: float,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Estimate UAV -> satellite homography using RANSAC."""
    if cv2 is None:
        print("[Warn] OpenCV is not installed. RANSAC and warping are skipped.")
        return None, None
    if len(kpts_uav) < 4:
        print("[Warn] Fewer than 4 matches after filtering. Homography cannot be estimated.")
        return None, None

    H, mask = cv2.findHomography(
        kpts_uav.astype(np.float32),
        kpts_sat.astype(np.float32),
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thr,
    )
    if H is None or mask is None:
        print("[Warn] cv2.findHomography failed.")
        return None, None

    return H.astype(np.float64), mask.reshape(-1).astype(bool)

    # # 不使用 RANSAC：普通最小二乘估计单应矩阵
    # H, _ = cv2.findHomography(
    #     kpts_uav.astype(np.float32),
    #     kpts_sat.astype(np.float32),
    #     method=0,
    # )
    # if H is None:
    #     print("[Warn] cv2.findHomography without RANSAC failed.")
    #     return None, None
    #
    # # 为了表格统计，仍然用重投影误差判断“几何内点”
    # pts_uav_h = np.concatenate(
    #     [kpts_uav, np.ones((len(kpts_uav), 1), dtype=np.float32)],
    #     axis=1,
    # )
    # proj = (H @ pts_uav_h.T).T
    # proj = proj[:, :2] / np.maximum(proj[:, 2:3], 1e-8)
    #
    # errors = np.linalg.norm(proj - kpts_sat, axis=1)
    # mask = errors <= ransac_thr
    #
    # return H.astype(np.float64), mask.astype(bool)


def select_match_indices(
    conf: np.ndarray,
    inlier_mask: Optional[np.ndarray],
    max_vis_matches: int,
) -> np.ndarray:
    """Select a compact subset of matches for visualization."""
    n = len(conf)
    indices = np.arange(n)
    if inlier_mask is not None:
        indices = indices[inlier_mask]

    if len(indices) == 0:
        return indices

    if max_vis_matches > 0 and len(indices) > max_vis_matches:
        local_conf = conf[indices]
        order = np.argsort(-local_conf)[:max_vis_matches]
        indices = indices[order]

    return indices


def draw_matches(
    uav_img: Image.Image,
    sat_img: Image.Image,
    kpts_uav: np.ndarray,
    kpts_sat: np.ndarray,
    conf: np.ndarray,
    save_path: str,
    title: str,
    inlier_mask: Optional[np.ndarray] = None,
    max_vis_matches: int = 300,
    vis_height: int = 520,
) -> None:
    """Draw side-by-side matching visualization."""
    margin = 18
    title_h = 42
    footer_h = 40
    gap = 24

    uav_vis = pil_resize_keep_height(uav_img, vis_height)
    sat_vis = pil_resize_keep_height(sat_img, vis_height)

    sx_uav = uav_vis.width / uav_img.width
    sy_uav = uav_vis.height / uav_img.height
    sx_sat = sat_vis.width / sat_img.width
    sy_sat = sat_vis.height / sat_img.height

    canvas_w = margin * 2 + uav_vis.width + gap + sat_vis.width
    canvas_h = margin * 2 + title_h + vis_height + footer_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    left_x = margin
    right_x = margin + uav_vis.width + gap
    top_y = margin + title_h

    canvas.paste(uav_vis, (left_x, top_y))
    canvas.paste(sat_vis, (right_x, top_y))

    draw.text((margin, margin), title, fill=(0, 0, 0), font=font)
    draw.text((left_x, margin + 20), "UAV image", fill=(0, 0, 0), font=font)
    draw.text((right_x, margin + 20), "Satellite image", fill=(0, 0, 0), font=font)

    indices = select_match_indices(conf, inlier_mask, max_vis_matches)

    for idx in indices:
        x1, y1 = kpts_uav[idx]
        x2, y2 = kpts_sat[idx]
        p1 = (left_x + x1 * sx_uav, top_y + y1 * sy_uav)
        p2 = (right_x + x2 * sx_sat, top_y + y2 * sy_sat)
        draw.line((*p1, *p2), fill=(255, 0, 0), width=1)
        r = 2
        draw.ellipse((p1[0] - r, p1[1] - r, p1[0] + r, p1[1] + r), fill=(255, 0, 0))
        draw.ellipse((p2[0] - r, p2[1] - r, p2[0] + r, p2[1] + r), fill=(0, 128, 255))

    if inlier_mask is None:
        count_text = f"visualized {len(indices)} / {len(kpts_uav)} confidence-filtered RoMa matches"
    else:
        count_text = f"visualized {len(indices)} / {int(inlier_mask.sum())} RANSAC inliers"

    draw.text((margin, top_y + vis_height + 12), count_text, fill=(0, 0, 0), font=font)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(save_path)


def warp_uav_to_satellite(
    uav_img: Image.Image,
    sat_img: Image.Image,
    H_uav_to_sat: np.ndarray,
    alpha: float,
) -> Tuple[Image.Image, Image.Image]:
    """Warp UAV image into satellite image coordinates and create an overlay."""
    if cv2 is None:
        raise RuntimeError("OpenCV is required for warpPerspective.")

    uav_np = np.asarray(uav_img)
    sat_np = np.asarray(sat_img)
    sat_w, sat_h = sat_img.size

    warped = cv2.warpPerspective(uav_np, H_uav_to_sat, (sat_w, sat_h))
    valid_mask = cv2.warpPerspective(
        np.ones((uav_img.height, uav_img.width), dtype=np.uint8) * 255,
        H_uav_to_sat,
        (sat_w, sat_h),
    ) > 0

    overlay = sat_np.astype(np.float32).copy()
    overlay[valid_mask] = (
        alpha * warped[valid_mask].astype(np.float32)
        + (1.0 - alpha) * sat_np[valid_mask].astype(np.float32)
    )
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return Image.fromarray(warped), Image.fromarray(overlay)


def add_title(img: Image.Image, title: str) -> Image.Image:
    """Add a simple title bar to an image."""
    margin = 16
    title_h = 38
    font = ImageFont.load_default()
    canvas = Image.new("RGB", (img.width, img.height + title_h + margin), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, margin), title, fill=(0, 0, 0), font=font)
    canvas.paste(img, (0, title_h + margin))
    return canvas


def resize_to_width(img: Image.Image, width: int) -> Image.Image:
    """Resize image to a fixed width while keeping aspect ratio."""
    resample = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
    h = max(1, int(round(img.height * width / img.width)))
    return img.resize((width, h), resample)


def make_summary_figure(
    inlier_match_path: str,
    overlay_path: str,
    save_path: str,
    summary_width: int = 1200,
) -> None:
    """Create one compact summary figure for the paper/demo."""
    if not Path(inlier_match_path).exists() or not Path(overlay_path).exists():
        return

    match_img = Image.open(inlier_match_path).convert("RGB")
    overlay_img = Image.open(overlay_path).convert("RGB")

    match_img = resize_to_width(match_img, summary_width)
    overlay_img = resize_to_width(overlay_img, summary_width)
    overlay_img = add_title(overlay_img, "Warped UAV image overlaid on satellite image")

    gap = 24
    canvas = Image.new(
        "RGB",
        (summary_width, match_img.height + gap + overlay_img.height),
        "white",
    )
    canvas.paste(match_img, (0, 0))
    canvas.paste(overlay_img, (0, match_img.height + gap))
    canvas.save(save_path)


def save_homography(path: str, H: Optional[np.ndarray]) -> None:
    """Save homography matrix."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if H is None:
            f.write("Homography estimation failed.\n")
        else:
            f.write("# UAV image -> satellite image homography\n")
            np.savetxt(f, H, fmt="%.10f")


def save_statistics(path: str, stats: Dict[str, float]) -> None:
    """Save statistics as readable text."""
    with open(path, "w", encoding="utf-8") as f:
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")


def find_image_pairs(batch_dir: Path):
    """Find UAV-satellite image pairs in a directory.

    Matches files whose names differ only by 'uav' <-> 'sat' (case-insensitive).
    Returns list of (uav_path: str, sat_path: str, pair_name: str) tuples.
    """
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    image_files = sorted(
        [f for f in batch_dir.iterdir() if f.is_file() and f.suffix.lower() in extensions]
    )

    pairs = []
    used = set()

    for f in image_files:
        if f in used:
            continue
        name = f.stem
        if "uav" in name.lower():
            partner_name = re.sub("uav", "sat", name, flags=re.IGNORECASE)
        elif "sat" in name.lower():
            partner_name = re.sub("sat", "uav", name, flags=re.IGNORECASE)
        else:
            continue

        partner = None
        for other in image_files:
            if other.stem.lower() == partner_name.lower() and other not in used:
                partner = other
                break

        if partner is not None:
            used.add(f)
            used.add(partner)
            if "uav" in f.stem.lower():
                uav_path, sat_path = str(f), str(partner)
            else:
                uav_path, sat_path = str(partner), str(f)
            pair_name = re.sub(
                r"[\_\-]?(uav|sat)[\_\-]?", "", f.stem, flags=re.IGNORECASE
            )
            pairs.append((uav_path, sat_path, pair_name))

    return pairs


def compute_reprojection_error(
    kpts_uav: np.ndarray,
    kpts_sat: np.ndarray,
    H: Optional[np.ndarray],
    inlier_mask: Optional[np.ndarray],
) -> float:
    """Compute mean reprojection error (pixels) for RANSAC inliers."""
    if H is None or inlier_mask is None or not inlier_mask.any():
        return 0.0

    uav_in = kpts_uav[inlier_mask]
    sat_in = kpts_sat[inlier_mask]

    ones = np.ones((len(uav_in), 1), dtype=np.float64)
    uav_h = np.hstack([uav_in, ones])
    projected = uav_h @ H.T
    projected = projected[:, :2] / projected[:, 2:3]

    errors = np.linalg.norm(projected - sat_in, axis=1)
    return float(np.mean(errors))


def parse_args():
    parser = ArgumentParser(description="RoMa demo for UAV-satellite image registration.")
    parser.add_argument("--uav_path", type=str, help="Path to UAV/drone image (ignored in batch mode).")
    parser.add_argument("--sat_path", type=str, help="Path to satellite image (ignored in batch mode).")
    parser.add_argument("--batch_dir", type=str, default=None, help="Directory of image pairs for batch processing.")
    parser.add_argument("--output_dir", default="result/roma_uav_sat", type=str, help="Directory to save results.")

    # RoMa resolution settings. Defaults are close to the official demo.
    parser.add_argument("--coarse_res", default=560, type=int)
    parser.add_argument("--upsample_h", default=864, type=int)
    parser.add_argument("--upsample_w", default=1152, type=int)

    # Matching and RANSAC settings.
    parser.add_argument("--num_matches", default=10000, type=int, help="Number of RoMa matches to sample.")
    parser.add_argument("--conf_thr", default=0.05, type=float, help="Confidence threshold before RANSAC.")
    parser.add_argument("--ransac_thr", default=5.0, type=float, help="RANSAC reprojection threshold in pixels.")

    # Visualization settings.
    parser.add_argument("--max_vis_matches", default=300, type=int, help="Maximum matches drawn in each figure.")
    parser.add_argument("--vis_height", default=520, type=int, help="Display height for match visualization.")
    parser.add_argument("--overlay_alpha", default=0.50, type=float, help="Alpha of warped UAV image in overlay.")
    parser.add_argument("--device", default="auto", type=str, help="auto, cuda, cpu, mps, etc.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)
    print(f"[Info] device = {device}")
    print(f"[Info] UAV image       = {args.uav_path}")
    print(f"[Info] Satellite image = {args.sat_path}")

    uav_img = load_rgb(args.uav_path)
    sat_img = load_rgb(args.sat_path)

    print("[Info] Loading RoMa outdoor model...")
    roma_model = roma_outdoor(
        device=device,
        coarse_res=args.coarse_res,
        upsample_res=(args.upsample_h, args.upsample_w),
    )

    with torch.no_grad():
        print("[Info] Running RoMa dense matching...")
        warp, certainty = roma_model.match(args.uav_path, args.sat_path, device=device)

    print("[Info] Sampling dense correspondences...")
    kpts_uav, kpts_sat, conf = sample_roma_matches(
        roma_model=roma_model,
        warp=warp,
        certainty=certainty,
        uav_size=uav_img.size,
        sat_size=sat_img.size,
        num_matches=args.num_matches,
    )

    raw_matches = len(kpts_uav)
    kpts_uav, kpts_sat, conf = filter_matches_by_confidence(
        kpts_uav, kpts_sat, conf, args.conf_thr
    )
    kept_matches = len(kpts_uav)

    print(f"[Info] sampled matches = {raw_matches}")
    print(f"[Info] kept matches after conf_thr={args.conf_thr} = {kept_matches}")

    matches_all_path = output_dir / "matches_all.png"
    draw_matches(
        uav_img=uav_img,
        sat_img=sat_img,
        kpts_uav=kpts_uav,
        kpts_sat=kpts_sat,
        conf=conf,
        save_path=str(matches_all_path),
        title="RoMa sampled correspondences before RANSAC",
        inlier_mask=None,
        max_vis_matches=args.max_vis_matches,
        vis_height=args.vis_height,
    )

    H_uav_to_sat, inlier_mask = estimate_homography(kpts_uav, kpts_sat, args.ransac_thr)

    if inlier_mask is None:
        inliers = 0
        inlier_ratio = 0.0
        inlier_mask_for_vis = np.zeros(len(kpts_uav), dtype=bool)
    else:
        inliers = int(inlier_mask.sum())
        inlier_ratio = inliers / max(1, kept_matches)
        inlier_mask_for_vis = inlier_mask

    print(f"[Result] RANSAC inliers = {inliers}/{kept_matches}, ratio = {inlier_ratio:.4f}")

    matches_inliers_path = output_dir / "matches_inliers.png"
    draw_matches(
        uav_img=uav_img,
        sat_img=sat_img,
        kpts_uav=kpts_uav,
        kpts_sat=kpts_sat,
        conf=conf,
        save_path=str(matches_inliers_path),
        title="RoMa + RANSAC inlier correspondences",
        inlier_mask=inlier_mask_for_vis,
        max_vis_matches=args.max_vis_matches,
        vis_height=args.vis_height,
    )

    homography_path = output_dir / "homography.txt"
    save_homography(str(homography_path), H_uav_to_sat)

    warped_path = output_dir / "warped_uav_to_sat.png"
    overlay_path = output_dir / "warped_overlay.png"
    summary_path = output_dir / "registration_result.png"

    if H_uav_to_sat is not None:
        warped_img, overlay_img = warp_uav_to_satellite(
            uav_img=uav_img,
            sat_img=sat_img,
            H_uav_to_sat=H_uav_to_sat,
            alpha=args.overlay_alpha,
        )
        warped_img.save(warped_path)
        overlay_img.save(overlay_path)
        make_summary_figure(
            inlier_match_path=str(matches_inliers_path),
            overlay_path=str(overlay_path),
            save_path=str(summary_path),
        )
    else:
        print("[Warn] Homography is unavailable, so warped overlay is not generated.")

    stats = {
        "raw_sampled_matches": raw_matches,
        "confidence_threshold": args.conf_thr,
        "kept_matches": kept_matches,
        "ransac_threshold_px": args.ransac_thr,
        "ransac_inliers": inliers,
        "ransac_inlier_ratio": round(float(inlier_ratio), 6),
        "uav_width": uav_img.width,
        "uav_height": uav_img.height,
        "satellite_width": sat_img.width,
        "satellite_height": sat_img.height,
    }
    save_statistics(str(output_dir / "statistics.txt"), stats)

    print(f"[Saved] {matches_all_path}")
    print(f"[Saved] {matches_inliers_path}")
    print(f"[Saved] {homography_path}")
    if H_uav_to_sat is not None:
        print(f"[Saved] {warped_path}")
        print(f"[Saved] {overlay_path}")
        print(f"[Saved] {summary_path}")


if __name__ == "__main__":
    main()
