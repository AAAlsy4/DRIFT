"""
Batch RoMa UAV--satellite registration for 20 image pairs.

Input:
    python demo/batch_pairs.py --uav_dir data/uav --sat_dir data/sat

Outputs:
    batch_results.csv     per-pair metrics
    batch_summary.txt     average metrics over the processed pairs
    pair_001/ ...         optional per-pair visualizations/statistics if --save_vis is used
"""

import csv
import math
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import warnings
warnings.filterwarnings("ignore", message="Local correlation is not supported on non-Linux platforms")

from demo_match import (
    get_device,
    load_rgb,
    sample_roma_matches,
    filter_matches_by_confidence,
    estimate_homography,
    save_homography,
    save_statistics,
    draw_matches,
    warp_uav_to_satellite,
    make_summary_figure,
    roma_outdoor,
)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_pair_line(line: str) -> Optional[Tuple[str, str]]:
    """Parse one line from pairs.txt."""
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    if "," in line:
        parts = [p.strip() for p in line.split(",") if p.strip()]
    else:
        parts = line.split()

    if len(parts) < 2:
        raise ValueError(f"Invalid pair line: {line}")
    return parts[0], parts[1]


def load_pairs_from_txt(pairs_txt: str, num_pairs: int) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    with open(pairs_txt, "r", encoding="utf-8") as f:
        for line in f:
            pair = parse_pair_line(line)
            if pair is not None:
                pairs.append(pair)
            if len(pairs) >= num_pairs:
                break
    return pairs


def list_images(folder: str) -> List[str]:
    paths = []
    for p in Path(folder).iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            paths.append(str(p))
    return sorted(paths)


def load_pairs_from_dirs(uav_dir: str, sat_dir: str, num_pairs: int) -> List[Tuple[str, str]]:
    uav_paths = list_images(uav_dir)
    sat_paths = list_images(sat_dir)
    n = min(len(uav_paths), len(sat_paths), num_pairs)
    return list(zip(uav_paths[:n], sat_paths[:n]))


def project_points_by_homography(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Project [N, 2] points with homography H."""
    points = points.astype(np.float64)
    ones = np.ones((len(points), 1), dtype=np.float64)
    pts_h = np.concatenate([points, ones], axis=1)  # [N, 3]
    proj_h = pts_h @ H.T
    denom = proj_h[:, 2:3]
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    return proj_h[:, :2] / denom


def compute_reprojection_error(
    kpts_uav: np.ndarray,
    kpts_sat: np.ndarray,
    H_uav_to_sat: Optional[np.ndarray],
    inlier_mask: Optional[np.ndarray],
) -> Dict[str, float]:
    """
    Compute reprojection error on RANSAC inliers.

    Error = || H * p_uav - p_sat ||_2, in satellite pixel coordinates.
    """
    if H_uav_to_sat is None or inlier_mask is None or int(inlier_mask.sum()) == 0:
        return {
            "reproj_error_mean": math.nan,
            "reproj_error_median": math.nan,
            "reproj_error_max": math.nan,
        }

    src = kpts_uav[inlier_mask]
    dst = kpts_sat[inlier_mask]
    pred = project_points_by_homography(src, H_uav_to_sat)
    err = np.linalg.norm(pred - dst, axis=1)

    return {
        "reproj_error_mean": float(np.mean(err)),
        "reproj_error_median": float(np.median(err)),
        "reproj_error_max": float(np.max(err)),
    }


def process_one_pair(
    pair_id: int,
    roma_model,
    device: torch.device,
    uav_path: str,
    sat_path: str,
    args,
) -> Dict[str, float]:
    """Run RoMa + RANSAC + reprojection-error evaluation for one image pair."""
    pair_name = f"pair_{pair_id:03d}"
    pair_dir = Path(args.output_dir) / pair_name
    pair_dir.mkdir(parents=True, exist_ok=True)

    uav_img = load_rgb(uav_path)
    sat_img = load_rgb(sat_path)

    with torch.no_grad():
        warp, certainty = roma_model.match(uav_path, sat_path, device=device)

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
        kpts_uav=kpts_uav,
        kpts_sat=kpts_sat,
        conf=conf,
        conf_thr=args.conf_thr,
    )
    kept_matches = len(kpts_uav)

    H_uav_to_sat, inlier_mask = estimate_homography(
        kpts_uav=kpts_uav,
        kpts_sat=kpts_sat,
        ransac_thr=args.ransac_thr,
    )

    if inlier_mask is None:
        inliers = 0
        inlier_ratio = 0.0
        success = 0
        inlier_mask_for_vis = np.zeros(kept_matches, dtype=bool)
    else:
        inliers = int(inlier_mask.sum())
        inlier_ratio = inliers / max(1, kept_matches)
        success = 1 if H_uav_to_sat is not None else 0
        inlier_mask_for_vis = inlier_mask

    reproj_stats = compute_reprojection_error(
        kpts_uav=kpts_uav,
        kpts_sat=kpts_sat,
        H_uav_to_sat=H_uav_to_sat,
        inlier_mask=inlier_mask,
    )

    stats: Dict[str, float] = {
        "pair_id": pair_id,
        "uav_path": uav_path,
        "sat_path": sat_path,
        "success": success,
        "raw_sampled_matches": raw_matches,
        "kept_matches": kept_matches,
        "ransac_inliers": inliers,
        "ransac_inlier_ratio": float(inlier_ratio),
        "uav_width": uav_img.width,
        "uav_height": uav_img.height,
        "satellite_width": sat_img.width,
        "satellite_height": sat_img.height,
        **reproj_stats,
    }

    # Save lightweight per-pair text result even when --save_vis is not used.
    save_homography(str(pair_dir / "homography.txt"), H_uav_to_sat)
    save_statistics(str(pair_dir / "statistics.txt"), stats)

    if args.save_vis:
        matches_all_path = pair_dir / "matches_all.png"
        matches_inliers_path = pair_dir / "matches_inliers.png"
        warped_path = pair_dir / "warped_uav_to_sat.png"
        overlay_path = pair_dir / "warped_overlay.png"
        summary_path = pair_dir / "registration_result.png"

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

    return stats


def nanmean(values: List[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return math.nan
    return float(np.nanmean(arr))


def save_batch_results(output_dir: str, results: List[Dict[str, float]]) -> None:
    output_dir = str(output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    csv_path = Path(output_dir) / "batch_results.csv"
    if len(results) > 0:
        fieldnames = list(results[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    summary = {
        "num_pairs_processed": len(results),
        "success_pairs": int(sum(int(r["success"]) for r in results)),
        "success_rate": nanmean([float(r["success"]) for r in results]),
        "avg_raw_sampled_matches": nanmean([float(r["raw_sampled_matches"]) for r in results]),
        "avg_kept_matches": nanmean([float(r["kept_matches"]) for r in results]),
        "avg_ransac_inliers": nanmean([float(r["ransac_inliers"]) for r in results]),
        "avg_ransac_inlier_ratio": nanmean([float(r["ransac_inlier_ratio"]) for r in results]),
        "avg_reproj_error_mean_px_valid_only": nanmean([float(r["reproj_error_mean"]) for r in results]),
        "avg_reproj_error_median_px_valid_only": nanmean([float(r["reproj_error_median"]) for r in results]),
    }

    summary_path = Path(output_dir) / "batch_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    print("\n========== Batch Summary ==========")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"\n[Saved] {csv_path}")
    print(f"[Saved] {summary_path}")


def parse_args():
    parser = ArgumentParser(description="Batch RoMa registration for 20 UAV-satellite image pairs.")

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--pairs_txt", type=str, help="Txt/csv file: each line contains uav_path and sat_path.")
    input_group.add_argument("--uav_dir", type=str, help="UAV image directory. Must be used with --sat_dir.")
    parser.add_argument("--sat_dir", type=str, default=None, help="Satellite image directory, paired by sorted order.")

    parser.add_argument("--output_dir", default="result/batch_20", type=str)
    parser.add_argument("--num_pairs", default=1000, type=int)

    # Keep the same RoMa settings as demo_match.py.
    parser.add_argument("--coarse_res", default=560, type=int)
    parser.add_argument("--upsample_h", default=864, type=int)
    parser.add_argument("--upsample_w", default=1152, type=int)
    parser.add_argument("--num_matches", default=2000, type=int)
    parser.add_argument("--conf_thr", default=0.05, type=float)
    parser.add_argument("--ransac_thr", default=5.0, type=float)
    parser.add_argument("--device", default="auto", type=str, help="auto, cuda, cpu, mps, etc.")

    # Visualization is optional because saving 20 groups of figures is slower.
    parser.add_argument("--save_vis", action="store_true", help="Save match/warp visualizations for each pair.")
    parser.add_argument("--max_vis_matches", default=300, type=int)
    parser.add_argument("--vis_height", default=520, type=int)
    parser.add_argument("--overlay_alpha", default=0.50, type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.uav_dir is not None and args.sat_dir is None:
        raise ValueError("When using --uav_dir, you must also set --sat_dir.")

    if args.pairs_txt is not None:
        pairs = load_pairs_from_txt(args.pairs_txt, args.num_pairs)
    else:
        pairs = load_pairs_from_dirs(args.uav_dir, args.sat_dir, args.num_pairs)

    if len(pairs) == 0:
        raise RuntimeError("No image pairs found.")

    print(f"[Info] Will process {len(pairs)} image pairs.")
    print(f"[Info] Output directory: {args.output_dir}")

    device = get_device(args.device)
    print(f"[Info] device = {device}")
    print("[Info] Loading RoMa outdoor model once...")
    roma_model = roma_outdoor(
        device=device,
        coarse_res=args.coarse_res,
        upsample_res=(args.upsample_h, args.upsample_w),
    )

    results: List[Dict[str, float]] = []
    for i, (uav_path, sat_path) in enumerate(pairs, start=1):
        print(f"\n[{i}/{len(pairs)}] UAV: {uav_path}")
        print(f"[{i}/{len(pairs)}] SAT: {sat_path}")
        try:
            stats = process_one_pair(
                pair_id=i,
                roma_model=roma_model,
                device=device,
                uav_path=uav_path,
                sat_path=sat_path,
                args=args,
            )
            results.append(stats)
            print(
                f"[Result] inliers={stats['ransac_inliers']}/{stats['kept_matches']} "
                f"ratio={stats['ransac_inlier_ratio']:.4f} "
                f"mean_reproj={stats['reproj_error_mean']:.4f}px"
            )
        except Exception as e:
            print(f"[Error] Failed on pair {i}: {e}")
            results.append({
                "pair_id": i,
                "uav_path": uav_path,
                "sat_path": sat_path,
                "success": 0,
                "raw_sampled_matches": 0,
                "kept_matches": 0,
                "ransac_inliers": 0,
                "ransac_inlier_ratio": 0.0,
                "uav_width": math.nan,
                "uav_height": math.nan,
                "satellite_width": math.nan,
                "satellite_height": math.nan,
                "reproj_error_mean": math.nan,
                "reproj_error_median": math.nan,
                "reproj_error_max": math.nan,
            })

    save_batch_results(args.output_dir, results)


if __name__ == "__main__":
    main()
