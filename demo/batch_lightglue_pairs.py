"""
Batch LightGlue image matching for UAV--satellite image pairs.

Example: use two folders, paired by sorted order
    python demo/batch_lightglue_pairs.py --uav_dir data/uav --sat_dir data/sat --features superpoint

format:
    data/uav/001.jpg,data/sat/001.jpg
"""

import csv
import math
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet
from lightglue.utils import load_image, rbd


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def get_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def build_lightglue_models(
    features: str,
    max_num_keypoints: int,
    device: torch.device,
):
    """
    Build extractor and LightGlue matcher.

    Supported features:
        superpoint, disk, sift, aliked, doghardnet
    """
    features = features.lower()

    if features == "superpoint":
        extractor = SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(device)
        matcher = LightGlue(features="superpoint").eval().to(device)

    elif features == "disk":
        extractor = DISK(max_num_keypoints=max_num_keypoints).eval().to(device)
        matcher = LightGlue(features="disk").eval().to(device)

    elif features == "sift":
        extractor = SIFT(max_num_keypoints=max_num_keypoints).eval().to(device)
        matcher = LightGlue(features="sift").eval().to(device)

    elif features == "aliked":
        extractor = ALIKED(max_num_keypoints=max_num_keypoints).eval().to(device)
        matcher = LightGlue(features="aliked").eval().to(device)

    elif features == "doghardnet":
        extractor = DoGHardNet(max_num_keypoints=max_num_keypoints).eval().to(device)
        matcher = LightGlue(features="doghardnet").eval().to(device)

    else:
        raise ValueError(
            f"Unsupported features: {features}. "
            f"Choose from: superpoint, disk, sift, aliked, doghardnet."
        )

    return extractor, matcher


def parse_pair_line(line: str) -> Optional[Tuple[str, str]]:
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
    pairs = []

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


def load_pairs_from_dirs(
    uav_dir: str,
    sat_dir: str,
    num_pairs: int,
) -> List[Tuple[str, str]]:
    uav_paths = list_images(uav_dir)
    sat_paths = list_images(sat_dir)

    n = min(len(uav_paths), len(sat_paths), num_pairs)

    return list(zip(uav_paths[:n], sat_paths[:n]))


def match_one_pair(
    extractor,
    matcher,
    image0_path: str,
    image1_path: str,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Match one image pair using LightGlue.

    image0: UAV image
    image1: satellite image

    Returns:
        points0: matched points in image0, shape [K, 2]
        points1: matched points in image1, shape [K, 2]
        scores : matching confidence scores, shape [K]
    """
    image0 = load_image(image0_path).to(device)
    image1 = load_image(image1_path).to(device)

    with torch.no_grad():
        feats0 = extractor.extract(image0)
        feats1 = extractor.extract(image1)

        matches01 = matcher({
            "image0": feats0,
            "image1": feats1,
        })

    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

    matches = matches01["matches"]  # [K, 2]

    points0 = feats0["keypoints"][matches[:, 0]]
    points1 = feats1["keypoints"][matches[:, 1]]

    points0 = points0.detach().cpu().numpy().astype(np.float32)
    points1 = points1.detach().cpu().numpy().astype(np.float32)

    if "scores" in matches01:
        scores = matches01["scores"].detach().cpu().numpy().astype(np.float32)
    elif "matching_scores0" in matches01:
        scores = matches01["matching_scores0"][matches[:, 0]]
        scores = scores.detach().cpu().numpy().astype(np.float32)
    else:
        scores = np.ones(len(points0), dtype=np.float32)

    return points0, points1, scores


def estimate_homography(
    points0: np.ndarray,
    points1: np.ndarray,
    ransac_thr: float,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Estimate homography from image0 to image1.

    For UAV-satellite registration:
        H maps UAV image coordinates to satellite image coordinates.
    """
    if len(points0) < 4:
        return None, None

    H, mask = cv2.findHomography(
        points0,
        points1,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thr,
    )

    if H is None or mask is None:
        return None, None

    return H.astype(np.float64), mask.reshape(-1).astype(bool)


def project_points_by_homography(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    points = points.astype(np.float64)

    ones = np.ones((len(points), 1), dtype=np.float64)
    points_h = np.concatenate([points, ones], axis=1)

    proj_h = points_h @ H.T
    denom = proj_h[:, 2:3]

    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)

    return proj_h[:, :2] / denom


def compute_reprojection_error(
    points0: np.ndarray,
    points1: np.ndarray,
    H: Optional[np.ndarray],
    inlier_mask: Optional[np.ndarray],
) -> Dict[str, float]:
    """
    Compute reprojection error on RANSAC inliers.

    Error = || H * p_uav - p_sat ||_2
    """
    if H is None or inlier_mask is None or int(inlier_mask.sum()) == 0:
        return {
            "reproj_error_mean": math.nan,
            "reproj_error_median": math.nan,
            "reproj_error_max": math.nan,
        }

    src = points0[inlier_mask]
    dst = points1[inlier_mask]

    pred = project_points_by_homography(src, H)
    err = np.linalg.norm(pred - dst, axis=1)

    return {
        "reproj_error_mean": float(np.mean(err)),
        "reproj_error_median": float(np.median(err)),
        "reproj_error_max": float(np.max(err)),
    }


def save_homography(path: str, H: Optional[np.ndarray]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        if H is None:
            f.write("Homography estimation failed.\n")
        else:
            f.write("# image0 UAV -> image1 satellite homography\n")
            np.savetxt(f, H, fmt="%.10f")


def save_statistics(path: str, stats: Dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")


def draw_matches(
    image0_path: str,
    image1_path: str,
    points0: np.ndarray,
    points1: np.ndarray,
    inlier_mask: Optional[np.ndarray],
    save_path: str,
    max_vis_matches: int = 300,
) -> None:
    """
    Save a simple side-by-side match visualization.
    Red lines are matches. If inlier_mask is given, only RANSAC inliers are drawn.
    """
    img0 = cv2.imread(image0_path)
    img1 = cv2.imread(image1_path)

    if img0 is None or img1 is None:
        return

    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]

    target_h = 520

    scale0 = target_h / h0
    scale1 = target_h / h1

    img0_vis = cv2.resize(img0, (int(w0 * scale0), target_h))
    img1_vis = cv2.resize(img1, (int(w1 * scale1), target_h))

    canvas = np.concatenate([img0_vis, img1_vis], axis=1)
    offset_x = img0_vis.shape[1]

    if inlier_mask is not None:
        indices = np.where(inlier_mask)[0]
    else:
        indices = np.arange(len(points0))

    if len(indices) > max_vis_matches:
        indices = indices[:max_vis_matches]

    for idx in indices:
        x0, y0 = points0[idx]
        x1, y1 = points1[idx]

        p0 = (int(x0 * scale0), int(y0 * scale0))
        p1 = (int(x1 * scale1) + offset_x, int(y1 * scale1))

        cv2.circle(canvas, p0, 2, (0, 0, 255), -1)
        cv2.circle(canvas, p1, 2, (255, 0, 0), -1)
        cv2.line(canvas, p0, p1, (0, 255, 0), 1)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(save_path, canvas)


def process_one_pair(
    pair_id: int,
    extractor,
    matcher,
    uav_path: str,
    sat_path: str,
    device: torch.device,
    args,
) -> Dict:
    pair_name = f"pair_{pair_id:03d}"
    pair_dir = Path(args.output_dir) / pair_name
    pair_dir.mkdir(parents=True, exist_ok=True)

    points0, points1, scores = match_one_pair(
        extractor=extractor,
        matcher=matcher,
        image0_path=uav_path,
        image1_path=sat_path,
        device=device,
    )

    raw_matches = len(points0)

    H, inlier_mask = estimate_homography(
        points0=points0,
        points1=points1,
        ransac_thr=args.ransac_thr,
    )

    if inlier_mask is None:
        inliers = 0
        inlier_ratio = 0.0
        success = 0
    else:
        inliers = int(inlier_mask.sum())
        inlier_ratio = inliers / max(1, raw_matches)
        success = 1 if H is not None else 0

    reproj_stats = compute_reprojection_error(
        points0=points0,
        points1=points1,
        H=H,
        inlier_mask=inlier_mask,
    )

    stats = {
        "pair_id": pair_id,
        "uav_path": uav_path,
        "sat_path": sat_path,
        "features": args.features,
        "success": success,
        "matches": raw_matches,
        "ransac_inliers": inliers,
        "ransac_inlier_ratio": float(inlier_ratio),
        **reproj_stats,
    }

    save_homography(str(pair_dir / "homography.txt"), H)
    save_statistics(str(pair_dir / "statistics.txt"), stats)

    np.savez(
        pair_dir / "matches.npz",
        points0=points0,
        points1=points1,
        scores=scores,
        inlier_mask=inlier_mask if inlier_mask is not None else np.zeros(raw_matches, dtype=bool),
        H=H if H is not None else np.eye(3),
    )

    if args.save_vis:
        draw_matches(
            image0_path=uav_path,
            image1_path=sat_path,
            points0=points0,
            points1=points1,
            inlier_mask=inlier_mask,
            save_path=str(pair_dir / "matches_inliers.png"),
            max_vis_matches=args.max_vis_matches,
        )

        draw_matches(
            image0_path=uav_path,
            image1_path=sat_path,
            points0=points0,
            points1=points1,
            inlier_mask=None,
            save_path=str(pair_dir / "matches_all.png"),
            max_vis_matches=args.max_vis_matches,
        )

    return stats


def nanmean(values: List[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)

    if arr.size == 0 or np.all(np.isnan(arr)):
        return math.nan

    return float(np.nanmean(arr))


def save_batch_results(output_dir: str, results: List[Dict]) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "batch_results.csv"

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
        "avg_matches": nanmean([float(r["matches"]) for r in results]),
        "avg_ransac_inliers": nanmean([float(r["ransac_inliers"]) for r in results]),
        "avg_ransac_inlier_ratio": nanmean([float(r["ransac_inlier_ratio"]) for r in results]),
        "avg_reproj_error_mean_px": nanmean([float(r["reproj_error_mean"]) for r in results]),
        "avg_reproj_error_median_px": nanmean([float(r["reproj_error_median"]) for r in results]),
    }

    summary_path = output_dir / "batch_summary.txt"

    with open(summary_path, "w", encoding="utf-8") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    print("\n========== Batch Summary ==========")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print(f"\n[Saved] {csv_path}")
    print(f"[Saved] {summary_path}")


def parse_args():
    parser = ArgumentParser(description="Batch LightGlue matching for UAV-satellite image pairs.")

    input_group = parser.add_mutually_exclusive_group(required=True)

    input_group.add_argument(
        "--pairs_txt",
        type=str,
        help="Txt/csv file. Each line contains uav_path and sat_path.",
    )
    input_group.add_argument(
        "--uav_dir",
        type=str,
        help="UAV image directory. Must be used with --sat_dir.",
    )

    parser.add_argument(
        "--sat_dir",
        type=str,
        default=None,
        help="Satellite image directory, paired with UAV images by sorted order.",
    )

    parser.add_argument("--output_dir", type=str, default="result/lightglue_batch")
    parser.add_argument("--num_pairs", type=int, default=20)

    parser.add_argument(
        "--features",
        type=str,
        default="superpoint",
        choices=["superpoint", "disk", "sift", "aliked", "doghardnet"],
        help="Local feature extractor used with LightGlue.",
    )

    parser.add_argument("--max_num_keypoints", type=int, default=2048)
    parser.add_argument("--ransac_thr", type=float, default=5.0)
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--save_vis", action="store_true")
    parser.add_argument("--max_vis_matches", type=int, default=300)

    return parser.parse_args()


def main():
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

    device = get_device(args.device)

    print(f"[Info] device = {device}")
    print(f"[Info] features = {args.features}")
    print(f"[Info] max_num_keypoints = {args.max_num_keypoints}")
    print(f"[Info] ransac_thr = {args.ransac_thr}")
    print(f"[Info] num_pairs = {len(pairs)}")
    print(f"[Info] output_dir = {args.output_dir}")

    print("[Info] Loading extractor and LightGlue matcher once...")
    extractor, matcher = build_lightglue_models(
        features=args.features,
        max_num_keypoints=args.max_num_keypoints,
        device=device,
    )

    results = []

    for i, (uav_path, sat_path) in enumerate(pairs, start=1):
        print(f"\n[{i}/{len(pairs)}] UAV: {uav_path}")
        print(f"[{i}/{len(pairs)}] SAT: {sat_path}")

        try:
            stats = process_one_pair(
                pair_id=i,
                extractor=extractor,
                matcher=matcher,
                uav_path=uav_path,
                sat_path=sat_path,
                device=device,
                args=args,
            )

            results.append(stats)

            print(
                f"[Result] matches={stats['matches']} "
                f"inliers={stats['ransac_inliers']} "
                f"ratio={stats['ransac_inlier_ratio']:.4f} "
                f"mean_reproj={stats['reproj_error_mean']:.4f}px"
            )

        except Exception as e:
            print(f"[Error] Failed on pair {i}: {e}")

            results.append({
                "pair_id": i,
                "uav_path": uav_path,
                "sat_path": sat_path,
                "features": args.features,
                "success": 0,
                "matches": 0,
                "ransac_inliers": 0,
                "ransac_inlier_ratio": 0.0,
                "reproj_error_mean": math.nan,
                "reproj_error_median": math.nan,
                "reproj_error_max": math.nan,
            })

    save_batch_results(args.output_dir, results)


if __name__ == "__main__":
    main()