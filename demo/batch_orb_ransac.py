"""
Batch ORB + RANSAC UAV--satellite registration.

Outputs:
    batch_orb_results.csv
    batch_orb_summary.txt

Metrics:
    平均匹配点数
    平均内点数
    平均内点率
    平均重投影误差 / px
"""

import csv
import math
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


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


def load_pairs_from_dirs(uav_dir: str, sat_dir: str, num_pairs: int) -> List[Tuple[str, str]]:
    uav_paths = list_images(uav_dir)
    sat_paths = list_images(sat_dir)
    n = min(len(uav_paths), len(sat_paths), num_pairs)
    return list(zip(uav_paths[:n], sat_paths[:n]))


def compute_reprojection_error(
    kpts_uav: np.ndarray,
    kpts_sat: np.ndarray,
    H: Optional[np.ndarray],
    inlier_mask: Optional[np.ndarray],
) -> Dict[str, float]:
    if H is None or inlier_mask is None or int(inlier_mask.sum()) == 0:
        return {
            "reproj_error_mean": math.nan,
            "reproj_error_median": math.nan,
            "reproj_error_max": math.nan,
        }

    src = kpts_uav[inlier_mask].reshape(-1, 1, 2).astype(np.float32)
    dst = kpts_sat[inlier_mask].astype(np.float32)

    pred = cv2.perspectiveTransform(src, H).reshape(-1, 2)
    err = np.linalg.norm(pred - dst, axis=1)

    return {
        "reproj_error_mean": float(np.mean(err)),
        "reproj_error_median": float(np.median(err)),
        "reproj_error_max": float(np.max(err)),
    }


def orb_ransac_one_pair(
    pair_id: int,
    uav_path: str,
    sat_path: str,
    args,
) -> Dict[str, float]:
    img_uav = cv2.imread(uav_path, cv2.IMREAD_GRAYSCALE)
    img_sat = cv2.imread(sat_path, cv2.IMREAD_GRAYSCALE)

    if img_uav is None:
        raise RuntimeError(f"Failed to read UAV image: {uav_path}")
    if img_sat is None:
        raise RuntimeError(f"Failed to read satellite image: {sat_path}")

    orb = cv2.ORB_create(
        nfeatures=args.nfeatures,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=31,
        firstLevel=0,
        WTA_K=2,
        scoreType=cv2.ORB_HARRIS_SCORE,
        patchSize=31,
        fastThreshold=args.fast_threshold,
    )

    kp1, des1 = orb.detectAndCompute(img_uav, None)
    kp2, des2 = orb.detectAndCompute(img_sat, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return {
            "pair_id": pair_id,
            "uav_path": uav_path,
            "sat_path": sat_path,
            "success": 0,
            "num_keypoints_uav": len(kp1) if kp1 is not None else 0,
            "num_keypoints_sat": len(kp2) if kp2 is not None else 0,
            "matched_points": 0,
            "ransac_inliers": 0,
            "ransac_inlier_ratio": 0.0,
            "reproj_error_mean": math.nan,
            "reproj_error_median": math.nan,
            "reproj_error_max": math.nan,
        }

    # ORB 是二进制描述子，所以使用 Hamming 距离
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    knn_matches = matcher.knnMatch(des1, des2, k=2)

    good_matches = []
    for item in knn_matches:
        if len(item) < 2:
            continue
        m, n = item
        if m.distance < args.ratio * n.distance:
            good_matches.append(m)

    # 按 Hamming 距离从小到大排序
    good_matches = sorted(good_matches, key=lambda x: x.distance)

    # 为了和 RoMa Top-K=2000 对齐，最多保留 max_matches 个匹配进入 RANSAC
    if args.max_matches > 0:
        good_matches = good_matches[:args.max_matches]

    matched_points = len(good_matches)

    if matched_points < 4:
        return {
            "pair_id": pair_id,
            "uav_path": uav_path,
            "sat_path": sat_path,
            "success": 0,
            "num_keypoints_uav": len(kp1),
            "num_keypoints_sat": len(kp2),
            "matched_points": matched_points,
            "ransac_inliers": 0,
            "ransac_inlier_ratio": 0.0,
            "reproj_error_mean": math.nan,
            "reproj_error_median": math.nan,
            "reproj_error_max": math.nan,
        }

    kpts_uav = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    kpts_sat = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    H, mask = cv2.findHomography(
        kpts_uav,
        kpts_sat,
        method=cv2.RANSAC,
        ransacReprojThreshold=args.ransac_thr,
    )

    if H is None or mask is None:
        inlier_mask = None
        inliers = 0
        inlier_ratio = 0.0
        success = 0
    else:
        inlier_mask = mask.reshape(-1).astype(bool)
        inliers = int(inlier_mask.sum())
        inlier_ratio = inliers / max(1, matched_points)
        success = 1

    reproj_stats = compute_reprojection_error(
        kpts_uav=kpts_uav,
        kpts_sat=kpts_sat,
        H=H,
        inlier_mask=inlier_mask,
    )

    return {
        "pair_id": pair_id,
        "uav_path": uav_path,
        "sat_path": sat_path,
        "success": success,
        "num_keypoints_uav": len(kp1),
        "num_keypoints_sat": len(kp2),
        "matched_points": matched_points,
        "ransac_inliers": inliers,
        "ransac_inlier_ratio": float(inlier_ratio),
        **reproj_stats,
    }


def nanmean(values: List[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return math.nan
    return float(np.nanmean(arr))


def save_results(output_dir: str, results: List[Dict[str, float]]) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "batch_orb_results.csv"
    summary_path = output_dir / "batch_orb_summary.txt"

    if len(results) > 0:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

    summary = {
        "num_pairs_processed": len(results),
        "success_pairs": int(sum(int(r["success"]) for r in results)),
        "success_rate": nanmean([float(r["success"]) for r in results]),
        "avg_matched_points": nanmean([float(r["matched_points"]) for r in results]),
        "avg_ransac_inliers": nanmean([float(r["ransac_inliers"]) for r in results]),
        "avg_ransac_inlier_ratio": nanmean([float(r["ransac_inlier_ratio"]) for r in results]),
        "avg_reproj_error_mean_px_valid_only": nanmean([float(r["reproj_error_mean"]) for r in results]),
        "avg_reproj_error_median_px_valid_only": nanmean([float(r["reproj_error_median"]) for r in results]),
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    print("\n========== ORB + RANSAC Batch Summary ==========")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print(f"\n[Saved] {csv_path}")
    print(f"[Saved] {summary_path}")


def parse_args():
    parser = ArgumentParser(description="Batch ORB + RANSAC for UAV-satellite registration.")

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--pairs_txt", type=str, help="Txt/csv file: each line contains uav_path and sat_path.")
    input_group.add_argument("--uav_dir", type=str, help="UAV image directory. Must be used with --sat_dir.")

    parser.add_argument("--sat_dir", type=str, default=None, help="Satellite image directory.")
    parser.add_argument("--output_dir", default="result/orb_ransac", type=str)
    parser.add_argument("--num_pairs", default=20, type=int)

    # ORB 参数
    parser.add_argument("--nfeatures", default=5000, type=int, help="Maximum number of ORB keypoints.")
    parser.add_argument("--fast_threshold", default=20, type=int, help="ORB FAST threshold.")
    parser.add_argument("--ratio", default=0.75, type=float, help="Lowe ratio test threshold.")

    # 和 RoMa Top-K=2000 对齐，最多保留 2000 个匹配进入 RANSAC
    parser.add_argument("--max_matches", default=2000, type=int, help="Maximum matches sent to RANSAC.")

    # RANSAC 参数，和 RoMa 实验保持一致
    parser.add_argument("--ransac_thr", default=5.0, type=float, help="RANSAC reprojection threshold in pixels.")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.uav_dir is not None and args.sat_dir is None:
        raise ValueError("When using --uav_dir, you must also set --sat_dir.")

    if args.pairs_txt is not None:
        pairs = load_pairs_from_txt(args.pairs_txt, args.num_pairs)
    else:
        pairs = load_pairs_from_dirs(args.uav_dir, args.sat_dir, args.num_pairs)

    if len(pairs) == 0:
        raise RuntimeError("No image pairs found.")

    print(f"[Info] Will process {len(pairs)} image pairs.")
    print(f"[Info] nfeatures = {args.nfeatures}")
    print(f"[Info] max_matches = {args.max_matches}")
    print(f"[Info] ratio = {args.ratio}")
    print(f"[Info] ransac_thr = {args.ransac_thr}")

    results = []

    for i, (uav_path, sat_path) in enumerate(pairs, start=1):
        print(f"\n[{i}/{len(pairs)}] UAV: {uav_path}")
        print(f"[{i}/{len(pairs)}] SAT: {sat_path}")

        try:
            stats = orb_ransac_one_pair(
                pair_id=i,
                uav_path=uav_path,
                sat_path=sat_path,
                args=args,
            )
            results.append(stats)

            print(
                f"[Result] matches={stats['matched_points']} "
                f"inliers={stats['ransac_inliers']} "
                f"ratio={stats['ransac_inlier_ratio']:.4f} "
                f"reproj={stats['reproj_error_mean']:.4f}px"
            )

        except Exception as e:
            print(f"[Error] Failed on pair {i}: {e}")
            results.append({
                "pair_id": i,
                "uav_path": uav_path,
                "sat_path": sat_path,
                "success": 0,
                "num_keypoints_uav": 0,
                "num_keypoints_sat": 0,
                "matched_points": 0,
                "ransac_inliers": 0,
                "ransac_inlier_ratio": 0.0,
                "reproj_error_mean": math.nan,
                "reproj_error_median": math.nan,
                "reproj_error_max": math.nan,
            })

    save_results(args.output_dir, results)


if __name__ == "__main__":
    main()