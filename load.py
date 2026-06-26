from pathlib import Path
import shutil
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(description="Load dataset.")
    parser.add_argument("--src_root", required=True, type=str)
    parser.add_argument("--num_pairs", default=1000, type=int, help="Number of pictures")

    return parser.parse_args()

args = parse_args()

src_root = Path(args.src_root)
src_root_1 = src_root / "train"
src_root_2 = src_root / "test"
dst_root = Path("data")

# 创建目标目录
(dst_root / "uav").mkdir(parents=True, exist_ok=True)
(dst_root / "sat").mkdir(parents=True, exist_ok=True)

# 处理无人机图片
drone_dir_1 = src_root_1 / "drone"
drone_dir_2 = src_root_2 / "query_drone"

num_pairs = args.num_pairs
cur_uav = 0
for folder in sorted(drone_dir_1.iterdir()):
    if cur_uav == num_pairs:
        break
    cur_uav = cur_uav + 1

    first_img = next(p for p in folder.iterdir())   # 取排序后的第一张
    dst_name = f"{cur_uav:04d}.jpg"
    dst_path = dst_root / "uav" / dst_name
    shutil.copy2(first_img, dst_path)   # copy2 保留元数据

for folder in sorted(drone_dir_2.iterdir()):
    if cur_uav == num_pairs:
        break
    cur_uav = cur_uav + 1

    first_img = next(p for p in folder.iterdir())   # 取排序后的第一张
    dst_name = f"{cur_uav:04d}.jpg"
    dst_path = dst_root / "uav" / dst_name
    shutil.copy2(first_img, dst_path)   # copy2 保留元数据

# 处理卫星图片
sat_dir_1 = src_root_1 / "satellite"
sat_dir_2 = src_root_2 / "gallery_satellite"

cur_sat = 0
for folder in sorted(sat_dir_1.iterdir()):
    if cur_sat == num_pairs:
        break
    cur_sat = cur_sat + 1

    first_img = next(p for p in folder.iterdir())   # 取第一个
    dst_name = f"{cur_sat:04d}.jpg"
    dst_path = dst_root / "sat" / dst_name
    shutil.copy2(first_img, dst_path)

for folder in sorted(sat_dir_2.iterdir()):
    if cur_sat == num_pairs:
        break
    cur_sat = cur_sat + 1

    first_img = next(p for p in folder.iterdir())   # 取第一个
    dst_name = f"{cur_sat:04d}.jpg"
    dst_path = dst_root / "sat" / dst_name
    shutil.copy2(first_img, dst_path)