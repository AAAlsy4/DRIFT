# DRIFT
基于稠密特征匹配的遥感图像配准方法研究。DRIFT(Dense RoMa Image Feature Transformation)

![RoMa](./figs/RoMa.png) 

RoMa

![matches_all](./figs/matches_all.png) 

matches_all

![matches_inliers](./figs/matches_inliers.png) 

matches_inliers

![warped_overlay](./figs/warped_overlay.png) 

warped_overlay

![warped_uav_to_sat](./figs/warped_uav_to_sat.png) 

warped_uav_to_sat

```bash
conda create -n roma python=3.10

conda activate roma

pip install -e .
```

```bash
python demo/batch_20_pairs.py --uav_dir data/uav --sat_dir data/sat

python demo/batch_sift_ransac.py --uav_dir data/uav --sat_dir data/sat

python demo/batch_orb_ransac.py --uav_dir data/uav --sat_dir data/sat
```

```bash
python demo/demo_match.py --uav_path uav.jpg --sat_path sat.jpg
```