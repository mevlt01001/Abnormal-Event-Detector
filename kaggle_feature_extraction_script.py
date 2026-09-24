# %% [code]
!pip install -q decord
!git clone https://github.com/mevlt01001/Abnormal-Event-Detector.git

# %% [code]
import os 
os.chdir("Abnormal-Event-Detector")

# %% [code]
# =========================================================================
# ⚙️ PARALEL KAGGLE HESAP AYARI
# 1. Hesap için: PART = 1
# 2. Hesap için: PART = 2
# 3. Hesap için: PART = 3
# 4. Hesap için: PART = 4
# =========================================================================
PART = 1
TOTAL_PARTS = 4

import os
import math
from collections import defaultdict

# ----------------- Veri Seti Kök Dizinleri -----------------
# UCF-Crime
ucf_abuse_videos_root = "/kaggle/input/datasets/minmints/ufc-crime-full-dataset/Anomaly-Videos-Part-1/Anomaly-Videos-Part-1/Abuse"
ucf_arrest_videos_root = "/kaggle/input/datasets/minmints/ufc-crime-full-dataset/Anomaly-Videos-Part-1/Anomaly-Videos-Part-1/Arrest"
ucf_arson_videos_root = "/kaggle/input/datasets/minmints/ufc-crime-full-dataset/Anomaly-Videos-Part-1/Anomaly-Videos-Part-1/Arson"
ucf_assault_videos_root = "/kaggle/input/datasets/minmints/ufc-crime-full-dataset/Anomaly-Videos-Part-1/Anomaly-Videos-Part-1/Assault"
ucf_burglary_videos_root = "/kaggle/input/datasets/minmints/ufc-crime-full-dataset/Anomaly-Videos-Part-2/Anomaly-Videos-Part-2/Burglary"
ucf_explosion_videos_root = "/kaggle/input/datasets/minmints/ufc-crime-full-dataset/Anomaly-Videos-Part-2/Anomaly-Videos-Part-2/Explosion"
ucf_fighting_videos_root = "/kaggle/input/datasets/minmints/ufc-crime-full-dataset/Anomaly-Videos-Part-2/Anomaly-Videos-Part-2/Fighting"
ucf_road_accidents_videos_root = "/kaggle/input/datasets/minmints/ufc-crime-full-dataset/Anomaly-Videos-Part-3/Anomaly-Videos-Part-3/RoadAccidents"
ucf_robbery_videos_root = "/kaggle/input/datasets/minmints/ufc-crime-full-dataset/Anomaly-Videos-Part-3/Anomaly-Videos-Part-3/Robbery"
ucf_shooting_videos_root = "/kaggle/input/datasets/minmints/ufc-crime-full-dataset/Anomaly-Videos-Part-3/Anomaly-Videos-Part-3/Shooting"
ucf_shoplifting_videos_root = "/kaggle/input/datasets/minmints/ufc-crime-full-dataset/Anomaly-Videos-Part-4/Anomaly-Videos-Part-4/Shoplifting"
ucf_stealing_videos_root = "/kaggle/input/datasets/minmints/ufc-crime-full-dataset/Anomaly-Videos-Part-4/Anomaly-Videos-Part-4/Stealing"
ucf_vandalism_videos_root = "/kaggle/input/datasets/minmints/ufc-crime-full-dataset/Anomaly-Videos-Part-4/Anomaly-Videos-Part-4/Vandalism"
ucf_normal_videos_1_root = "/kaggle/input/datasets/minmints/ufc-crime-full-dataset/Training-Normal-Videos-Part-1/Training-Normal-Videos-Part-1"
ucf_normal_videos_2_root = "/kaggle/input/datasets/minmints/ufc-crime-full-dataset/Training-Normal-Videos-Part-2/Training-Normal-Videos-Part-2"
ucf_normal_test_videos_root = "/kaggle/input/datasets/minmints/ufc-crime-full-dataset/Testing_Normal_Videos/Testing_Normal_Videos_Anomaly"

# XD-Violence
xdv_abuse_videos_root = "/kaggle/input/datasets/bypktt/xd-violence/train/Abuse"
xdv_road_accidents_videos_root = "/kaggle/input/datasets/bypktt/xd-violence/train/CarAccident"
xdv_explosion_videos_root = "/kaggle/input/datasets/bypktt/xd-violence/train/Explosion"
xdv_fighting_videos_root = "/kaggle/input/datasets/bypktt/xd-violence/train/Fighting"
xdv_normal_videos_root = "/kaggle/input/datasets/bypktt/xd-violence/train/Normal"
xdv_riot_videos_root = "/kaggle/input/datasets/bypktt/xd-violence/train/Riot"
xdv_shooting_videos_root = "/kaggle/input/datasets/bypktt/xd-violence/train/Shooting"
xdv_test_abuse_videos_root = "/kaggle/input/datasets/bypktt/xd-violence/test/Abuse"
xdv_test_road_accidents_videos_root = "/kaggle/input/datasets/bypktt/xd-violence/test/CarAccident"
xdv_test_explosion_videos_root = "/kaggle/input/datasets/bypktt/xd-violence/test/Explosion"
xdv_test_fighting_videos_root = "/kaggle/input/datasets/bypktt/xd-violence/test/Fighting"
xdv_test_normal_videos_root = "/kaggle/input/datasets/bypktt/xd-violence/test/Normal"
xdv_test_riot_videos_root = "/kaggle/input/datasets/bypktt/xd-violence/test/Riot"
xdv_test_shooting_videos_root = "/kaggle/input/datasets/bypktt/xd-violence/test/Shooting"

video_roots = [
    ucf_abuse_videos_root, ucf_arrest_videos_root, ucf_arson_videos_root,
    ucf_assault_videos_root, ucf_burglary_videos_root, ucf_explosion_videos_root,
    ucf_fighting_videos_root, ucf_road_accidents_videos_root, ucf_robbery_videos_root,
    ucf_shooting_videos_root, ucf_shoplifting_videos_root, ucf_stealing_videos_root,
    ucf_vandalism_videos_root, ucf_normal_videos_1_root, ucf_normal_videos_2_root,
    xdv_abuse_videos_root, xdv_road_accidents_videos_root, xdv_explosion_videos_root,
    xdv_fighting_videos_root, xdv_normal_videos_root, xdv_riot_videos_root,
    xdv_shooting_videos_root, ucf_normal_test_videos_root, xdv_test_abuse_videos_root,
    xdv_test_road_accidents_videos_root, xdv_test_explosion_videos_root, xdv_test_fighting_videos_root,
    xdv_test_normal_videos_root, xdv_test_riot_videos_root, xdv_test_shooting_videos_root
]

video_names = [
    "ucf_abuse_videos_root", "ucf_arrest_videos_root", "ucf_arson_videos_root",
    "ucf_assault_videos_root", "ucf_burglary_videos_root", "ucf_explosion_videos_root",
    "ucf_fighting_videos_root", "ucf_road_accidents_videos_root", "ucf_robbery_videos_root",
    "ucf_shooting_videos_root", "ucf_shoplifting_videos_root", "ucf_stealing_videos_root",
    "ucf_vandalism_videos_root", "ucf_normal_videos_1_root", "ucf_normal_videos_2_root",
    "xdv_abuse_videos_root", "xdv_road_accidents_videos_root", "xdv_explosion_videos_root",
    "xdv_fighting_videos_root", "xdv_normal_videos_root", "xdv_riot_videos_root",
    "xdv_shooting_videos_root", "ucf_normal_test_videos_root", "xdv_test_abuse_videos_root",
    "xdv_test_road_accidents_videos_root", "xdv_test_explosion_videos_root", "xdv_test_fighting_videos_root",
    "xdv_test_normal_videos_root", "xdv_test_riot_videos_root", "xdv_test_shooting_videos_root"
]

# ----------------- 1. Tüm Videoları Belirle -----------------
valid_exts = {".mp4", ".avi", ".mkv", ".mov"}
all_items = []

for v_root, v_class in zip(video_roots, video_names):
    if not os.path.exists(v_root):
        print(f"⚠️ Klasör bulunamadı: {v_root}")
        continue
    files = sorted([
        os.path.join(v_root, f) for f in os.listdir(v_root)
        if os.path.splitext(f)[1].lower() in valid_exts
    ])
    for f in files:
        all_items.append((v_class, f))

total_videos = len(all_items)
chunk_size = math.ceil(total_videos / TOTAL_PARTS)
start_idx = (PART - 1) * chunk_size
end_idx = min(PART * chunk_size, total_videos)

my_part_items = all_items[start_idx:end_idx]

print(f"==================================================")
print(f"📦 KAGGLE HESAP: PART {PART} / {TOTAL_PARTS}")
print(f"Toplam Video Sayısı:  {total_videos}")
print(f"Bu Hesaba Düşen:      {len(my_part_items)} video ({start_idx} → {end_idx})")
print(f"==================================================")

# ----------------- 2. Bu Part İçin Sınıf Haritasını Kur -----------------
part_dict = defaultdict(list)
for c_name, v_path in my_part_items:
    part_dict[c_name].append(v_path)

part_class_names = list(part_dict.keys())
part_videos_map = [part_dict[k] for k in part_class_names]

# Sınıf bazlı dağılımı göster:
for c_name in part_class_names:
    print(f"  • {c_name:<30}: {len(part_dict[c_name])} video")

# %% [code]
# ----------------- 3. Öznitelik Çıkarımını Başlat -----------------
import sys
if os.path.isdir("Abnormal-Event-Detector"):
    os.chdir("Abnormal-Event-Detector")
sys.path.insert(0, ".")

from scripts.kaggle_extract import extract_features_from_class_map

result = extract_features_from_class_map(
    class_video_map=part_videos_map,
    class_names=part_class_names,
    output_dir="features",
    model_names="MViT_v2_S",
    num_segments=32,
    clip_size=16,
    fps=20.0,
    overlap_ratio=0.50,  # Kısa segmentlerde %50 overlap korur
    max_clips_per_segment=24,  # Uzun segmentlerde sıfır overlap ile eşit dağılır
    batch_size=8,
)
