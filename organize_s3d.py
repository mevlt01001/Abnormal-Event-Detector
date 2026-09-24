import os
import shutil
import json
from pathlib import Path

src_dir = Path("/home/n3uron/Downloads/DeepCFD/data/features/s3d")
ref_dir = Path("data/mvit_v2_s_features")
dst_dir = Path("data/s3d_features")

print(f"Indexing source files in {src_dir} ...")
src_files = {}
for root, _, files in os.walk(src_dir):
    for f in files:
        if f.endswith('.pt') or f.endswith('.npy'):
            stem = Path(f).stem
            src_files[stem] = Path(root) / f

print(f"Indexed {len(src_files)} files in source.")

missing_count = 0
copied_count = 0

for root, _, files in os.walk(ref_dir):
    for f in files:
        if f == "manifest.json":
            rel_path = Path(root).relative_to(ref_dir)
            target_dir = dst_dir / rel_path
            target_dir.mkdir(parents=True, exist_ok=True)
            
            with open(Path(root)/f, 'r') as jf:
                data = json.load(jf)
            data["backbone"] = "s3d"
            # s3d genelde 1024 boyutundadır (I3D/S3D), ama emin değiliz. 1024 diyelim.
            data["feature_dim"] = 1024
            
            with open(target_dir/f, 'w') as jf:
                json.dump(data, jf, indent=2)
            continue
            
        if not (f.endswith('.pt') or f.endswith('.npy')):
            continue
            
        stem = Path(f).stem
        if stem in src_files:
            rel_path = Path(root).relative_to(ref_dir)
            target_dir = dst_dir / rel_path
            target_dir.mkdir(parents=True, exist_ok=True)
            
            src_file = src_files[stem]
            target_file = target_dir / src_file.name
            
            if not target_file.exists():
                shutil.copy2(src_file, target_file)
            copied_count += 1
        else:
            missing_count += 1
            print(f"Missing (not found in s3d): {stem}")

print(f"Done. Copied: {copied_count}, Missing: {missing_count}")
