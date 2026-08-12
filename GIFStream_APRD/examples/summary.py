import os
import json
from collections import defaultdict
import zipfile
import argparse

from gsplat.compression.ap_gifstream import deterministic_zip_directory

parser = argparse.ArgumentParser(description='Count zip and json info under scene_xx directories')
parser.add_argument('--root_dir', type=str, default='.', help='Root directory, default is current directory')
args = parser.parse_args()

root_dir = args.root_dir

# Top-level scene folders to traverse
scene_folders = [
    d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith('.') and not d.startswith('__')
]

result = {}

for scene in scene_folders:
    scene_path = os.path.join(root_dir, scene)
    if not os.path.isdir(scene_path):
        continue
    # Statistics: scene-rxx -> list of dicts
    rxx_data = defaultdict(list)
    for gop in os.listdir(scene_path):
        gop_path = os.path.join(scene_path, gop)
        if not os.path.isdir(gop_path) or not gop.startswith('GOP_'):
            continue
        for rxx in os.listdir(gop_path):
            rxx_path = os.path.join(gop_path, rxx)
            if not os.path.isdir(rxx_path):
                continue
            # compression folder
            compression_dir = os.path.join(rxx_path, 'compression')
            zip_path = os.path.join(rxx_path, 'compression.zip')
            zip_audit = None
            if os.path.isdir(compression_dir):
                # If zip exists, delete it before compressing again
                if os.path.exists(zip_path):
                    os.remove(zip_path)
                zip_audit = deterministic_zip_directory(compression_dir, zip_path)
            # Count zip size
            if os.path.exists(zip_path):
                size_mb = os.path.getsize(zip_path) / (1024 * 1024)
                size_bytes = os.path.getsize(zip_path)
            else:
                size_mb = 0.0
                size_bytes = 0
            stats_path = os.path.join(rxx_path, 'stats', 'compress_step29999.json')
            if os.path.exists(stats_path):
                try:
                    with open(stats_path, 'r') as f:
                        data = json.load(f)
                    data['size'] = size_mb
                    data['size_bytes'] = size_bytes
                    data['size_mb_decimal'] = size_bytes / 1_000_000
                    if zip_audit is not None:
                        data['container_sha256'] = zip_audit['sha256']
                    rxx_data[rxx].append(data)
                except Exception as e:
                    print(f"Failed to read: {stats_path}, error: {e}")
    # Merge same-named rxx
    for rxx, items in rxx_data.items():
        merged = {}
        if not items:
            continue
        keys = items[0].keys()
        for k in keys:
            if k in {'size', 'size_bytes', 'size_mb_decimal'}:
                # Sum for size field
                merged[k] = sum(float(item[k]) for item in items)
            else:
                # Average for numeric fields
                try:
                    merged[k] = sum(float(item[k]) for item in items) / len(items)
                except Exception:
                    merged[k] = items[0][k]
        result[f'{scene}-{rxx}'] = merged

# Calculate summary for each rxx type
if result:
    # Collect all rxx suffixes
    rxx_groups = defaultdict(list)
    for k, v in result.items():
        if '-' in k and k.split('-')[-1].startswith('r'):
            rxx_suffix = k.split('-')[-1]
            rxx_groups[rxx_suffix].append(v)
    # Average for each rxx suffix
    for rxx_suffix, items in rxx_groups.items():
        summary = {}
        if not items:
            continue
        keys = list(items[0].keys())
        for k in keys:
            try:
                summary[k] = sum(float(item[k]) for item in items) / len(items)
            except Exception:
                summary[k] = items[0][k]
        result[f'summary-{rxx_suffix}'] = summary

import pprint
pprint.pprint(result)

# Save as summary.json
summary_path = os.path.join(root_dir, 'summary.json')
with open(summary_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print(f'Statistics saved to: {summary_path}')
