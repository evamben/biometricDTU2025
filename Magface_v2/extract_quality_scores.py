import numpy as np
import os

with open('Magface_v2/preprocessing_v1/LCC_FASD/feat_training2.list', 'r') as f:
    lines = f.readlines()

img_2_mag = {}
for line in lines:
    parts = line.strip().split(' ')
    imgname = parts[0].replace('\\', '/')
    feats = np.array([float(e) for e in parts[1:]], dtype=np.float32)
    mag = np.linalg.norm(feats)
    img_2_mag[imgname] = mag

imgnames = list(img_2_mag.keys())
mags = np.array([img_2_mag[name] for name in imgnames])

p33 = np.percentile(mags, 33)
p66 = np.percentile(mags, 66)

print(f"Low quality range: magnitude < {p33:.3f}")
print(f"Medium quality range: {p33:.3f} <= magnitude < {p66:.3f}")
print(f"High quality range: magnitude >= {p66:.3f}")

quality_ranges = {
    'low_quality': mags < p33,
    'medium_quality': (mags >= p33) & (mags < p66),
    'high_quality': mags >= p66
}

quality_splits = {}
for quality_name, mask in quality_ranges.items():
    quality_splits[quality_name] = [imgnames[i] for i in np.where(mask)[0]]
    print(f"{quality_name}: {len(quality_splits[quality_name])} images")

output_dir = 'quality_splits'
os.makedirs(output_dir, exist_ok=True)

for quality_name, img_list in quality_splits.items():
    filepath = os.path.join(output_dir, f"{quality_name}.txt")
    with open(filepath, 'w') as f:
        for imgname in img_list:
            f.write(f"{imgname}\n")
    print(f"Saved {len(img_list)} images to {filepath}")
