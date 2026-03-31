import os, json, numpy as np
from analyzer.kestrel_analyzer.image_utils import read_image_for_pipeline
from analyzer.kestrel_analyzer.pipeline import AnalysisPipeline

FOLDER = r"C:\Data\Code\Project Kestrel\Kestrel Development\ProjectKestrel\test_sets\failure_cases\failure_001_meadowlark_blur"


def stats(arr):
    f = arr.astype(np.float32) / 255.0
    lum = 0.2126 * f[..., 0] + 0.7152 * f[..., 1] + 0.0722 * f[..., 2]
    any_clip = np.any(f >= 0.995, axis=2)
    all_clip = np.all(f >= 0.995, axis=2)
    return {
        "lum_p95": float(np.percentile(lum, 95)),
        "lum_p98": float(np.percentile(lum, 98)),
        "lum_p99": float(np.percentile(lum, 99)),
        "lum_clip985": float(np.mean(lum >= 0.985)),
        "lum_clip965": float(np.mean(lum >= 0.965)),
        "chan_any_clip995": float(np.mean(any_clip)),
        "chan_all_clip995": float(np.mean(all_clip)),
        "chan_p99_9": [float(np.percentile(f[..., c], 99.9)) for c in range(3)],
    }

# find candidate bright frames
rows = []
for n in sorted(os.listdir(FOLDER)):
    if not n.lower().endswith('.cr3'):
        continue
    p = os.path.join(FOLDER, n)
    img, raw = read_image_for_pipeline(p)
    if img is None:
        continue
    st = stats(img)
    rows.append((n, st))
    if raw is not None:
        raw.close()

# Rank by strongest highlight indicators
ranked = sorted(rows, key=lambda x: (x[1]["lum_p98"], x[1]["chan_any_clip995"]), reverse=True)
print(f"total raw files parsed: {len(ranked)}")
print("top 8 by brightness/clipping:")
for n, st in ranked[:8]:
    print(n, json.dumps(st))

print("\nprofile results on top 6:")
for n, base_st in ranked[:6]:
    p = os.path.join(FOLDER, n)
    img, raw = read_image_for_pipeline(p)
    mask = np.ones(img.shape[:2], dtype=bool)
    print(f"\nFILE {n}")
    print("  base", json.dumps(base_st))
    for prof in ["lenient", "normal", "aggressive"]:
        s0 = AnalysisPipeline._compute_exposure_stops(img, mask, prof)
        s = AnalysisPipeline._refine_exposure_stops(img, mask, s0, prof, raw_obj=raw)
        corr = AnalysisPipeline._apply_exposure_correction(img, s, raw_obj=raw)
        st = stats(corr)
        print(
            f"  {prof:10s} stops={s:+.4f} "
            f"lum_p98={st['lum_p98']:.4f} lum_clip985={st['lum_clip985']:.5f} "
            f"chan_any_clip995={st['chan_any_clip995']:.5f}"
        )
    if raw is not None:
        raw.close()
