from pathlib import Path

VERSION = "1.7.0"

ANALYZER_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ANALYZER_DIR.parent
DOCUMENTATION_DIR = REPO_ROOT / "documentation"
MODEL_CANDIDATE_DIR = DOCUMENTATION_DIR / "model_candidates"
MODEL_CANDIDATE_WEIGHTS_DIR = MODEL_CANDIDATE_DIR / "weights"
MODELS_DIR = ANALYZER_DIR / "models"

SPECIESCLASSIFIER_PATH = MODELS_DIR / "model.onnx"
SPECIESCLASSIFIER_LABELS = MODELS_DIR / "labels.txt"
QUALITYCLASSIFIER_PATH = MODELS_DIR / "quality.onnx"
QUALITY_NORMALIZATION_DATA_PATH = MODELS_DIR / "quality_normalization_data.csv"
MASK_RCNN_WEIGHTS_PATH = MODELS_DIR / "mask_rcnn_resnet50_fpn_v2.pth"
# SAM-HQ: ViT-Tiny default (faster). For ViT-B quality, set path to sam_hq_vit_b.pth and SAM_HQ_MODEL_KEY = "vit_b".
SAM_HQ_WEIGHTS_PATH = MODELS_DIR / "sam_hq_vit_tiny.pth"
SAM_HQ_MODEL_KEY = "vit_tiny"  # segment_anything_hq.sam_model_registry

# SpeciesNet: bundled Kaggle-style folder (info.json + .pt + taxonomy). Passed as local model_name to speciesnet.ModelInfo.
SPECIESNET_MODEL_DIR = MODELS_DIR / "speciesnet"

# Runtime-selectable MegaDetector ONNX variants (all require .onnx.data sidecar files).
# mdv6-c/e stay under models/speciesnet; experimental candidates are archived under documentation/model_candidates/weights.
DEFAULT_DETECTOR_NAME = "mdv6-e"
DETECTOR_ONNX_PATHS = {
    "mdv6-c": SPECIESNET_MODEL_DIR / "mdv6-apa-rtdetr-c.onnx",
    "mdv6-e": SPECIESNET_MODEL_DIR / "mdv6-apa-rtdetr-e.onnx",
    "mdv5a": MODEL_CANDIDATE_WEIGHTS_DIR / "mdv5a.onnx",
    "mdv6-mit-yolov9-c": MODEL_CANDIDATE_WEIGHTS_DIR / "mdv6-mit-yolov9-c.onnx",
    "mdv6-mit-yolov9-e": MODEL_CANDIDATE_WEIGHTS_DIR / "mdv6-mit-yolov9-e.onnx",
}

# Backward-compatible alias used by existing call sites.
MDV6_ONNX_PATH = DETECTOR_ONNX_PATHS[DEFAULT_DETECTOR_NAME]

# SAM-HQ ViT-Tiny: split encoder + decoder ONNX files.
SAM_ENC_ONNX_PATH = SPECIESNET_MODEL_DIR / "sam_hq_vit_tiny_encoder.onnx"
SAM_DEC_ONNX_PATH = SPECIESNET_MODEL_DIR / "sam_hq_vit_tiny_decoder.onnx"

WILDLIFE_CATEGORIES = [
    "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "bird"
]

RAW_EXTENSIONS = [".cr2", ".cr3", ".nef", ".arw", ".dng", ".orf", ".rw2", ".pef", ".sr2"]
JPEG_EXTENSIONS = [".jpg", ".jpeg", ".png", '.tiff', '.tif']

DATABASE_NAME = "kestrel_database.csv"
METADATA_FILENAME = "kestrel_metadata.json"
SCENEDATA_FILENAME = "kestrel_scenedata.json"
KESTREL_DIR_NAME = ".kestrel"
LOG_FILENAME_PREFIX = "kestrel_error"
LOG_FILE_EXTENSION = "json"
