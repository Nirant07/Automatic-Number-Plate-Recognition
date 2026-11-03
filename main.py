
import cv2
import re
import json
import shutil
import subprocess
import sys

from pathlib import Path
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

ORIGINAL_DIR = BASE_DIR / "output" / "plates_original"
PLATE_DIR = BASE_DIR / "output" / "plates"
ANNOTATED_DIR = BASE_DIR / "output" / "annotated"

PADDLE_DIR = BASE_DIR / "PaddleOCR"
OCR_SCRIPT = PADDLE_DIR / "test.py"
OCR_WEIGHTS = PADDLE_DIR / "model.safetensors"

OCR_JSON = BASE_DIR / "output" / "ocr_results.json"

CONFIDENCE_THRESHOLD = 0.25

# Crop padding
HORIZONTAL_PADDING = 0.08
VERTICAL_PADDING = 0.15

for folder in [PLATE_DIR, ANNOTATED_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# ==========================================
# CLEAN PREVIOUS OUTPUT
# ==========================================

for folder in [PLATE_DIR, ANNOTATED_DIR]:
    for file in folder.iterdir():
        if file.is_file():
            file.unlink()

# ==========================================
# CHECK INPUT
# ==========================================

if not ORIGINAL_DIR.exists():
    raise FileNotFoundError(ORIGINAL_DIR)

images = [
    p for p in ORIGINAL_DIR.iterdir()
    if p.suffix.lower() in
    [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
]

if not images:
    raise RuntimeError("No vehicle images found.")

print(f"\nFound {len(images)} original images.")

# ==========================================
# LOAD YOLO
# ==========================================

print("\nLoading YOLO detector...")

weights = hf_hub_download(
    repo_id="Koushim/yolov8-license-plate-detection",
    filename="best.pt"
)

detector = YOLO(weights)

# ==========================================
# PROCESS IMAGES
# ==========================================

total_plates = 0
image_results = {}

for image_path in images:

    print(f"\nProcessing: {image_path.name}")

    image = cv2.imread(str(image_path))

    if image is None:
        print("Unable to read image. Skipping.")
        continue

    height, width = image.shape[:2]
    annotated = image.copy()

    results = detector.predict(
        source=image,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False
    )

    detected_in_image = 0

    for result in results:

        for box in result.boxes:

            x1, y1, x2, y2 = map(
                int, box.xyxy[0].tolist()
            )

            confidence = float(box.conf[0])

            # Add padding around the plate
            pad_x = int((x2 - x1) * HORIZONTAL_PADDING)
            pad_y = int((y2 - y1) * VERTICAL_PADDING)

            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(width, x2 + pad_x)
            y2 = min(height, y2 + pad_y)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = image[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            total_plates += 1
            detected_in_image += 1

            plate_name = f"plate_{total_plates}.jpg"

            crop_path = PLATE_DIR / plate_name

            cv2.imwrite(
                str(crop_path),
                crop,
                [cv2.IMWRITE_JPEG_QUALITY, 100]
            )

            # Draw detection box
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

            cv2.putText(
                annotated,
                f"{plate_name} ({confidence:.2f})",
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            print(
                f"  Detected {plate_name} "
                f"(confidence: {confidence:.3f})"
            )

    image_results[image_path.name] = detected_in_image

    annotated_path = ANNOTATED_DIR / image_path.name

    cv2.imwrite(str(annotated_path), annotated)

    print(f"  Plates found: {detected_in_image}")

# ==========================================
# RUN AWIROS OCR
# ==========================================

if total_plates == 0:

    print("\nNo plates detected by YOLO.")
    print("Inspect output/annotated to check detection.")

else:

    print(f"\nTotal detected plates: {total_plates}")
    print("\nStarting Awiros OCR...")

    command = [
        sys.executable,
        str(OCR_SCRIPT),
        "--image_path", str(PLATE_DIR),
        "--weights", str(OCR_WEIGHTS),
        "--device", "cpu",
        "--output_json", str(OCR_JSON)
    ]

    process = subprocess.run(
        command,
        cwd=str(PADDLE_DIR),
        capture_output=True,
        text=True
    )

    print("\n========== OCR OUTPUT ==========")
    print(process.stdout)

    if process.returncode != 0:
        print("\nOCR ERROR:")
        print(process.stderr)
        raise RuntimeError("OCR inference failed.")

    # ======================================
    # READ OCR JSON
    # ======================================

    if OCR_JSON.exists():

        with open(OCR_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)

        with open(
            BASE_DIR / "output" / "ocr_raw.txt",
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                data,
                f,
                indent=4,
                ensure_ascii=False
            )

        print("\n========== RAW OCR RESULTS ==========")
        print(json.dumps(data, indent=2))

    else:
        print("\nOCR JSON was not generated.")

# ==========================================
# SUMMARY
# ==========================================

print("\n====================================")
print("             SUMMARY")
print("====================================")

print(f"Original images: {len(images)}")
print(f"Total plates detected: {total_plates}")

for name, count in image_results.items():
    print(f"{name}: {count} plate(s)")

print("\nAnnotated images:", ANNOTATED_DIR)
print("Cropped plates:", PLATE_DIR)
print("OCR results:", OCR_JSON)

print("\nProcessing completed.")
