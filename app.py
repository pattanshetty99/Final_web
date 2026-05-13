import os
import json
import pickle
from pathlib import Path

import cv2
import dlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from flask import Flask, flash, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

# ============================================================
# Config
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
LANDMARK_MODEL = BASE_DIR / "shape_predictor_68_face_landmarks.dat"

IMG_SIZE = 224
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)
app.secret_key = "change-this-secret-key"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Load saved models and encoders
# ============================================================
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

model_prakriti = load_pickle(OUTPUT_DIR / "xgb_prakriti.pkl")
model_hairfall = load_pickle(OUTPUT_DIR / "xgb_hairfall.pkl")
enc_prakriti = load_pickle(OUTPUT_DIR / "le_prakriti.pkl")
enc_hairfall = load_pickle(OUTPUT_DIR / "le_hairfall.pkl")
feat_names_p = load_pickle(OUTPUT_DIR / "prakriti_feature_names.pkl")
feat_names_h = load_pickle(OUTPUT_DIR / "hf_feature_names.pkl")

# ============================================================
# Build the same ResNet-18 feature extractor used in notebook
# ============================================================
def build_hair_resnet(num_classes: int):
    model = models.resnet18(weights=None)
    for p in model.parameters():
        p.requires_grad = False
    for p in model.layer4.parameters():
        p.requires_grad = True

    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )
    return model

resnet_model = build_hair_resnet(len(enc_hairfall.classes_))
resnet_model.load_state_dict(
    torch.load(OUTPUT_DIR / "hair_resnet18_finetuned.pth", map_location=DEVICE)
)
resnet_model = resnet_model.to(DEVICE).eval()

feature_extractor = nn.Sequential(*list(resnet_model.children())[:-1]).to(DEVICE).eval()

feat_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ============================================================
# Dlib face detector + landmark model
# ============================================================
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(str(LANDMARK_MODEL))

# ============================================================
# Helpers
# ============================================================
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def encode_input(responses: dict, feature_names: list) -> np.ndarray:
    df = pd.DataFrame([responses])
    enc = pd.get_dummies(df)
    enc = enc.reindex(columns=feature_names, fill_value=0)
    return enc.values.astype(float)

def get_hair_only_mask(image_bgr: np.ndarray):
    """
    Returns (mask, bbox) where mask isolates hair region.
    bbox = (x1, y1, x2, y2)
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = image_bgr.shape[:2]
    faces = detector(gray, 1)

    mask = np.zeros((h, w), dtype=np.uint8)

    if len(faces) == 0:
        mask[: int(h * 0.40), :] = 255
        return mask, (0, 0, w, int(h * 0.40))

    shape = predictor(gray, faces[0])
    points = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)])

    eyebrow_pts = points[17:27]
    hairline_y = int(eyebrow_pts[:, 1].min()) + 10

    all_hull = cv2.convexHull(points)
    cv2.rectangle(mask, (0, 0), (w, hairline_y), 255, -1)

    face_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(face_mask, all_hull, 255)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(face_mask))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

    coords = cv2.findNonZero(mask)
    if coords is not None:
        bx, by, bw, bh = cv2.boundingRect(coords)
        pad = 8
        x1 = max(bx - pad, 0)
        y1 = max(by - pad, 0)
        x2 = min(bx + bw + pad, w)
        y2 = min(by + bh + pad, h)
    else:
        x1, y1, x2, y2 = 0, 0, w, int(h * 0.40)

    return mask, (x1, y1, x2, y2)

def preprocess_image(image_path: str) -> np.ndarray | None:
    """
    Load image -> hair mask -> crop to hair region -> resize -> RGB
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    try:
        hair_mask, (x1, y1, x2, y2) = get_hair_only_mask(img)
        result = np.full_like(img, 255, dtype=np.uint8)
        result[hair_mask == 255] = img[hair_mask == 255]
        cropped = result[y1:y2, x1:x2]
        if cropped.size == 0:
            cropped = result
        resized = cv2.resize(cropped, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    except Exception:
        resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

def extract_image_features(image_path: str) -> np.ndarray:
    img = preprocess_image(image_path)
    if img is None:
        return np.zeros((1, 512), dtype=np.float32)

    t = feat_tf(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat = feature_extractor(t).squeeze().cpu().numpy()
    return feat.reshape(1, -1)

def predict_prakriti(prakriti_responses: dict):
    x = encode_input(prakriti_responses, feat_names_p)
    pred = model_prakriti.predict(x)[0]
    proba = model_prakriti.predict_proba(x)[0]
    label = enc_prakriti.inverse_transform([pred])[0]
    probs = {
        enc_prakriti.classes_[i]: round(float(p) * 100, 1)
        for i, p in enumerate(proba)
    }
    return label, probs

def predict_hairfall(hair_responses: dict, image_path: str):
    img_feat = extract_image_features(image_path)
    tab_feat = encode_input(hair_responses, feat_names_h)
    x_fused = np.concatenate([img_feat, tab_feat], axis=1)

    pred = model_hairfall.predict(x_fused)[0]
    proba = model_hairfall.predict_proba(x_fused)[0]
    label = enc_hairfall.inverse_transform([pred])[0]
    probs = {
        enc_hairfall.classes_[i]: round(float(p) * 100, 1)
        for i, p in enumerate(proba)
    }
    return label, probs

# ============================================================
# Recommendations
# ============================================================
PRAKRITI_ADVICE = {
    "Vata": {
        "diet": [
            "Eat warm, cooked, slightly oily foods.",
            "Prefer ghee, sesame, soups, and warm milk if suitable.",
            "Reduce cold, raw, dry foods."
        ],
        "lifestyle": [
            "Keep a regular sleep schedule.",
            "Avoid overthinking and excessive screen time.",
            "Use calming breathing or light yoga."
        ],
        "hair_care": [
            "Use gentle oiling regularly.",
            "Avoid excessive heat styling.",
            "Choose nourishing shampoos and avoid over-washing."
        ],
    },
    "Pitta": {
        "diet": [
            "Prefer cooling foods like cucumber, coconut, and pomegranate.",
            "Avoid very spicy, fried, and sour foods.",
            "Stay hydrated through the day."
        ],
        "lifestyle": [
            "Avoid overheating and direct sun exposure.",
            "Practice cooling activities and stress reduction.",
            "Try to keep work and rest balanced."
        ],
        "hair_care": [
            "Use cooling oils like coconut oil if tolerated.",
            "Avoid hot water and harsh chemical treatments.",
            "Protect scalp from heat."
        ],
    },
    "Kapha": {
        "diet": [
            "Prefer light, warm, and less oily meals.",
            "Reduce heavy, sugary, and overly dairy-rich foods.",
            "Include spices like ginger and turmeric in moderation."
        ],
        "lifestyle": [
            "Exercise daily and keep a fixed routine.",
            "Avoid too much daytime sleeping.",
            "Stay active to reduce sluggishness."
        ],
        "hair_care": [
            "Wash scalp regularly if it tends to be oily.",
            "Avoid excessive product buildup.",
            "Use lightweight hair care products."
        ],
    },
}

HAIRFALL_ADVICE = {
    "Vata": [
        "Hair looks dry, brittle, and prone to breakage.",
        "Focus on moisture, oiling, and gentle handling.",
    ],
    "Pitta": [
        "Hair/scalp may show heat, irritation, or inflammation.",
        "Use cooling care and avoid overheating the scalp.",
    ],
    "Kapha": [
        "Hair/scalp may be oily or feel heavy with buildup.",
        "Use cleansing, lightweight care and regular wash routines.",
    ],
}

def get_recommendations(prakriti: str, hairfall: str):
    p = PRAKRITI_ADVICE.get(prakriti, {})
    h = HAIRFALL_ADVICE.get(hairfall, [])

    summary = f"Predicted Prakriti: {prakriti} | Predicted Hairfall Type: {hairfall}"

    return {
        "summary": summary,
        "diet": p.get("diet", []),
        "lifestyle": p.get("lifestyle", []),
        "hair_care": p.get("hair_care", []),
        "hairfall_points": h,
    }

# ============================================================
# Routes
# ============================================================
@app.route("/", methods=["GET"])
def index():
    demo_prakriti = {
        "how_would_you_describe_your_overall_body_build_and_muscle_development": "Thin, lean, low muscle mass",
        "how_would_you_describe_your_body_frame_or_chest_width": "Narrow / slim frame",
        "what_best_describes_your_natural_skin_complexion_or_color": "Dusky / wheatish",
        "what_best_describes_the_condition_of_your_nails": "Dry, rough, brittle, easily break",
        "how_sensitive_is_your_skin_to_environment_cosmetics_or_weather": "Very sensitive, easily irritated",
        "how_would_you_describe_your_appetite": "Irregular or low appetite",
        "how_would_you_describe_your_digestion_after_meals": "Moderate digestion",
        "how_would_you_describe_your_metabolism_and_weight_change": "Slow metabolism, difficult to gain weight",
        "how_would_you_describe_your_sleep_pattern": "Light, disturbed, short sleep",
        "how_would_you_respond_to_stress_or_pressure": "Worry, anxiety, overthinking",
    }

    demo_hair = {
        "hair_type": "Dry and frizzy",
        "split_ends": "Yes",
        "breakage": "Yes",
        "thinning": "Diffuse (all over scalp)",
        "greying": "Yes",
        "thickness": "Thin",
        "growth": "Slow",
        "severity": "Moderate",
        "scalp": "Dry and tight",
        "dandruff": "Dry flakes",
        "heat": "No",
        "irritation": "Yes",
        "dry_skin": "Yes",
        "anxiety": "Yes",
        "heat_sensitive": "No",
        "oily_skin": "No",
    }

    return render_template(
        "index.html",
        demo_prakriti=json.dumps(demo_prakriti, indent=2),
        demo_hair=json.dumps(demo_hair, indent=2),
    )

@app.route("/predict", methods=["POST"])
def predict():
    prakriti_json = request.form.get("prakriti_json", "").strip()
    hair_json = request.form.get("hair_json", "").strip()
    image_file = request.files.get("image")

    if not prakriti_json or not hair_json:
        flash("Please paste both JSON blocks for Prakriti and Hairfall.", "danger")
        return redirect(url_for("index"))

    if image_file is None or image_file.filename == "":
        flash("Please upload a face image.", "danger")
        return redirect(url_for("index"))

    if not allowed_file(image_file.filename):
        flash("Only PNG, JPG, JPEG, and WEBP images are allowed.", "danger")
        return redirect(url_for("index"))

    try:
        prakriti_responses = json.loads(prakriti_json)
        hair_responses = json.loads(hair_json)

        if not isinstance(prakriti_responses, dict) or not isinstance(hair_responses, dict):
            raise ValueError("Both inputs must be JSON objects.")

    except Exception as e:
        flash(f"Invalid JSON input: {e}", "danger")
        return redirect(url_for("index"))

    filename = secure_filename(image_file.filename)
    save_path = UPLOAD_DIR / filename
    image_file.save(save_path)

    try:
        predicted_prakriti, prakriti_probs = predict_prakriti(prakriti_responses)
        predicted_hairfall, hairfall_probs = predict_hairfall(hair_responses, str(save_path))
        recs = get_recommendations(predicted_prakriti, predicted_hairfall)

        return render_template(
            "result.html",
            image_url=url_for("static", filename=f"uploads/{filename}"),
            prakriti=predicted_prakriti,
            hairfall=predicted_hairfall,
            prakriti_probs=sorted(prakriti_probs.items(), key=lambda x: -x[1]),
            hairfall_probs=sorted(hairfall_probs.items(), key=lambda x: -x[1]),
            recs=recs,
        )

    except Exception as e:
        flash(f"Prediction failed: {e}", "danger")
        return redirect(url_for("index"))

@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    app.run(debug=True)
