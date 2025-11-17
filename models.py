from huggingface_hub import hf_hub_download
import joblib
import numpy as np
import requests
import os
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# Config / Paths
# -----------------------------
# ===== BiLSTM Model  =====
model_path_1 = hf_hub_download("Ranasalh/DLmodel", filename="DL_BiLSTM_model.h5")
model_1 = keras.models.load_model(model_path_1)

vec_path_1 = hf_hub_download("Ranasalh/DLmodel", filename="tokenizer.pkl")
vectorizer_1 = joblib.load(vec_path_1)

max_length = 100

# ===== Logistic Regression Model  =====
repo_id_2 = "Ranasalh/prompt_safety_2_lr_more_f_and_new_data"
model_path_2 = hf_hub_download(repo_id=repo_id_2, filename="best_logistic_model.pkl")
vectorizer_path_2 = hf_hub_download(repo_id=repo_id_2, filename="tfidf_vectorizer.pkl")
model_2 = joblib.load(model_path_2)
vectorizer_2 = joblib.load(vectorizer_path_2)

# ===== OpenRouter =====n
os.environ.setdefault("OPEN_ROUTER", "")  # ضع مفتاحك في متغير البيئة OPEN_ROUTER
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "nvidia/nemotron-nano-12b-v2-vl:free"

# -----------------------------
# Calibration (no retrain) -- Option A
# -----------------------------
# Using a fixed temperature scalar to reduce BiLSTM overconfidence.
# This is a simple, safe approach that doesn't require a validation set
# or retraining. You can tweak TEMPERATURE if you want to adjust
# how aggressively the BiLSTM confidences are pulled down.

TEMPERATURE = 1.8  # قيمة مبدئية جيدة — زدها لتقليل الثقة أكثر، قللها لعكس ذلك
EPS = 1e-8

def apply_temperature_to_sigmoid_prob(prob, temperature=TEMPERATURE):
    """Convert sigmoid prob -> logit -> scale by temperature -> back to prob.

    This approximates temperature scaling for models that output a single
    sigmoid probability.
    """
    prob = float(np.clip(prob, EPS, 1.0 - EPS))
    logit = np.log(prob / (1.0 - prob))
    logit_scaled = logit / float(temperature)
    prob_scaled = 1.0 / (1.0 + np.exp(-logit_scaled))
    return float(prob_scaled)

def normalized_confidence_from_proba_vector(probas):
    """Return a confidence score in [0,1] based on the difference
    between class probabilities. This avoids trusting the raw 'max prob'
    which can be misleading across different model types.
    For binary: |p_pos - p_neg|. For multiclass this generalizes to
    max(p) - second_max(p).
    """
    probs = np.asarray(probas, dtype=float)
    if probs.ndim == 1 and probs.size == 2:
        return float(abs(probs[1] - probs[0]))
    # multiclass fallback: margin between top two
    sorted_idx = np.argsort(probs)
    top = probs[sorted_idx[-1]]
    second = probs[sorted_idx[-2]] if probs.size > 1 else 0.0
    return float(top - second)

# -----------------------------
# Prediction helpers
# -----------------------------
def predict_bilstm_raw_and_calibrated(text):
    """Return (pred_raw, prob_raw, pred_calibrated, prob_calibrated, conf_calibrated)

    - model_1 is expected to output a single sigmoid unit for positive class (UNSAFE).
    - raw prob is the model output; calibrated prob is after temperature scaling.
    - pred values: 0 -> SAFE, 1 -> UNSAFE
    """
    # prepare input (uses the same message formatting you used)
    message = f"Classify the following user prompt --> {text}, as 'SAFE' or 'UNSAFE'. Respond with a single word."
    seq = vectorizer_1.texts_to_sequences([message])
    X1_pad = pad_sequences(seq, maxlen=max_length, padding='post', truncating='post')

    # raw prediction: sigmoid output (float in [0,1])
    raw_out = model_1.predict(X1_pad)
    # support shapes: (1,1) or (1,) etc.
    raw_prob = float(np.squeeze(raw_out))
    raw_pred = 1 if raw_prob >= 0.5 else 0

    # calibrated using temperature scalar (no retraining)
    prob_cal = apply_temperature_to_sigmoid_prob(raw_prob, TEMPERATURE)
    pred_cal = 1 if prob_cal >= 0.5 else 0

    # confidence measure for BiLSTM: margin between positive/negative
    # for binary sigmoid we can derive negative prob = 1 - prob
    conf_cal = abs(prob_cal - (1.0 - prob_cal))  # equals |2*p-1|

    return {
        "raw_pred": raw_pred,
        "raw_prob": raw_prob,
        "pred_cal": pred_cal,
        "prob_cal": prob_cal,
        "conf_cal": conf_cal,
    }

def predict_logistic_raw_and_normalized(text):
    """Return (pred, prob_vector, conf_normalized)
    - model_2.predict_proba returns [p_safe, p_unsafe]
    - we compute normalized confidence as |p_pos - p_neg| (margin)
    """
    message = f"Classify the following user prompt --> {text}, as 'SAFE' or 'UNSAFE'. Respond with a single word."
    X2 = vectorizer_2.transform([message])
    probas2 = model_2.predict_proba(X2)[0]
    pred2 = int(np.argmax(probas2))
    # normalized confidence based on margin
    conf2 = normalized_confidence_from_proba_vector(probas2)
    return {
        "pred": pred2,
        "probas": probas2,
        "conf": conf2,
    }

# -----------------------------
# Ensemble decision logic
# -----------------------------
def classify_with_confidence(prompt):
    """Main entry: returns a dict with both models' decisions and a final fused decision.

    Behavior (Option A - no retrain):
      - BiLSTM: apply temperature scaling to sigmoid output
      - LR: use normalized margin confidence (p_pos - p_neg)
      - If both agree -> average calibrated confidences
      - If disagree -> choose the model with higher calibrated/normalized confidence
    """
    b = predict_bilstm_raw_and_calibrated(prompt)
    l = predict_logistic_raw_and_normalized(prompt)

    pred1 = b["pred_cal"]
    conf1 = b["conf_cal"]

    pred2 = l["pred"]
    conf2 = l["conf"]

    if pred1 == pred2:
        final_pred = int(pred1)
        final_conf = float((conf1 + conf2) / 2.0)
        match_type = "Match"
        winner = "Both (agreement)"
    else:
        # choose the model with higher confidence
        if conf1 > conf2:
            final_pred = int(pred1)
            final_conf = float(conf1)
            match_type = "Mismatch"
            winner = "BiLSTM (higher calibrated confidence)"
        else:
            final_pred = int(pred2)
            final_conf = float(conf2)
            match_type = "Mismatch"
            winner = "Logistic Regression (higher normalized confidence)"

    label = "SAFE" if final_pred == 0 else "UNSAFE"

    return {
        "model_1": {"label": "SAFE" if pred1 == 0 else "UNSAFE", "confidence": conf1, "prob_cal": b["prob_cal"], "raw_prob": b["raw_prob"]},
        "model_2": {"label": "SAFE" if pred2 == 0 else "UNSAFE", "confidence": conf2, "probas": l["probas"]},
        "final": {
            "label": label,
            "confidence": final_conf,
            "winner": winner,
            "match_type": match_type,
        },
    }

# -----------------------------
# OpenRouter helper (unchanged)
# -----------------------------
def get_openrouter_response(prompt):
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPEN_ROUTER')}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 200,
    }

    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"⚠️ Error getting LLM response: {e}"

# -----------------------------
# Public API used by the app
# -----------------------------
def process_prompt(prompt):
    result = classify_with_confidence(prompt)

    conf1 = result['model_1']['confidence']
    conf2 = result['model_2']['confidence']
    final_conf = result['final']['confidence']

    msg = (
        f"Model 1 (BiLSTM): {result['model_1']['label']} "
        f"(calibrated confidence {conf1:.3f}, prob {result['model_1']['prob_cal']:.3f}, raw {result['model_1']['raw_prob']:.3f})\n"
        f"Model 2 (Logistic): {result['model_2']['label']} (normalized confidence {conf2:.3f})\n"
        f"Match type: {result['final']['match_type']}\n\n"
        f"Final decision: {result['final']['label']} "
        f"(chosen from {result['final']['winner']}, confidence {final_conf:.3f})"
    )
    if result['final']['label'] == 'UNSAFE':
       # msg = "\n\n⚠️ The LLM refused to answer due to safety concerns, so the response was blocked."
       msg

    # Only call LLM if SAFE
    if result['final']['label'] == 'SAFE':
        ai_response = get_openrouter_response(prompt)

        # list of refusal patterns
        refusal_keywords = [
            "i cannot", "i can't", "cannot provide", "i am unable",
            "i’m unable", "i cannot help", "cannot assist", "i won't"
        ]

        # check if model refused
        if any(key in ai_response.lower() for key in refusal_keywords):
            msg += "\n\n⚠️ The LLM refused to answer due to safety concerns, so the response was blocked."
        else:
            msg = f"\n\nResponse:\n{ai_response}"

    return msg


