"""Plain-language descriptions and measured results for every model EmotiSense uses.

The numbers are from real runs: the timeline models are 5-fold cross-validated on held-out IEMOCAP sessions, and the
classifier on held-out sessions as well. Nothing here is estimated.
"""
from __future__ import annotations

TIMELINE = {
    "bilstm": {
        "name": "BiLSTM",
        "tagline": "Reads the recording in both directions",
        "description": (
            "A bidirectional recurrent network. For each passage it looks at the ones before and after, so a quiet "
            "sentence that follows a sharp one is judged in that context. It is the default and the most accurate "
            "timeline model here."
        ),
        "strengths": [
            "Best at following how a feeling builds and eases (CCC 0.51)",
            "Lowest average error (MAE 0.18)",
            "Small and fast: runs comfortably on a laptop",
        ],
        "limits": [
            "Needs the whole recording first, so it cannot score live audio",
            "Frustration is still the hardest feeling to track (CCC 0.36)",
        ],
        "stats": {
            "ccc": {"mean": 0.5129, "std": 0.0358}, "rmse": 0.3469, "mae": 0.1834, "alert_f1": 0.888, "trend_accuracy": 0.676,
            "per_emotion_ccc": {"anger": 0.535, "frustration": 0.359, "happiness": 0.649, "sadness": 0.571, "neutral": 0.451},
        },
    },
    "transformer": {
        "name": "Transformer",
        "tagline": "Every passage can look at every other one",
        "description": (
            "An attention-based model: each passage weighs all the others at once. It is powerful in general, but it "
            "needs a lot of examples, and here it was trained on only 151 dialogues."
        ),
        "strengths": [
            "Strong on anger and joy (CCC 0.54 and 0.62)",
            "Reads a long recording in a single pass",
        ],
        "limits": [
            "With so little training data it did not beat the simple no-context model overall",
            "Weakest at telling whether tension is rising or easing (65%)",
        ],
        "stats": {
            "ccc": {"mean": 0.4665, "std": 0.0426}, "rmse": 0.3676, "mae": 0.1908, "alert_f1": 0.860, "trend_accuracy": 0.652,
            "per_emotion_ccc": {"anger": 0.538, "frustration": 0.330, "happiness": 0.617, "sadness": 0.484, "neutral": 0.364},
        },
    },
    "mlp": {
        "name": "No-context baseline (MLP)",
        "tagline": "Judges every passage entirely on its own",
        "description": (
            "A tiny network that sees one passage at a time and knows nothing about what came before or after. It is "
            "here as a yardstick: the gap between it and the BiLSTM shows how much reading context really helps."
        ),
        "strengths": [
            "Smallest and fastest model (0.33 million parameters)",
            "On par with the others for alerts (F1 0.89) and for telling rising from easing (68%)",
        ],
        "limits": [
            "Cannot tell a brief spike from a sustained build",
            "Lowest overall tracking of the emotional arc (CCC 0.46)",
        ],
        "stats": {
            "ccc": {"mean": 0.4636, "std": 0.0217}, "rmse": 0.3594, "mae": 0.2042, "alert_f1": 0.894, "trend_accuracy": 0.684,
            "per_emotion_ccc": {"anger": 0.516, "frustration": 0.322, "happiness": 0.534, "sadness": 0.558, "neutral": 0.388},
        },
    },
    "lstm": {
        "name": "LSTM (one-way)",
        "tagline": "Reads only what has already happened",
        "description": "A causal variant meant for live monitoring. It is supported by the code but has not been trained here.",
        "strengths": ["Could run on live audio, because it never looks ahead"],
        "limits": ["Not trained in this project, so there is no measured accuracy"],
        "stats": None,
    },
    "transformer_causal": {
        "name": "Transformer (one-way)",
        "tagline": "Attention that only looks backwards",
        "description": "A causal variant meant for live monitoring. It is supported by the code but has not been trained here.",
        "strengths": ["Could run on live audio, because it never looks ahead"],
        "limits": ["Not trained in this project, so there is no measured accuracy"],
        "stats": None,
    },
}

ENCODER = {
    "id": "wavlm-large",
    "name": "WavLM-Large",
    "role": "encoder",
    "kind": "encoder",
    "tagline": "Stage 1: turns sound into numbers a model can reason about",
    "description": (
        "A self-supervised speech model from Microsoft, pretrained on about 94,000 hours of English speech without any "
        "emotion labels. Each speech passage becomes a list of 1,024 numbers describing how it sounds: pitch, "
        "energy, rhythm, voice quality. EmotiSense uses it as it is (frozen): it is not retrained."
    ),
    "strengths": [
        "Learned from a huge amount of real speech, so its description of a voice is rich",
        "The same encoder feeds every timeline model, so you can switch models instantly",
    ],
    "limits": [
        "Large: 316 million parameters and a 1.2 GB download the first time",
        "Takes about 40 seconds per two minutes of audio on a laptop processor",
    ],
    "details": [
        ["Hugging Face name", "microsoft/wavlm-large"],
        ["Parameters", "316 million"],
        ["Transformer layers", "24, plus a convolutional front end"],
        ["Output", "1,024 numbers per passage"],
        ["How it is used", "Frozen; all layers averaged, then mean-pooled per passage"],
        ["Pretraining data", "About 94,000 hours of English speech"],
    ],
}

CLASSIFIER = {
    "id": "wavlm-large-classifier",
    "name": "Fine-tuned WavLM-Large classifier",
    "role": "classifier",
    "kind": "classifier",
    "tagline": "Reference model: names one of four emotions per clip",
    "description": (
        "WavLM-Large trained end to end to label a clip as neutral, happy, angry or sad, following the recipe of a "
        "2025 research paper plus a few extras (weight averaging, a valence/arousal side task, gentler learning "
        "rates for lower layers). It was trained and tested five times, each time holding out one whole session of "
        "speakers it had never heard."
    ),
    "strengths": [
        "Directly trained to recognise emotion, rather than reusing a general-purpose encoder",
        "Evaluated honestly: always on speakers and sessions it had not seen",
    ],
    "limits": [
        "Not part of the app's timeline: it gives one label per clip, not five curves over time",
        "Trained on one copy of an acted dataset; the figures are not comparable with published benchmarks",
        "641 MB per fold, so only one fold is kept on this computer",
    ],
    "details": [
        ["Classes", "neutral, happy, angry, sad"],
        ["Training data", "IEMOCAP (acted dialogues), 7,536 labelled clips in these four classes"],
        ["Training", "12 epochs per fold, early stopping, 5 folds by session"],
        ["Average over 5 folds", "69.1% weighted, 67.4% unweighted accuracy"],
        ["Chance level", "25%"],
    ],
    "stats": {
        "folds": [
            {"name": "Session 1", "wa": 72.72, "ua": 73.13}, {"name": "Session 2", "wa": 73.24, "ua": 71.73},
            {"name": "Session 3", "wa": 68.14, "ua": 67.16}, {"name": "Session 4", "wa": 67.09, "ua": 62.71},
            {"name": "Session 5", "wa": 64.49, "ua": 62.31},
        ],
        "mean": {"wa": 69.14, "ua": 67.41}, "std": {"wa": 3.36, "ua": 4.46},
        "recipe_comparison": {
            "label": "Session 1 only: plain paper recipe vs the improved recipe",
            "baseline": {"wa": 71.18, "ua": 71.59}, "improved": {"wa": 72.72, "ua": 73.13},
        },
    },
}

GLOSSARY = [
    {"term": "CCC", "plain": "Concordance correlation. How closely a curve follows the true emotional arc, in both shape and level. 1 is perfect, 0 is no better than guessing. Higher is better."},
    {"term": "RMSE and MAE", "plain": "Average size of the mistakes, on a 0 to 1 scale. RMSE punishes big misses more than MAE. Lower is better."},
    {"term": "Alert F1", "plain": "How well the model raises a tension alert in the same conversations where the labelled version does, balancing missed alerts against false ones. 1 is perfect."},
    {"term": "Trend accuracy", "plain": "How often the model correctly says whether tension is rising, easing or steady across a whole recording."},
    {"term": "Weighted and unweighted accuracy", "plain": "Weighted counts every clip equally. Unweighted averages the accuracy of each emotion, so rare emotions count as much as common ones."},
    {"term": "Held-out session", "plain": "Every result is measured on speakers the model never saw while learning, which is a much tougher and more honest test."},
]

SUPPORTED = [
    {"name": "WavLM-Base+", "note": "A smaller, faster encoder (94 million parameters). Lower accuracy; useful for quick experiments."},
    {"name": "HuBERT-Large and wav2vec 2.0", "note": "Alternative speech encoders that can be swapped in by changing one line in a config file."},
    {"name": "One-way LSTM and Transformer", "note": "Live-capable timeline heads that only look backwards. Supported in code, not trained here."},
    {"name": "Your own models", "note": "Train a new timeline model with `make timeline-all`; any checkpoint placed in the models folder appears here automatically."},
]

DATA_NOTE = (
    "All models were trained on the IEMOCAP acted-dialogue corpus (public copy with one label per clip). Timeline targets "
    "are therefore 0 or 1 per emotion, and results should be read as a demonstration of the method rather than a "
    "benchmark."
)
