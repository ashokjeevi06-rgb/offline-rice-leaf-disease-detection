import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageEnhance
import numpy as np
import pandas as pd
from datetime import datetime
import hashlib

from spikingjelly.activation_based import neuron
from spikingjelly.activation_based import functional
from spikingjelly.activation_based import surrogate


# ============================================================
# CONFIGURATION
# ============================================================


from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent / "rice_snn_v5_confusion_best.pth"


CLASS_NAMES = [
    "bacterial_leaf_blight",
    "bacterial_leaf_streak",
    "bacterial_panicle_blight",
    "blast",
    "brown_spot",
    "dead_heart",
    "downy_mildew",
    "hispa",
    "normal",
    "tungro"
]

# ============================================================
# DISEASE INFORMATION AND MANAGEMENT GUIDE
# ============================================================

DISEASE_GUIDE = {
    "bacterial_leaf_blight": {
        "name": "Bacterial Leaf Blight",
        "symptoms": [
            "Yellowing and drying that often starts near leaf tips or edges.",
            "Long yellowish or whitish lesions may extend along the leaf.",
            "Severe infection can cause extensive leaf drying."
        ],
        "cause": "A bacterial disease caused by Xanthomonas oryzae pv. oryzae.",
        "management": [
            "Grow resistant rice varieties where they are available.",
            "Avoid excessive nitrogen fertilizer and follow local nutrient recommendations.",
            "Use healthy seed and maintain good field sanitation.",
            "Ask a local agricultural extension officer to confirm severe or uncertain cases."
        ]
    },
    "bacterial_leaf_streak": {
        "name": "Bacterial Leaf Streak",
        "symptoms": [
            "Narrow, translucent streaks between the leaf veins.",
            "Streaks may become yellowish or brown as they develop."
        ],
        "cause": "A bacterial disease associated with Xanthomonas oryzae pv. oryzicola.",
        "management": [
            "Use healthy seed and recommended varieties.",
            "Avoid spreading infected plant material between fields.",
            "Maintain field sanitation and monitor new growth.",
            "Confirm the diagnosis with a local agricultural expert."
        ]
    },
    "bacterial_panicle_blight": {
        "name": "Bacterial Panicle Blight",
        "symptoms": [
            "Panicles or grains may become discolored.",
            "Grain filling may be poor or uneven."
        ],
        "cause": "Associated with Burkholderia glumae and related bacteria; symptoms can resemble other panicle disorders.",
        "management": [
            "Use healthy seed from reliable sources.",
            "Monitor plants around flowering and grain filling.",
            "Follow locally recommended crop and water management practices.",
            "Seek expert confirmation before choosing a treatment."
        ]
    },
    "blast": {
        "name": "Rice Blast",
        "symptoms": [
            "Spindle-shaped leaf spots, often with gray or whitish centers and darker margins.",
            "In severe cases, lesions can occur on nodes or panicles."
        ],
        "cause": "A fungal disease caused by Magnaporthe oryzae.",
        "management": [
            "Grow resistant varieties where available.",
            "Avoid excessive nitrogen fertilizer.",
            "Inspect the crop regularly, especially during favorable disease conditions.",
            "Use fungicides only when recommended by local agricultural guidance."
        ]
    },
    "brown_spot": {
        "name": "Brown Spot",
        "symptoms": [
            "Round or oval brown spots on leaves.",
            "Spots may have yellow halos and can increase in number.",
            "Severe infection may affect grain quality."
        ],
        "cause": "A fungal disease commonly associated with Bipolaris oryzae.",
        "management": [
            "Maintain balanced crop nutrition based on soil and local recommendations.",
            "Use healthy seed and locally recommended seed treatment practices.",
            "Manage water and crop health appropriately.",
            "Manage infected residues where practical and locally advised."
        ]
    },
    "dead_heart": {
        "name": "Dead Heart",
        "symptoms": [
            "The central shoot of a young rice plant dries out.",
            "The affected central shoot may pull out easily.",
            "Inspect for signs of stem-borer larvae or feeding damage."
        ],
        "cause": "A symptom commonly caused by rice stem-borer larvae damaging the growing point; it is not itself a disease name.",
        "management": [
            "Inspect plants and stems to confirm possible stem-borer damage.",
            "Monitor pest levels and conserve natural enemies.",
            "Use integrated pest management and locally recommended controls when needed.",
            "Avoid applying pesticides without confirming the pest and need."
        ]
    },
    "downy_mildew": {
        "name": "Downy Mildew",
        "symptoms": [
            "Pale yellow or whitish streaks may appear on leaves.",
            "Leaf shape or plant growth may become abnormal."
        ],
        "cause": "The label should be verified against the training dataset: downy mildew is not a typical standard rice disease class, and similar symptoms can have different causes.",
        "management": [
            "Do not select a treatment from the AI label alone.",
            "Take a clear image of the whole plant and affected leaves.",
            "Ask a local agricultural expert to identify the cause.",
            "Follow local guidance after the diagnosis is confirmed."
        ]
    },
    "hispa": {
        "name": "Rice Hispa",
        "symptoms": [
            "White scraped streaks or patches may appear on leaves.",
            "Leaves can look whitish or damaged.",
            "Beetles or larvae may be visible on affected plants."
        ],
        "cause": "Feeding damage caused by the rice hispa beetle, Dicladispa armigera.",
        "management": [
            "Inspect leaves for beetles and feeding damage.",
            "Monitor pest populations and conserve natural enemies.",
            "Remove heavily affected leaves where practical and appropriate.",
            "Use locally recommended integrated pest management measures if needed."
        ]
    },
    "normal": {
        "name": "Healthy / Normal Leaf",
        "symptoms": [
            "The model did not detect one of its trained disease classes with the highest score.",
            "A normal prediction does not guarantee that the plant is disease-free."
        ],
        "cause": "The model classified the uploaded image as normal.",
        "management": [
            "Continue regular field monitoring.",
            "Maintain balanced fertilizer and water management.",
            "Take another clear image or seek expert advice if symptoms appear."
        ]
    },
    "tungro": {
        "name": "Rice Tungro",
        "symptoms": [
            "Yellow to orange-yellow leaf discoloration.",
            "Stunted growth and reduced tillering may occur.",
            "Symptoms can vary with rice variety and infection stage."
        ],
        "cause": "Rice tungro is a viral disease complex transmitted by green leafhoppers.",
        "management": [
            "Grow resistant or tolerant varieties where available.",
            "Monitor green leafhoppers and use integrated pest management.",
            "Follow local guidance on managing infected plants and volunteer rice.",
            "Seek local agricultural advice; viral diseases cannot be cured by a fungicide."
        ]
    }
}


def show_disease_guide(predicted_class):
    """Display educational information for the predicted class."""
    guide = DISEASE_GUIDE.get(predicted_class)

    st.markdown("---")
    st.header("🌾 Disease Information & Management Guide")

    if guide is None:
        st.warning("No disease guide is available for this model class.")
        return

    st.subheader(guide["name"])

    with st.expander("🔍 Symptoms to look for", expanded=True):
        for symptom in guide["symptoms"]:
            st.markdown(f"- {symptom}")

    with st.expander("🦠 Possible cause", expanded=True):
        st.write(guide["cause"])

    with st.expander("🌱 Prevention and management", expanded=True):
        for item in guide["management"]:
            st.markdown(f"- {item}")

    st.info(
        "Important: This guide is for education only. The model's output is a "
        "screening prediction, not a confirmed diagnosis. Similar symptoms can "
        "have different causes. Confirm the problem with a local agricultural "
        "extension officer before making treatment decisions."
    )



NUM_CLASSES = 10
TIME_STEPS = 4


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Rice Leaf Disease Detection",
    page_icon="🌾",
    layout="wide"
)


# ============================================================
# EXACT MODEL ARCHITECTURE FOR CHECKPOINT
#
# Checkpoint dimensions:
#
# conv1.weight = [32, 3, 3, 3]
# conv2.weight = [64, 32, 3, 3]
# conv3.weight = [96, 64, 3, 3]
# fc.weight    = [10, 6144]
#
# 6144 = 96 × 8 × 8
#
# Input 64×64
#    ↓ pool1
# 32×32
#    ↓ pool2
# 16×16
#    ↓ pool3
#  8×8
# ============================================================

class RiceSNNV5(nn.Module):

    def __init__(self):

        super().__init__()

        # ----------------------------------------------------
        # BLOCK 1
        # ----------------------------------------------------

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.bn1 = nn.BatchNorm2d(32)

        self.lif1 = neuron.LIFNode(
            tau=2.0,
            surrogate_function=surrogate.ATan()
        )

        self.pool1 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )


        # ----------------------------------------------------
        # BLOCK 2
        # ----------------------------------------------------

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        self.bn2 = nn.BatchNorm2d(64)

        self.lif2 = neuron.LIFNode(
            tau=2.0,
            surrogate_function=surrogate.ATan()
        )

        self.pool2 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )


        # ----------------------------------------------------
        # BLOCK 3
        # EXACT CHECKPOINT: 64 -> 96
        # ----------------------------------------------------

        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=96,
            kernel_size=3,
            padding=1
        )

        self.bn3 = nn.BatchNorm2d(96)

        self.lif3 = neuron.LIFNode(
            tau=2.0,
            surrogate_function=surrogate.ATan()
        )

        self.pool3 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )


        # ----------------------------------------------------
        # CLASSIFIER
        #
        # 64x64
        # -> 32x32
        # -> 16x16
        # -> 8x8
        #
        # 96 × 8 × 8 = 6144
        # ----------------------------------------------------

        self.fc = nn.Linear(
            6144,
            NUM_CLASSES
        )


    def forward(self, x):

        outputs = []

        for _ in range(TIME_STEPS):

            # ----------------------------
            # BLOCK 1
            # ----------------------------

            x1 = self.conv1(x)

            x1 = self.bn1(x1)

            x1 = self.lif1(x1)

            x1 = self.pool1(x1)


            # ----------------------------
            # BLOCK 2
            # ----------------------------

            x2 = self.conv2(x1)

            x2 = self.bn2(x2)

            x2 = self.lif2(x2)

            x2 = self.pool2(x2)


            # ----------------------------
            # BLOCK 3
            # ----------------------------

            x3 = self.conv3(x2)

            x3 = self.bn3(x3)

            x3 = self.lif3(x3)

            x3 = self.pool3(x3)


            # ----------------------------
            # FLATTEN
            # ----------------------------

            x3 = torch.flatten(
                x3,
                start_dim=1
            )


            # ----------------------------
            # CLASSIFIER
            # ----------------------------

            out = self.fc(x3)

            outputs.append(out)


        # Temporal average
        return torch.stack(
            outputs,
            dim=0
        ).mean(dim=0)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    device = torch.device("cpu")

    model = RiceSNNV5()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )


    # --------------------------------------------------------
    # Handle different checkpoint formats
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
            ]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint


    # --------------------------------------------------------
    # Remove module. prefix if present
    # --------------------------------------------------------

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):

            key = key[7:]


        cleaned_state_dict[key] = value


    # --------------------------------------------------------
    # STRICT LOAD
    # --------------------------------------------------------

    model.load_state_dict(
        cleaned_state_dict,
        strict=True
    )


    model.to(device)

    model.eval()

    return model, device


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (64, 64)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# CREATE MULTI-VIEWS
# ============================================================

def create_views(image):

    views = [

        (
            "Original",
            image
        ),

        (
            "Horizontal Flip",
            image.transpose(
                Image.Transpose.FLIP_LEFT_RIGHT
            )
        ),

        (
            "Brightness +8%",
            ImageEnhance.Brightness(
                image
            ).enhance(1.08)
        ),

        (
            "Contrast +8%",
            ImageEnhance.Contrast(
                image
            ).enhance(1.08)
        ),

        (
            "Brightness -6%",
            ImageEnhance.Brightness(
                image
            ).enhance(0.94)
        )
    ]

    return views


# ============================================================
# SINGLE VIEW PREDICTION
# ============================================================

def run_single_view(
    model,
    device,
    image
):

    x = transform(
        image
    ).unsqueeze(0).to(device)


    with torch.no_grad():

        logits = model(x)

        probabilities = F.softmax(
            logits,
            dim=1
        )


    # Reset SNN membrane states
    functional.reset_net(model)


    return (
        probabilities[0]
        .cpu()
        .numpy()
    )


# ============================================================
# FINAL MULTI-VIEW PREDICTION
# ============================================================

def predict_multiview(
    model,
    device,
    image
):

    views = create_views(image)


    # Same weights used during
    # final test evaluation

    weights = np.array([
        1.00,
        0.95,
        0.90,
        0.90,
        0.90
    ])


    weights = (
        weights /
        weights.sum()
    )


    all_probabilities = []

    view_predictions = []


    # --------------------------------------------------------
    # Process all 5 views
    # --------------------------------------------------------

    for name, view in views:

        probabilities = run_single_view(
            model,
            device,
            view
        )


        all_probabilities.append(
            probabilities
        )


        prediction = int(
            np.argmax(
                probabilities
            )
        )


        view_predictions.append({

            "name":
                name,

            "prediction":
                prediction,

            "confidence":
                float(
                    probabilities[
                        prediction
                    ]
                )
        })


    all_probabilities = np.array(
        all_probabilities
    )


    # --------------------------------------------------------
    # Weighted fusion
    # --------------------------------------------------------

    fused_probabilities = np.sum(
        all_probabilities *
        weights[:, None],
        axis=0
    )


    fused_probabilities = (
        fused_probabilities /
        fused_probabilities.sum()
    )


    # --------------------------------------------------------
    # Top 2 predictions
    # --------------------------------------------------------

    order = np.argsort(
        fused_probabilities
    )[::-1]


    top1 = int(
        order[0]
    )

    top2 = int(
        order[1]
    )


    confidence = float(
        fused_probabilities[
            top1
        ]
    )


    second_confidence = float(
        fused_probabilities[
            top2
        ]
    )


    # --------------------------------------------------------
    # Prediction margin
    # --------------------------------------------------------

    margin = (
        confidence -
        second_confidence
    )


    # --------------------------------------------------------
    # Entropy
    # --------------------------------------------------------

    entropy = -np.sum(

        fused_probabilities *

        np.log(
            fused_probabilities
            + 1e-12
        )
    )


    # --------------------------------------------------------
    # View agreement
    # --------------------------------------------------------

    predicted_views = [

        view["prediction"]

        for view in view_predictions

    ]


    agreement = (

        predicted_views.count(
            top1
        )

        /

        len(predicted_views)

    )


    # ========================================================
    # DECISION GATE
    #
    # SAME THRESHOLDS AS FINAL EVALUATION
    # ========================================================

    if (

        confidence >= 0.70

        and

        margin >= 0.20

        and

        agreement >= 0.60

        and

        entropy <= 1.50

    ):

        decision = "ACCEPT"


    elif (

        confidence >= 0.55

        and

        margin >= 0.10

        and

        agreement >= 0.40

    ):

        decision = "REVIEW"


    else:

        decision = "RECAPTURE"


    return {

        "probabilities":
            fused_probabilities,

        "top1":
            top1,

        "top2":
            top2,

        "confidence":
            confidence,

        "second_confidence":
            second_confidence,

        "margin":
            margin,

        "entropy":
            float(entropy),

        "agreement":
            agreement,

        "decision":
            decision,

        "views":
            view_predictions
    }




# ============================================================
# EXPLAINABILITY: INPUT-GRADIENT SALIENCY HEATMAP
# ============================================================

def create_saliency_heatmap(model, device, image):
    """Exploratory input-gradient saliency overlay, not disease segmentation."""
    x = transform(image).unsqueeze(0).to(device)
    x.requires_grad_(True)
    functional.reset_net(model)
    model.zero_grad(set_to_none=True)
    try:
        logits = model(x)
        class_index = int(torch.argmax(logits, dim=1).item())
        logits[0, class_index].backward()
        if x.grad is None:
            return None
        saliency = x.grad.detach().abs().max(dim=1)[0][0].cpu().numpy()
        saliency -= saliency.min()
        if saliency.max() > 0:
            saliency /= saliency.max()
        saliency_img = Image.fromarray(
            np.uint8(np.clip(saliency, 0, 1) * 255)
        ).resize(image.size, Image.Resampling.BILINEAR)
        saliency = np.asarray(saliency_img).astype(np.float32) / 255.0
        heat = np.zeros((saliency.shape[0], saliency.shape[1], 3), dtype=np.uint8)
        heat[..., 0] = np.uint8(np.clip(saliency * 255 * 1.2, 0, 255))
        heat[..., 1] = np.uint8(np.clip((saliency ** 2) * 180, 0, 255))
        overlay = Image.blend(
            image.convert("RGB"), Image.fromarray(heat, mode="RGB"), alpha=0.42
        )
        return overlay
    finally:
        functional.reset_net(model)
        model.zero_grad(set_to_none=True)


def calculate_classification_metrics(y_true, y_pred, class_names):
    """Calculate metrics from labeled evaluation predictions."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError("True and predicted labels must be non-empty and equal in length.")
    n_classes = len(class_names)
    if np.any(y_true < 0) or np.any(y_true >= n_classes):
        raise ValueError("True labels must be class indices from 0 to 9.")
    if np.any(y_pred < 0) or np.any(y_pred >= n_classes):
        raise ValueError("Predicted labels must be class indices from 0 to 9.")
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        cm[true_label, pred_label] += 1
    accuracy = float(np.mean(y_true == y_pred))
    per_class, precisions, recalls, f1s = [], [], [], []
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        per_class.append({
            "Disease": name.replace("_", " ").title(),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1-score": round(f1, 4),
            "Support": int(cm[i, :].sum())
        })
    return {
        "accuracy": accuracy,
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "confusion_matrix": cm,
        "per_class": per_class
    }


# ============================================================
# HEADER
# ============================================================

st.title(
    "🌾 Rice Leaf Disease Detection"
)

st.subheader(
    "Spiking Neural Network + Multi-View Decision System"
)

st.write(
    "Upload a rice leaf image to obtain a disease "
    "prediction using multi-view SNN inference "
    "and confidence-aware decision gating."
)


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model, device = load_model()

    st.success(
        "✅ SNN model loaded successfully."
    )

except Exception as e:

    st.error(
        "❌ Model loading failed."
    )

    st.exception(e)

    st.stop()


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(

    "Upload Rice Leaf Image",

    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# PROCESS IMAGE
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")


    # --------------------------------------------------------
    # IMAGE + RESULT COLUMNS
    # --------------------------------------------------------

    left, right = st.columns(
        [1, 1]
    )


    with left:

        st.image(
            image,

            caption="Uploaded Rice Leaf",

            use_container_width=True
        )


        with st.expander("🔥 Explainability Heatmap (experimental)"):
            st.caption(
                "This input-gradient saliency overlay highlights pixels that influence "
                "the model score. It is not a confirmed disease segmentation mask."
            )
            if st.button("Generate Heatmap", key="generate_saliency_heatmap"):
                with st.spinner("Generating saliency heatmap..."):
                    try:
                        heatmap_image = create_saliency_heatmap(model, device, image)
                        if heatmap_image is not None:
                            st.image(
                                heatmap_image,
                                caption="Experimental input-gradient saliency overlay",
                                use_container_width=True
                            )
                        else:
                            st.warning("The model did not return input gradients.")
                    except Exception as heatmap_error:
                        st.warning("Heatmap could not be generated for this model/runtime.")
                        st.caption(str(heatmap_error))


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    with st.spinner(
        "Running SNN multi-view inference..."
    ):

        result = predict_multiview(

            model,

            device,

            image

        )


    predicted_class = (
        CLASS_NAMES[
            result["top1"]
        ]
    )


    second_class = (
        CLASS_NAMES[
            result["top2"]
        ]
    )


    # Store each unique uploaded image prediction in this session.
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []
    if "processed_image_hashes" not in st.session_state:
        st.session_state.processed_image_hashes = set()
    image_hash = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
    if image_hash not in st.session_state.processed_image_hashes:
        st.session_state.prediction_history.append({
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Image": uploaded_file.name,
            "Predicted Disease": predicted_class.replace("_", " ").title(),
            "Confidence (%)": round(result["confidence"] * 100, 2),
            "Decision": result["decision"]
        })
        st.session_state.processed_image_hashes.add(image_hash)


    # --------------------------------------------------------
    # MAIN PREDICTION
    # --------------------------------------------------------

    with right:

        st.markdown(
            "### 🌱 Prediction"
        )


        st.success(

            predicted_class
            .replace(
                "_",
                " "
            )
            .title()

        )


        st.metric(

            "Confidence",

            f"{result['confidence'] * 100:.2f}%"

        )


        st.metric(

            "Second Prediction",

            second_class
            .replace(
                "_",
                " "
            )
            .title()

        )


        st.metric(

            "Prediction Margin",

            f"{result['margin'] * 100:.2f}%"

        )


    # ========================================================
    # DECISION GATE
    # ========================================================

    st.markdown("---")

    st.markdown(
        "### 🧠 Decision Gate"
    )


    if result["decision"] == "ACCEPT":

        st.success(

            "✅ ACCEPT — prediction satisfies "
            "the defined confidence, margin, "
            "agreement and uncertainty criteria."

        )


    elif result["decision"] == "REVIEW":

        st.warning(

            "⚠️ REVIEW — prediction has moderate "
            "certainty. Manual verification is recommended."

        )


    else:

        st.error(

            "🔄 RECAPTURE — prediction is "
            "uncertain or unstable."

        )


        st.info(

            "Capture another clear image of the leaf. "
            "Use good lighting, keep the leaf in focus, "
            "fill most of the frame with the leaf, and "
            "reduce background clutter."

        )


    # ========================================================
    # UNCERTAINTY METRICS
    # ========================================================

    st.markdown("---")

    st.markdown(
        "### 📊 Uncertainty Analysis"
    )


    c1, c2, c3 = st.columns(3)


    with c1:

        st.metric(

            "View Agreement",

            f"{result['agreement'] * 100:.2f}%"

        )


    with c2:

        st.metric(

            "Prediction Margin",

            f"{result['margin'] * 100:.2f}%"

        )


    with c3:

        st.metric(

            "Entropy",

            f"{result['entropy']:.4f}"

        )


    # ========================================================
    # PROBABILITIES
    # ========================================================

    st.markdown("---")

    st.markdown(
        "### 📈 Disease Probabilities"
    )


    probability_data = []


    for i in range(NUM_CLASSES):

        probability_data.append({

            "Disease":

                CLASS_NAMES[i]
                .replace(
                    "_",
                    " "
                )
                .title(),

            "Probability":

                f"{result['probabilities'][i] * 100:.2f}%"

        })


    probability_data.sort(

        key=lambda x:

            float(
                x["Probability"]
                .replace(
                    "%",
                    ""
                )
            ),

        reverse=True

    )


    st.table(
        probability_data
    )


    # ========================================================
    # MULTI-VIEW ANALYSIS
    # ========================================================

    st.markdown("---")

    st.markdown(
        "### 🔍 Multi-View Analysis"
    )


    view_data = []


    for view in result["views"]:

        view_data.append({

            "View":
                view["name"],

            "Prediction":

                CLASS_NAMES[
                    view["prediction"]
                ]
                .replace(
                    "_",
                    " "
                )
                .title(),

            "Confidence":

                f"{view['confidence'] * 100:.2f}%"

        })


    st.table(
        view_data
    )


    # ========================================================
    # SYSTEM INFORMATION
    # ========================================================

    st.markdown("---")

    st.markdown(
        "### ⚙️ System Information"
    )


    info1, info2, info3 = st.columns(3)


    with info1:

        st.write(
            "**Model:** Rice SNN V5"
        )

        st.write(
            "**Inference:** Multi-view"
        )


    with info2:

        st.write(
            "**Views:** 5"
        )

        st.write(
            "**Classes:** 10"
        )


    with info3:

        st.write(
            "**Decision:** "
            + result["decision"]
        )

        st.write(
            "**Device:** CPU"
        )

    # ========================================================
    # INNOVATIVE FEATURE: DISEASE INFORMATION & MANAGEMENT GUIDE
    # ========================================================
    show_disease_guide(predicted_class)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Research prototype — Rice Leaf Disease Detection "
    "using Spiking Neural Network, multi-view inference, "
    "weighted probability fusion and uncertainty-aware "
    "decision gating."
)


# ============================================================
# EXTRA DASHBOARD FEATURES
# ============================================================

st.markdown("---")
st.header("📚 Prediction History")

if st.session_state.get("prediction_history"):
    history_df = pd.DataFrame(st.session_state.prediction_history)
    st.dataframe(history_df, use_container_width=True, hide_index=True)
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.subheader("Predictions by Disease")
        st.bar_chart(history_df["Predicted Disease"].value_counts())
    with chart_col2:
        st.subheader("Prediction Confidence")
        st.line_chart(history_df["Confidence (%)"])
    csv_data = history_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Prediction History (CSV)",
        data=csv_data,
        file_name="rice_prediction_history.csv",
        mime="text/csv"
    )
    if st.button("Clear Current Session History"):
        st.session_state.prediction_history = []
        st.session_state.processed_image_hashes = set()
        st.rerun()
else:
    st.info("Upload a rice leaf image to begin building prediction history.")

st.markdown("---")
st.header("📈 Model Performance Metrics")
st.write(
    "Upload a CSV containing actual ground-truth labels and predicted labels "
    "from a labeled evaluation/test dataset. Metrics are calculated from that CSV, "
    "not guessed from individual uploads."
)
with st.expander("CSV format instructions"):
    st.markdown(
        "CSV needs two columns named `y_true` and `y_pred`. Values can be class "
        "indices 0–9 or exact class names such as `blast` and `brown_spot`. "
        "Each row must represent one labeled evaluation image."
    )

metrics_file = st.file_uploader(
    "Upload labeled evaluation predictions CSV",
    type=["csv"],
    key="evaluation_metrics_csv"
)

def labels_to_indices(values, class_names):
    lookup = {name.lower(): idx for idx, name in enumerate(class_names)}
    lookup.update({name.replace("_", " ").lower(): idx for idx, name in enumerate(class_names)})
    converted = []
    for value in values:
        if pd.isna(value):
            raise ValueError("CSV labels cannot be empty.")
        text_value = str(value).strip()
        if text_value.isdigit():
            idx = int(text_value)
            if 0 <= idx < len(class_names):
                converted.append(idx)
                continue
        normalized = text_value.lower()
        if normalized in lookup:
            converted.append(lookup[normalized])
        else:
            raise ValueError(
                f"Unrecognized label '{text_value}'. Use class names or indices 0–9."
            )
    return converted

if metrics_file is not None:
    try:
        eval_df = pd.read_csv(metrics_file)
        if not {"y_true", "y_pred"}.issubset(eval_df.columns):
            st.error("CSV must contain columns named exactly `y_true` and `y_pred`.")
        else:
            y_true = labels_to_indices(eval_df["y_true"], CLASS_NAMES)
            y_pred = labels_to_indices(eval_df["y_pred"], CLASS_NAMES)
            metrics = calculate_classification_metrics(y_true, y_pred, CLASS_NAMES)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
            m2.metric("Macro Precision", f"{metrics['macro_precision'] * 100:.2f}%")
            m3.metric("Macro Recall", f"{metrics['macro_recall'] * 100:.2f}%")
            m4.metric("Macro F1-score", f"{metrics['macro_f1'] * 100:.2f}%")
            st.subheader("Confusion Matrix")
            st.caption("Rows = actual labels; columns = predicted labels.")
            pretty_names = [name.replace("_", " ").title() for name in CLASS_NAMES]
            cm_df = pd.DataFrame(
                metrics["confusion_matrix"], index=pretty_names, columns=pretty_names
            )
            st.dataframe(cm_df, use_container_width=True)
            st.subheader("Per-class Metrics")
            st.dataframe(pd.DataFrame(metrics["per_class"]), use_container_width=True, hide_index=True)
            st.download_button(
                "⬇️ Download Confusion Matrix (CSV)",
                data=cm_df.to_csv().encode("utf-8"),
                file_name="rice_confusion_matrix.csv",
                mime="text/csv"
            )
    except Exception as metrics_error:
        st.error(f"Could not calculate metrics: {metrics_error}")
else:
    st.info(
        "No evaluation CSV uploaded yet. Upload genuine labeled test results to display "
        "accuracy, precision, recall, F1-score, and the confusion matrix."
    )

