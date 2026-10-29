import streamlit as st
import torch
import torch.nn as nn
from pathlib import Path
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageEnhance
import numpy as np

from spikingjelly.activation_based import neuron
from spikingjelly.activation_based import functional
from spikingjelly.activation_based import surrogate


# ============================================================
# CONFIGURATION
# ============================================================

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