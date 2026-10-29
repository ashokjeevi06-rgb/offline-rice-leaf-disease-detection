import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from spikingjelly.activation_based import neuron, surrogate, functional, layer


# ==============================================================
# RICE LEAF SNN V5 - CONFIDENCE AWARE PREDICTION
# ==============================================================

print("=" * 70)
print("RICE LEAF SNN V5 - CONFIDENCE AWARE PREDICTION")
print("=" * 70)


# ==============================================================
# CONFIGURATION
# ==============================================================

BASE_DIR = r"C:\Users\NITHYASHREE\Desktop"

MODEL_PATH = os.path.join(
    BASE_DIR,
    "rice_snn_v5_best.pth"
)

IMAGE_SIZE = 64
TIME_STEPS = 6
NUM_CLASSES = 10

DEVICE = torch.device("cpu")

# --------------------------------------------------------------
# Confidence threshold
# --------------------------------------------------------------

CONFIDENCE_THRESHOLD = 0.70


# ==============================================================
# CLASS NAMES
# ==============================================================

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


# ==============================================================
# IMAGE TRANSFORM
# Same as V5 validation/test
# ==============================================================

transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor()
])


# ==============================================================
# V5 MODEL
# ==============================================================

class RiceSNNV5(nn.Module):

    def __init__(
        self,
        num_classes=10
    ):

        super().__init__()


        # BLOCK 1

        self.conv1 = layer.Conv2d(
            3,
            32,
            kernel_size=3,
            padding=1
        )

        self.bn1 = nn.BatchNorm2d(
            32
        )

        self.lif1 = neuron.LIFNode(
            surrogate_function=surrogate.ATan()
        )

        self.pool1 = layer.MaxPool2d(
            2,
            2
        )


        # BLOCK 2

        self.conv2 = layer.Conv2d(
            32,
            64,
            kernel_size=3,
            padding=1
        )

        self.bn2 = nn.BatchNorm2d(
            64
        )

        self.lif2 = neuron.LIFNode(
            surrogate_function=surrogate.ATan()
        )

        self.pool2 = layer.MaxPool2d(
            2,
            2
        )


        # BLOCK 3

        self.conv3 = layer.Conv2d(
            64,
            96,
            kernel_size=3,
            padding=1
        )

        self.bn3 = nn.BatchNorm2d(
            96
        )

        self.lif3 = neuron.LIFNode(
            surrogate_function=surrogate.ATan()
        )

        self.pool3 = layer.MaxPool2d(
            2,
            2
        )


        # CLASSIFIER

        self.dropout = nn.Dropout(
            0.20
        )

        self.fc = nn.Linear(
            96 * 8 * 8,
            num_classes
        )


    # ==========================================================
    # FORWARD
    # ==========================================================

    def forward(self, x):

        functional.reset_net(
            self
        )

        outputs = []


        for t in range(
            TIME_STEPS
        ):

            scale = (
                0.90
                +
                0.10 *
                t /
                max(
                    1,
                    TIME_STEPS - 1
                )
            )

            xt = x * scale


            # BLOCK 1

            y = self.conv1(
                xt
            )

            y = self.bn1(
                y
            )

            y = self.lif1(
                y
            )

            y = self.pool1(
                y
            )


            # BLOCK 2

            y = self.conv2(
                y
            )

            y = self.bn2(
                y
            )

            y = self.lif2(
                y
            )

            y = self.pool2(
                y
            )


            # BLOCK 3

            y = self.conv3(
                y
            )

            y = self.bn3(
                y
            )

            y = self.lif3(
                y
            )

            y = self.pool3(
                y
            )


            # CLASSIFIER

            y = torch.flatten(
                y,
                start_dim=1
            )

            y = self.dropout(
                y
            )

            y = self.fc(
                y
            )

            outputs.append(
                y
            )


        output = torch.stack(
            outputs,
            dim=0
        ).mean(
            dim=0
        )

        return output


# ==============================================================
# LOAD MODEL
# ==============================================================

print()
print("Loading V5 model...")

model = RiceSNNV5(
    num_classes=NUM_CLASSES
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.to(
    DEVICE
)

model.eval()

print("✓ Model loaded successfully.")


# ==============================================================
# GET IMAGE PATH
# ==============================================================

print()
print("=" * 70)
print("IMAGE INPUT")
print("=" * 70)

image_path = input(
    "Enter complete image path: "
).strip().strip('"')


# ==============================================================
# CHECK IMAGE
# ==============================================================

if not os.path.exists(
    image_path
):

    print()
    print("ERROR: Image file not found.")

    print()
    print("Check the path and try again.")

    raise SystemExit


# ==============================================================
# LOAD IMAGE
# ==============================================================

try:

    image = Image.open(
        image_path
    ).convert(
        "RGB"
    )

except Exception as e:

    print()
    print("ERROR: Could not open image.")

    print(e)

    raise SystemExit


print()
print(
    "Original image size:",
    image.size
)


# ==============================================================
# PREPROCESS
# ==============================================================

input_tensor = transform(
    image
)

input_tensor = input_tensor.unsqueeze(
    0
)

input_tensor = input_tensor.to(
    DEVICE
)


# ==============================================================
# PREDICTION
# ==============================================================

with torch.no_grad():

    outputs = model(
        input_tensor
    )

    probabilities = torch.softmax(
        outputs,
        dim=1
    )


# ==============================================================
# TOP 3
# ==============================================================

top_probabilities, top_indices = torch.topk(
    probabilities,
    k=3,
    dim=1
)


# ==============================================================
# FINAL PREDICTION
# ==============================================================

predicted_index = top_indices[0][0].item()

predicted_class = CLASS_NAMES[
    predicted_index
]

confidence = top_probabilities[
    0
][0].item()


# ==============================================================
# DECISION
# ==============================================================

if confidence >= CONFIDENCE_THRESHOLD:

    decision = "ACCEPT"

    status = "HIGH CONFIDENCE"

else:

    decision = "RECAPTURE IMAGE"

    status = "LOW CONFIDENCE"


# ==============================================================
# DISPLAY RESULTS
# ==============================================================

print()
print("=" * 70)
print("PREDICTION RESULT")
print("=" * 70)

print()
print(
    "Predicted Disease :",
    predicted_class
)

print(
    "Confidence        :",
    f"{confidence * 100:.2f}%"
)

print(
    "Threshold         :",
    f"{CONFIDENCE_THRESHOLD * 100:.0f}%"
)

print(
    "Status            :",
    status
)

print(
    "Decision          :",
    decision
)


# ==============================================================
# TOP 3 PREDICTIONS
# ==============================================================

print()
print("=" * 70)
print("TOP 3 PREDICTIONS")
print("=" * 70)


for rank in range(3):

    class_index = top_indices[
        0
    ][rank].item()

    class_name = CLASS_NAMES[
        class_index
    ]

    class_probability = top_probabilities[
        0
    ][rank].item()


    print(
        f"{rank + 1}. "
        f"{class_name:<32} "
        f"{class_probability * 100:.2f}%"
    )


# ==============================================================
# PATENT SYSTEM INTERPRETATION
# ==============================================================

print()
print("=" * 70)
print("SYSTEM DECISION")
print("=" * 70)


if confidence >= 0.80:

    print()
    print("✓ HIGH-CONFIDENCE DIAGNOSIS")

    print(
        "The prediction is sufficiently confident."
    )

elif confidence >= 0.70:

    print()
    print("✓ ACCEPT WITH VERIFICATION")

    print(
        "Prediction passed the confidence threshold."
    )

    print(
        "Additional image verification is recommended."
    )

else:

    print()
    print("⚠ UNCERTAIN PREDICTION")

    print(
        "Please capture another clear rice-leaf image."
    )

    print(
        "The system should not force a diagnosis."
    )


print()
print("=" * 70)
print("PREDICTION COMPLETED")
print("=" * 70)