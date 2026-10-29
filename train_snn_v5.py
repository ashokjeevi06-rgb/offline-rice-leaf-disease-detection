import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from spikingjelly.activation_based import neuron, surrogate, functional, layer


# ==============================================================
# RICE LEAF SNN V5
# Balanced + Improved SNN Training
# ==============================================================

print("=" * 70)
print("RICE LEAF SNN V5 - BALANCED IMPROVED TRAINING")
print("=" * 70)


# ==============================================================
# CONFIGURATION
# ==============================================================

BASE_DIR = r"C:\Users\NITHYASHREE\Desktop"

DATA_DIR = os.path.join(
    BASE_DIR,
    "rice_snn_preprocessed"
)

TRAIN_DIR = os.path.join(
    DATA_DIR,
    "train"
)

VAL_DIR = os.path.join(
    DATA_DIR,
    "validation"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "rice_snn_v5_best.pth"
)

IMG_SIZE = 64

TIME_STEPS = 6

BATCH_SIZE = 32

EPOCHS = 25

LEARNING_RATE = 0.0003

WEIGHT_DECAY = 0.0001

PATIENCE = 7


DEVICE = torch.device("cpu")


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

NUM_CLASSES = len(CLASS_NAMES)


# ==============================================================
# DATA TRANSFORMS
# ==============================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (IMG_SIZE, IMG_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        10
    ),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.10
    ),

    transforms.ToTensor()
])


val_transform = transforms.Compose([

    transforms.Resize(
        (IMG_SIZE, IMG_SIZE)
    ),

    transforms.ToTensor()
])


# ==============================================================
# DATASETS
# ==============================================================

print()
print("Loading datasets...")


train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)


val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=val_transform
)


print()
print(
    "Training images:",
    len(train_dataset)
)

print(
    "Validation images:",
    len(val_dataset)
)

print(
    "Number of classes:",
    NUM_CLASSES
)


# ==============================================================
# CHECK CLASS ORDER
# ==============================================================

print()
print("Class mapping:")

for index, name in enumerate(
    train_dataset.classes
):

    print(
        f"{index} -> {name}"
    )


# ==============================================================
# CLASS BALANCED SAMPLING
# ==============================================================

class_counts = [
    0
    for _ in range(NUM_CLASSES)
]


for _, label in train_dataset.samples:

    class_counts[label] += 1


print()
print("Original class distribution:")

for i, count in enumerate(
    class_counts
):

    print(
        f"{CLASS_NAMES[i]:<32}: {count}"
    )


# --------------------------------------------------------------
# Calculate sample weights
# --------------------------------------------------------------

class_weights = []

for count in class_counts:

    class_weights.append(
        1.0 / count
    )


sample_weights = []

for _, label in train_dataset.samples:

    sample_weights.append(
        class_weights[label]
    )


sample_weights = torch.DoubleTensor(
    sample_weights
)


sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)


# ==============================================================
# DATA LOADERS
# ==============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=0
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ==============================================================
# V5 SNN MODEL
# ==============================================================

class RiceSNNV5(nn.Module):

    def __init__(
        self,
        num_classes=10
    ):

        super().__init__()


        # ------------------------------------------------------
        # BLOCK 1
        # ------------------------------------------------------

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


        # ------------------------------------------------------
        # BLOCK 2
        # ------------------------------------------------------

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


        # ------------------------------------------------------
        # BLOCK 3
        # ------------------------------------------------------

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


        # ------------------------------------------------------
        # CLASSIFIER
        # ------------------------------------------------------

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

            # --------------------------------------------------
            # Small temporal variation
            # --------------------------------------------------

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


            # --------------------------------------------------
            # BLOCK 1
            # --------------------------------------------------

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


            # --------------------------------------------------
            # BLOCK 2
            # --------------------------------------------------

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


            # --------------------------------------------------
            # BLOCK 3
            # --------------------------------------------------

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


            # --------------------------------------------------
            # CLASSIFIER
            # --------------------------------------------------

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


        # ------------------------------------------------------
        # Temporal average
        # ------------------------------------------------------

        output = torch.stack(
            outputs,
            dim=0
        ).mean(
            dim=0
        )

        return output


# ==============================================================
# CREATE MODEL
# ==============================================================

model = RiceSNNV5(
    num_classes=NUM_CLASSES
)

model = model.to(
    DEVICE
)


# ==============================================================
# LOSS
# ==============================================================

# Moderate class weighting.
# Weighted sampling already handles imbalance,
# so we avoid excessively large class weights.

loss_weights = torch.tensor(
    [
        1.20,
        1.30,
        1.30,
        0.85,
        1.00,
        0.90,
        1.20,
        0.95,
        0.90,
        1.00
    ],
    dtype=torch.float32
).to(
    DEVICE
)


criterion = nn.CrossEntropyLoss(
    weight=loss_weights,
    label_smoothing=0.05
)


# ==============================================================
# OPTIMIZER
# ==============================================================

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ==============================================================
# LEARNING RATE SCHEDULER
# ==============================================================

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)


# ==============================================================
# TRAINING VARIABLES
# ==============================================================

best_val_accuracy = 0.0

best_epoch = 0

epochs_without_improvement = 0


print()
print("=" * 70)
print("STARTING V5 TRAINING")
print("=" * 70)

print(
    "Device:",
    DEVICE
)

print(
    "Time steps:",
    TIME_STEPS
)

print(
    "Batch size:",
    BATCH_SIZE
)

print(
    "Learning rate:",
    LEARNING_RATE
)

print(
    "Weighted sampling: ENABLED"
)

print(
    "Label smoothing: ENABLED"
)


# ==============================================================
# TRAINING LOOP
# ==============================================================

for epoch in range(
    1,
    EPOCHS + 1
):

    # ==========================================================
    # TRAIN
    # ==========================================================

    model.train()

    functional.reset_net(
        model
    )

    running_loss = 0.0

    correct = 0

    total = 0


    for images, labels in train_loader:

        images = images.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )


        optimizer.zero_grad()


        outputs = model(
            images
        )


        loss = criterion(
            outputs,
            labels
        )


        loss.backward()


        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=2.0
        )


        optimizer.step()


        running_loss += (
            loss.item()
            *
            images.size(0)
        )


        _, predicted = torch.max(
            outputs,
            1
        )


        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


    train_loss = (
        running_loss /
        total
    )

    train_accuracy = (
        correct /
        total
    ) * 100


    # ==========================================================
    # VALIDATION
    # ==========================================================

    model.eval()

    functional.reset_net(
        model
    )

    val_loss_total = 0.0

    val_correct = 0

    val_total = 0


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )


            outputs = model(
                images
            )


            loss = criterion(
                outputs,
                labels
            )


            val_loss_total += (
                loss.item()
                *
                images.size(0)
            )


            _, predicted = torch.max(
                outputs,
                1
            )


            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()


    val_loss = (
        val_loss_total /
        val_total
    )

    val_accuracy = (
        val_correct /
        val_total
    ) * 100


    # ==========================================================
    # SCHEDULER
    # ==========================================================

    scheduler.step(
        val_accuracy
    )


    current_lr = optimizer.param_groups[0]["lr"]


    # ==========================================================
    # DISPLAY
    # ==========================================================

    print()
    print(
        f"Epoch [{epoch:02d}/{EPOCHS}]"
    )

    print(
        f"Train Loss : {train_loss:.4f}"
    )

    print(
        f"Train Acc  : {train_accuracy:.2f}%"
    )

    print(
        f"Val Loss   : {val_loss:.4f}"
    )

    print(
        f"Val Acc    : {val_accuracy:.2f}%"
    )

    print(
        f"Learning Rate : {current_lr:.7f}"
    )


    # ==========================================================
    # SAVE BEST MODEL
    # ==========================================================

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        best_epoch = epoch

        epochs_without_improvement = 0


        torch.save(
            model.state_dict(),
            MODEL_PATH
        )


        print(
            "✓ NEW BEST V5 MODEL SAVED"
        )

        print(
            f"  Validation Accuracy: "
            f"{best_val_accuracy:.2f}%"
        )


    else:

        epochs_without_improvement += 1


    # ==========================================================
    # EARLY STOPPING
    # ==========================================================

    if epochs_without_improvement >= PATIENCE:

        print()
        print(
            "Early stopping."
        )

        print(
            "No validation improvement."
        )

        break


# ==============================================================
# LOAD BEST MODEL
# ==============================================================

print()
print("=" * 70)
print("LOADING BEST V5 MODEL")
print("=" * 70)


best_state = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)


model.load_state_dict(
    best_state
)


model.eval()


# ==============================================================
# FINAL VALIDATION
# ==============================================================

print()
print(
    f"Best Validation Accuracy : "
    f"{best_val_accuracy:.2f}%"
)

print(
    f"Best Epoch               : "
    f"{best_epoch}"
)


# ==============================================================
# SAVE TRAINING INFORMATION
# ==============================================================

INFO_FILE = os.path.join(
    BASE_DIR,
    "rice_snn_v5_training_info.txt"
)


with open(
    INFO_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "RICE LEAF SNN V5 TRAINING\n"
    )

    f.write(
        "=" * 60 + "\n"
    )

    f.write(
        f"Best Validation Accuracy: "
        f"{best_val_accuracy:.2f}%\n"
    )

    f.write(
        f"Best Epoch: "
        f"{best_epoch}\n"
    )

    f.write(
        f"Time Steps: "
        f"{TIME_STEPS}\n"
    )

    f.write(
        f"Batch Size: "
        f"{BATCH_SIZE}\n"
    )

    f.write(
        f"Initial Learning Rate: "
        f"{LEARNING_RATE}\n"
    )


# ==============================================================
# COMPLETE
# ==============================================================

print()
print("=" * 70)
print("V5 TRAINING COMPLETED")
print("=" * 70)

print()
print(
    "Best model:"
)

print(
    MODEL_PATH
)

print()
print(
    "Training information:"
)

print(
    INFO_FILE
)

print()
print(
    "IMPORTANT:"
)

print(
    "The 1570-image test set has NOT been used."
)

print(
    "Run the separate V5 evaluation only after training."
)

print("=" * 70)