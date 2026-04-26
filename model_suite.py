from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF


@dataclass(slots=True)
class ModelConfig:
    num_classes: int = 38

    # Experiment 5: model architecture.
    backbone_name: str = "resnet18"
    mlp_hidden_dims: Tuple[int, ...] = (1024, 512)
    pooled_image_size: Tuple[int, int] = (300, 560)

    # Experiment 6: pretrained condition.
    pretrained: bool = False
    freeze_backbone: bool = False
    input_channels: int = 1

    # Experiment 8: hidden layer.
    hidden_dim: int | None = 512

    # Experiment 9: dropout rate.
    dropout: float = 0.4


@dataclass(slots=True)
class TrainingConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    label_smoothing: float = 0.1

    # Experiment 1: normalization method.
    normalization: str = "none"

    # Experiment 7: class-balanced training.
    use_class_weights: bool = False
    class_weight_cap: float | None = 10.0

    # Experiment 10: learning rate scheduler.
    scheduler_name: str = "plateau"
    scheduler_step_size: int = 10
    scheduler_gamma: float = 0.5

    # Experiment 6: staged fine-tuning after freezing.
    staged_unfreeze: bool = False
    unfreeze_epoch: int | None = None


@dataclass(slots=True)
class EvaluationResult:
    loss: float
    accuracy: float
    predictions: List[int]
    labels: List[int]


@dataclass(slots=True)
class EpochResult:
    loss: float
    accuracy: float


MODEL_PRESETS: Dict[str, Tuple[ModelConfig, TrainingConfig]] = {
    "resnet not pretrained": (
        ModelConfig(backbone_name="resnet18", pretrained=False, input_channels=1),
        TrainingConfig(),
    ),
    "pretrained resnet": (
        ModelConfig(backbone_name="resnet18", pretrained=True, input_channels=1),
        TrainingConfig(),
    ),
    "pretrained resnet with time context": (
        ModelConfig(backbone_name="resnet18", pretrained=True, input_channels=3),
        TrainingConfig(scheduler_name="step"),
    ),
    "not pretrained resnet with time context": (
        ModelConfig(backbone_name="resnet18", pretrained=False, input_channels=3),
        TrainingConfig(),
    ),
    "mobilenet not pretrained": (
        ModelConfig(backbone_name="mobilenetv2_100", pretrained=False, input_channels=1),
        TrainingConfig(),
    ),
    "efficientnet not pretrained": (
        ModelConfig(backbone_name="efficientnet_b0", pretrained=False, input_channels=1),
        TrainingConfig(),
    ),
    "convnet not pretrained": (
        ModelConfig(backbone_name="convnext_tiny", pretrained=False, input_channels=1),
        TrainingConfig(learning_rate=3e-4, weight_decay=0.05),
    ),
    "min-max normalization": (
        ModelConfig(backbone_name="resnet18", pretrained=False, input_channels=1),
        TrainingConfig(normalization="minmax"),
    ),
    "z normalization": (
        ModelConfig(backbone_name="resnet18", pretrained=False, input_channels=1),
        TrainingConfig(normalization="z_score"),
    ),
    "weighted training": (
        ModelConfig(backbone_name="resnet18", pretrained=True, input_channels=1),
        TrainingConfig(use_class_weights=True, weight_decay=1e-5),
    ),
    "dropout of 0": (
        ModelConfig(backbone_name="resnet18", pretrained=False, input_channels=1, dropout=0.0),
        TrainingConfig(),
    ),
    "step based time schedular": (
        ModelConfig(backbone_name="resnet18", pretrained=True, input_channels=1),
        TrainingConfig(scheduler_name="step"),
    ),
    "fine tuned resnet with freezing (pretrained)": (
        ModelConfig(backbone_name="resnet18", pretrained=True, freeze_backbone=True, input_channels=1),
        TrainingConfig(staged_unfreeze=True, unfreeze_epoch=10),
    ),
    "feed-forward model": (
        ModelConfig(
            backbone_name="feed_forward",
            pretrained=False,
            input_channels=3,
            hidden_dim=None,
            mlp_hidden_dims=(1024, 512),
            pooled_image_size=(300, 560),
        ),
        TrainingConfig(),
    ),
}


class FeedForwardKeystroke(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.pool = nn.AdaptiveAvgPool2d(config.pooled_image_size)

        pooled_height, pooled_width = config.pooled_image_size
        input_dim = config.input_channels * pooled_height * pooled_width

        layers: List[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in config.mlp_hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(config.dropout),
                ]
            )
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, config.num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = x.flatten(1)
        return self.classifier(x)


class KeystrokeClassifier(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = timm.create_model(config.backbone_name, pretrained=config.pretrained)
        self._adapt_input_layer(config.input_channels)

        in_features = self.backbone.get_classifier().in_features
        self.backbone.reset_classifier(0)

        if config.freeze_backbone:
            freeze_module(self.backbone, freeze=True)

        if config.hidden_dim is None:
            self.head = nn.Sequential(
                nn.BatchNorm1d(in_features),
                nn.Dropout(config.dropout),
                nn.Linear(in_features, config.num_classes),
            )
        else:
            self.head = nn.Sequential(
                nn.BatchNorm1d(in_features),
                nn.Dropout(config.dropout),
                nn.Linear(in_features, config.hidden_dim),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(config.hidden_dim),
                nn.Linear(config.hidden_dim, config.num_classes),
            )

    def _adapt_input_layer(self, input_channels: int) -> None:
        if input_channels == 3:
            return

        old_conv, setter = _resolve_first_conv(self.backbone, self.config.backbone_name)
        new_conv = nn.Conv2d(
            input_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

        with torch.no_grad():
            if old_conv.weight.shape[1] == 3 and input_channels == 1:
                new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
            if old_conv.bias is not None and new_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)

        setter(new_conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(x)
        if features.ndim > 2:
            features = self.backbone.forward_head(features, pre_logits=True)
        return self.head(features)


def _resolve_first_conv(model: nn.Module, backbone_name: str):
    if backbone_name.startswith("resnet"):
        return model.conv1, lambda layer: setattr(model, "conv1", layer)
    if backbone_name.startswith("mobilenet"):
        return model.conv_stem, lambda layer: setattr(model, "conv_stem", layer)
    if backbone_name.startswith("convnext"):
        return model.stem[0], lambda layer: model.stem.__setitem__(0, layer)
    return model.conv_stem, lambda layer: setattr(model, "conv_stem", layer)


def freeze_module(module: nn.Module, freeze: bool = True) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = not freeze


def unfreeze_last_backbone_block(model: KeystrokeClassifier) -> None:
    backbone_name = model.config.backbone_name
    if backbone_name.startswith("resnet"):
        freeze_module(model.backbone.layer4, freeze=False)
    elif backbone_name.startswith("mobilenet"):
        freeze_module(model.backbone.blocks[-1], freeze=False)
    elif backbone_name.startswith("convnext"):
        freeze_module(model.backbone.stages[-1], freeze=False)
    else:
        freeze_module(model.backbone.blocks[-1], freeze=False)


def build_model(config: ModelConfig) -> nn.Module:
    if config.backbone_name == "feed_forward":
        return FeedForwardKeystroke(config)
    return KeystrokeClassifier(config)


def normalize_batch(images: torch.Tensor, training_config: TrainingConfig) -> torch.Tensor:
    images = images.float()
    if images.max() > 1.5:
        images = images / 255.0

    channels = images.shape[1]
    if training_config.normalization == "none":
        return images
    if training_config.normalization == "minmax":
        mean = [0.5] * channels
        std = [0.5] * channels
        return TF.normalize(images, mean=mean, std=std)
    if training_config.normalization == "z_score":
        mean = [0.352440] * channels
        std = [0.324912] * channels
        return TF.normalize(images, mean=mean, std=std)
    raise ValueError(f"Unsupported normalization mode: {training_config.normalization}")


def compute_class_weights(
    labels: Iterable[int],
    num_classes: int,
    clamp_max: float | None = 10.0,
) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.float32)
    total = 0
    for label in labels:
        counts[int(label)] += 1
        total += 1

    weights = torch.zeros_like(counts)
    for index, count in enumerate(counts):
        if count > 0:
            weights[index] = total / (num_classes * count)

    if clamp_max is not None:
        weights = torch.clamp(weights, max=clamp_max)
    return weights


def build_loss(
    training_config: TrainingConfig,
    class_weights: torch.Tensor | None = None,
) -> nn.Module:
    if training_config.use_class_weights and class_weights is not None:
        return nn.CrossEntropyLoss(
            label_smoothing=training_config.label_smoothing,
            weight=class_weights,
        )
    return nn.CrossEntropyLoss(label_smoothing=training_config.label_smoothing)


def build_optimizer(model: nn.Module, training_config: TrainingConfig) -> optim.Optimizer:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return optim.Adam(parameters, lr=training_config.learning_rate, weight_decay=training_config.weight_decay)


def build_scheduler(optimizer: optim.Optimizer, training_config: TrainingConfig):
    if training_config.scheduler_name == "none":
        return None
    if training_config.scheduler_name == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=training_config.scheduler_step_size,
            gamma=training_config.scheduler_gamma,
        )
    if training_config.scheduler_name == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=5,
            factor=training_config.scheduler_gamma,
        )
    raise ValueError(f"Unsupported scheduler: {training_config.scheduler_name}")


def maybe_unfreeze_backbone(
    model: KeystrokeClassifier,
    training_config: TrainingConfig,
    current_epoch: int,
) -> bool:
    if not training_config.staged_unfreeze:
        return False
    if training_config.unfreeze_epoch is None:
        return False
    if current_epoch != training_config.unfreeze_epoch:
        return False

    freeze_module(model.backbone, freeze=True)
    unfreeze_last_backbone_block(model)
    freeze_module(model.head, freeze=False)
    return True


def _compute_accuracy(predictions: List[int], labels: List[int]) -> float:
    if not labels:
        return 0.0
    correct = sum(int(prediction == label) for prediction, label in zip(predictions, labels))
    return correct / len(labels)


def train_one_epoch(
    model: nn.Module,
    train_loader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    training_config: TrainingConfig,
    device: torch.device | str,
) -> EpochResult:
    model.train()
    running_loss = 0.0
    predictions: List[int] = []
    labels_list: List[int] = []

    for images, labels in train_loader:
        images = normalize_batch(images.to(device), training_config)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        batch_predictions = outputs.argmax(dim=1)
        predictions.extend(batch_predictions.detach().cpu().tolist())
        labels_list.extend(labels.detach().cpu().tolist())

    dataset_size = len(train_loader.dataset)
    average_loss = running_loss / dataset_size if dataset_size else 0.0
    accuracy = _compute_accuracy(predictions, labels_list)
    return EpochResult(loss=average_loss, accuracy=accuracy)


def validate_model(
    model: nn.Module,
    validation_loader,
    criterion: nn.Module,
    training_config: TrainingConfig,
    device: torch.device | str,
) -> EvaluationResult:
    return evaluate_model(
        model=model,
        data_loader=validation_loader,
        criterion=criterion,
        training_config=training_config,
        device=device,
    )


def evaluate_model(
    model: nn.Module,
    data_loader,
    criterion: nn.Module,
    training_config: TrainingConfig,
    device: torch.device | str,
) -> EvaluationResult:
    model.eval()
    running_loss = 0.0
    predictions: List[int] = []
    labels_list: List[int] = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = normalize_batch(images.to(device), training_config)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)

            batch_predictions = outputs.argmax(dim=1)
            predictions.extend(batch_predictions.cpu().tolist())
            labels_list.extend(labels.cpu().tolist())

    dataset_size = len(data_loader.dataset)
    average_loss = running_loss / dataset_size if dataset_size else 0.0
    accuracy = _compute_accuracy(predictions, labels_list)
    return EvaluationResult(
        loss=average_loss,
        accuracy=accuracy,
        predictions=predictions,
        labels=labels_list,
    )


def test_model(
    model: nn.Module,
    test_loader,
    criterion: nn.Module,
    training_config: TrainingConfig,
    device: torch.device | str,
) -> EvaluationResult:
    return evaluate_model(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        training_config=training_config,
        device=device,
    )


def run_training_epoch(
    model: KeystrokeClassifier,
    train_loader,
    validation_loader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    training_config: TrainingConfig,
    device: torch.device | str,
    current_epoch: int,
) -> Tuple[EpochResult, EvaluationResult, optim.Optimizer, object]:
    unfreezing_applied = maybe_unfreeze_backbone(model, training_config, current_epoch)
    if unfreezing_applied:
        optimizer = build_optimizer(model, training_config)
        scheduler = build_scheduler(optimizer, training_config)

    train_result = train_one_epoch(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        training_config=training_config,
        device=device,
    )
    validation_result = validate_model(
        model=model,
        validation_loader=validation_loader,
        criterion=criterion,
        training_config=training_config,
        device=device,
    )

    if scheduler is not None:
        if training_config.scheduler_name == "plateau":
            scheduler.step(validation_result.loss)
        else:
            scheduler.step()

    return train_result, validation_result, optimizer, scheduler
