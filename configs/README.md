# configs/ — Centralized Configuration

`configs/config.yaml` is the **single source of truth** for every hyperparameter, path, and setting in the project. All training scripts, dataloaders, and the API server read from it at start-up instead of relying on hardcoded constants.

```
configs/
└── config.yaml    Main configuration file
```

---

## How Scripts Use config.yaml

```python
import yaml

with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

lr           = cfg["cnn"]["learning_rate"]                        # 0.0001
patience     = cfg["training"]["early_stopping"]["patience"]      # 3
classes      = cfg["emotions"]["classes"]                         # ["angry", ...]
```

Every training script accepts CLI flags that **override** config values at run time:

```bash
# No need to edit config.yaml for a one-off experiment
python scripts/train_cnn.py --epochs 50 --lr 5e-5
```

---

## Quick Reference

| What to change | Where in config.yaml |
|----------------|----------------------|
| Dataset location | `paths.data_raw` |
| Checkpoint save path | `paths.checkpoints` |
| CNN learning rate | `cnn.learning_rate` |
| BERT fine-tuning epochs | `bert.epochs` |
| Early stopping patience | `training.early_stopping.patience` |
| Fusion strategy | `fusion.type` |
| Claude model version | `genai.model` |
| Global random seed | `project.seed` |

---

## Section Reference

### project

```yaml
project:
  name: "Multimodal Emotion Recognition"
  seed: 42
```

`seed` is passed to `torch.manual_seed()` and sklearn splits for reproducibility.

---

### paths

```yaml
paths:
  data_raw:        "data/raw"
  data_processed:  "data/processed"
  checkpoints:     "outputs/checkpoints"
  figures:         "outputs/figures"
  reports:         "outputs/reports"
```

All paths are relative to the project root. Moving the project only requires a one-line edit here.

---

### emotions

```yaml
emotions:
  classes: ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
  num_classes: 7
```

The canonical ordered label list. `classes[i]` is the human-readable name for integer label `i`. All models output logits of shape `(B, 7)` matching this ordering.

---

### image

```yaml
image:
  size: 48           # FER2013 native resolution
  resize: 224        # Required by ResNet-50 / ViT
  channels: 3        # Convert grayscale → RGB
  mean: [0.485, 0.456, 0.406]
  std:  [0.229, 0.224, 0.225]
  augmentation:
    horizontal_flip: true
    random_crop: true
    color_jitter: true
    rotation_degrees: 10
```

`mean` and `std` are ImageNet statistics — ResNet-50 and ViT-B/16 are pretrained on ImageNet and expect the same normalisation. Augmentation flags are read by `src/preprocessing/image_preprocessing.py` when building the training transform pipeline.

---

### text

```yaml
text:
  max_length: 128
  vocab_size: 30000
  embedding_dim: 100
  glove_path: "data/raw/glove.6B.100d.txt"
```

| Key | Used by | Purpose |
|-----|---------|---------|
| `max_length` | BERT + LSTM | Truncation / padding length |
| `vocab_size` | LSTM fallback | Vocabulary ceiling |
| `embedding_dim` | GloVe + BiLSTM | Embedding vector width (100-d) |
| `glove_path` | `train_lstm.py` | Location of the pre-downloaded GloVe file |

---

### cnn (ResNet-50)

```yaml
cnn:
  backbone: "resnet50"
  pretrained: true
  dropout: 0.5
  learning_rate: 0.0001
  weight_decay: 0.0001
  backbone_lr_factor: 0.1
  batch_size: 64
  epochs: 30
  scheduler: "cosine"
```

`backbone_lr_factor: 0.1` means pretrained layers train at `lr × 0.1 = 1e-5` while the new classifier head trains at `lr = 1e-4`. This avoids catastrophic forgetting of ImageNet features.

---

### vit (ViT-B/16)

```yaml
vit:
  backbone: "google/vit-base-patch16-224"
  dropout: 0.1
  learning_rate: 0.00002
  weight_decay: 0.01
  batch_size: 32
  epochs: 20
```

ViT uses higher `weight_decay` (0.01) than ResNet-50 (0.0001) because it has no BatchNorm layers — explicit L2 is needed to stabilise fine-tuning on the smaller FER2013 dataset.

---

### lstm (BiLSTM + GloVe)

```yaml
lstm:
  hidden_size: 256
  num_layers: 2
  bidirectional: true
  dropout: 0.4
  learning_rate: 0.001
  weight_decay: 0.0001
  batch_size: 128
  epochs: 30
```

`hidden_size: 256` with `bidirectional: true` → 512-d concatenated hidden state used as the sentence feature vector.

---

### bert

```yaml
bert:
  model_name: "bert-base-uncased"
  dropout: 0.3
  learning_rate: 0.00002
  weight_decay: 0.01
  no_decay_params: ["bias", "LayerNorm.weight"]
  batch_size: 32
  epochs: 10
  warmup_steps: 500
```

`no_decay_params` lists parameter name substrings that receive **zero** weight decay. Applying L2 to bias terms and LayerNorm weights hurts BERT fine-tuning stability (per the original paper).

---

### fusion

```yaml
fusion:
  image_feature_dim: 512
  text_feature_dim: 768
  hidden_dim: 512
  dropout: 0.3
  learning_rate: 0.0001
  weight_decay_encoders: 0.0001
  weight_decay_fusion: 0.001
  batch_size: 32
  epochs: 20
  type: "attention"           # early | late | attention
  attention:
    num_heads: 8
    d_model: 512
```

Two separate weight-decay values: lighter penalty for pre-trained encoder layers, stronger penalty for the randomly-initialised fusion MLP. Override `type` at the command line with `--fusion early`.

---

### training.early_stopping

```yaml
training:
  early_stopping:
    patience: 3
    min_delta: 0.001
    monitor: "val_loss"
    restore_best: true
```

| Key | Effect |
|-----|--------|
| `patience` | Epochs to wait after last improvement before stopping |
| `min_delta` | Minimum absolute change that counts as an improvement |
| `monitor` | `"val_loss"` (mode=min) or `"val_acc"` (mode=max) |
| `restore_best` | Reload best checkpoint weights automatically when training stops |

---

### genai

```yaml
genai:
  provider: "ollama"
  model: "llama3.2"
  max_tokens: 600
  temperature: 0.7
```

Read by `src/genai/report_generator.py`. Set `temperature: 0.0` for deterministic, reproducible reports in testing. The model must be available locally via `ollama pull llama3.2`.

---

## Extending the Configuration

To add a new model section (e.g. `audio`):

```yaml
audio:
  model_name: "wav2vec2-base"
  sample_rate: 16000
  max_duration: 5.0
  learning_rate: 0.00005
  batch_size: 16
  epochs: 20
```

Then reference it in the new training script:

```python
audio_cfg = cfg["audio"]
lr = audio_cfg["learning_rate"]
```

---

*Last Updated: 23/05/2026 — Status: Active ✓*
