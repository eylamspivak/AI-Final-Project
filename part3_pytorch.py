# =============================================================================
# PART III: Neural Network Classification with PyTorch (PyTorchNN.ipynb)
# =============================================================================
# This script implements everything inside the TODO blocks the assignment asks for.
# It includes:
#   - Model architecture search (MLP with varying depth/width)
#   - Dropout regularisation
#   - Learning rate and batch size selection
#   - Training loop with early stopping
#   - Validation-based hyperparameter selection
#   - Final test evaluation
# =============================================================================

import numpy as np
import pickle, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# ---- PyTorch ----
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

plt.style.use('seaborn-v0_8-whitegrid')
COLORS = ['#2563EB', '#DC2626', '#16A34A', '#D97706', '#7C3AED']
CIFAR_CLASSES = ['airplane','automobile','bird','cat','deer',
                 'dog','frog','horse','ship','truck']

# --------------------------------------------------------------------------
# Device
# --------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --------------------------------------------------------------------------
# DATA LOADING  (same as Part II)
# --------------------------------------------------------------------------
CIFAR_PATH = r"C:\Users\Eilam\Downloads\cifar-10-python\cifar-10-batches-py"

def load_cifar10_batch(filepath):
    with open(filepath, 'rb') as f:
        d = pickle.load(f, encoding='bytes')
    X = d[b'data'].astype(np.float32)
    y = np.array(d[b'labels'])
    return X, y

def load_cifar10(data_dir):
    Xs, ys = [], []
    for i in range(1, 6):
        path = os.path.join(data_dir, f'data_batch_{i}')
        X, y = load_cifar10_batch(path); Xs.append(X); ys.append(y)
    X_train = np.concatenate(Xs); y_train = np.concatenate(ys)
    test_path = os.path.join(data_dir, 'test_batch')
    X_test, y_test = load_cifar10_batch(test_path)
    return X_train, y_train, X_test, y_test

print("Loading CIFAR-10...")
try:
    X_full_tr, y_full_tr, X_full_te, y_full_te = load_cifar10(CIFAR_PATH)
    print(f"Loaded — Train: {X_full_tr.shape}  Test: {X_full_te.shape}")
    REAL_DATA = True
except Exception as e:
    print(f"Could not load CIFAR-10: {e}")
    print("Using synthetic data for demonstration.")
    np.random.seed(42)
    X_full_tr = np.random.rand(50000, 3072).astype(np.float32) * 255
    y_full_tr = np.random.randint(0, 10, 50000)
    X_full_te = np.random.rand(10000, 3072).astype(np.float32) * 255
    y_full_te = np.random.randint(0, 10, 10000)
    REAL_DATA = False

# Subsample: 20 000 train / 5 000 val / 5 000 test
np.random.seed(42)
TRAIN_N, VAL_N, TEST_N = 20000, 5000, 5000

idx = np.random.permutation(len(X_full_tr))
X_tr_raw = X_full_tr[idx[:TRAIN_N]]
y_tr_raw = y_full_tr[idx[:TRAIN_N]]
X_va_raw = X_full_tr[idx[TRAIN_N:TRAIN_N+VAL_N]]
y_va_raw = y_full_tr[idx[TRAIN_N:TRAIN_N+VAL_N]]

idx_te = np.random.choice(len(X_full_te), TEST_N, replace=False)
X_te_raw = X_full_te[idx_te]
y_te_raw = y_full_te[idx_te]

# Normalise to [0,1] and compute per-channel mean/std on training set
X_tr_n = X_tr_raw / 255.0
X_va_n = X_va_raw / 255.0
X_te_n = X_te_raw / 255.0

mean = X_tr_n.mean(axis=0)
std  = X_tr_n.std(axis=0) + 1e-8

X_tr_n = (X_tr_n - mean) / std
X_va_n = (X_va_n - mean) / std
X_te_n = (X_te_n - mean) / std

print(f"Post-normalise shapes — Train: {X_tr_n.shape}  Val: {X_va_n.shape}  Test: {X_te_n.shape}")

# --------------------------------------------------------------------------
# BUILD PYTORCH DATASETS
# --------------------------------------------------------------------------
def make_loader(X, y, batch_size, shuffle=True):
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    ds = TensorDataset(Xt, yt)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

# --------------------------------------------------------------------------
# ============================================================
# TODO BLOCK — MODEL ARCHITECTURE (as required by the assignment)
# ============================================================
# We implement a Multi-Layer Perceptron (MLP) with:
#   • Configurable depth (number of hidden layers)
#   • Configurable width (hidden layer size)
#   • Batch Normalisation after each linear layer
#   • ReLU activations
#   • Dropout for regularisation
#   • A final linear output layer (10 classes)
# --------------------------------------------------------------------------

class MLP(nn.Module):
    """
    Flexible Multi-Layer Perceptron for CIFAR-10 classification.

    Architecture:
        Input (3072) → [Linear → BatchNorm → ReLU → Dropout] × n_hidden_layers
                     → Linear (10)

    Args:
        input_dim   : number of input features (3072 for flattened CIFAR-10)
        hidden_sizes: list of integers, one per hidden layer
        dropout_p   : dropout probability applied after each activation
        n_classes   : number of output classes (10)
    """
    def __init__(self, input_dim=3072, hidden_sizes=[512, 256], dropout_p=0.3, n_classes=10):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_sizes:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_p),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # x shape: (batch, 3072)

# --------------------------------------------------------------------------
# TODO BLOCK — TRAINING CONFIGURATION & HYPERPARAMETER SEARCH
# --------------------------------------------------------------------------
# We compare several configurations on the validation set and select the best.

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(Xb)
        loss   = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(yb)
        correct    += (logits.argmax(1) == yb).sum().item()
        n          += len(yb)
    return total_loss / n, correct / n

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        logits = model(Xb)
        loss   = criterion(logits, yb)
        total_loss += loss.item() * len(yb)
        correct    += (logits.argmax(1) == yb).sum().item()
        n          += len(yb)
    return total_loss / n, correct / n

def train_model(config, X_tr, y_tr, X_va, y_va, epochs=30, patience=5, verbose=True):
    """Train with early stopping, return best val accuracy and loss histories."""
    loader_tr = make_loader(X_tr, y_tr, config['batch_size'], shuffle=True)
    loader_va = make_loader(X_va, y_va, config['batch_size'], shuffle=False)

    model = MLP(
        input_dim    = X_tr.shape[1],
        hidden_sizes = config['hidden_sizes'],
        dropout_p    = config['dropout'],
        n_classes    = 10
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'],
                           weight_decay=config.get('weight_decay', 1e-4))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_state   = None
    no_improve   = 0
    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []

    for epoch in range(1, epochs+1):
        tl, ta = train_one_epoch(model, loader_tr, criterion, optimizer)
        vl, va = evaluate(model, loader_va, criterion)
        scheduler.step()

        train_losses.append(tl); val_losses.append(vl)
        train_accs.append(ta);   val_accs.append(va)

        if va > best_val_acc:
            best_val_acc = va
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve   = 0
        else:
            no_improve += 1

        if verbose and epoch % 5 == 0:
            print(f"  Epoch {epoch:3d}  train_loss={tl:.4f}  val_loss={vl:.4f}"
                  f"  train_acc={ta:.4f}  val_acc={va:.4f}")

        if no_improve >= patience:
            if verbose: print(f"  Early stopping at epoch {epoch}")
            break

    # Restore best weights
    model.load_state_dict(best_state)
    return model, best_val_acc, train_losses, val_losses, train_accs, val_accs

# ---- Hyperparameter grid ----
# We compare four configurations to justify our final choice.
configs = [
    {'name': 'Shallow-Small',  'hidden_sizes': [256],          'lr': 1e-3, 'batch_size': 128, 'dropout': 0.3},
    {'name': 'Deep-Medium',    'hidden_sizes': [512, 256],      'lr': 1e-3, 'batch_size': 128, 'dropout': 0.3},
    {'name': 'Deep-Large',     'hidden_sizes': [1024, 512, 256],'lr': 1e-3, 'batch_size': 128, 'dropout': 0.4},
    {'name': 'Deep-Large-LR',  'hidden_sizes': [1024, 512, 256],'lr': 3e-4, 'batch_size': 256, 'dropout': 0.4},
]

print("\n--- Hyperparameter Search ---")
results = {}
for cfg in configs:
    print(f"\nConfig: {cfg['name']}")
    model, best_va, tl, vl, ta, va = train_model(
        cfg, X_tr_n, y_tr_raw, X_va_n, y_va_raw,
        epochs=40, patience=8, verbose=True
    )
    results[cfg['name']] = {
        'model': model, 'val_acc': best_va,
        'train_losses': tl, 'val_losses': vl,
        'train_accs': ta, 'val_accs': va,
        'config': cfg
    }
    print(f"  => Best val acc: {best_va:.4f}")

# ---- Select best config ----
best_name = max(results, key=lambda k: results[k]['val_acc'])
best = results[best_name]
print(f"\nBest configuration: {best_name}  (val_acc = {best['val_acc']:.4f})")

# --- Figure 14: Training curves for all configs ---
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for i, (name, r) in enumerate(results.items()):
    ep = range(1, len(r['val_losses'])+1)
    axes[0].plot(ep, r['train_losses'], '-',  color=COLORS[i % len(COLORS)],
                 linewidth=1.5, label=f'{name} – Train', alpha=0.7)
    axes[0].plot(ep, r['val_losses'],   '--', color=COLORS[i % len(COLORS)],
                 linewidth=1.5, label=f'{name} – Val')
    axes[1].plot(ep, r['val_accs'],     '-',  color=COLORS[i % len(COLORS)],
                 linewidth=2,   label=name)

axes[0].set_xlabel('Epoch', fontsize=11); axes[0].set_ylabel('Cross-Entropy Loss', fontsize=11)
axes[0].set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=7, ncol=2)

axes[1].set_xlabel('Epoch', fontsize=11); axes[1].set_ylabel('Validation Accuracy', fontsize=11)
axes[1].set_title('Validation Accuracy by Configuration', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig('fig14_nn_training_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig14_nn_training_curves.png")

# --- Figure 15: Best model training curves ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ep = range(1, len(best['val_losses'])+1)
axes[0].plot(ep, best['train_losses'], '-',  color=COLORS[0], linewidth=2, label='Train')
axes[0].plot(ep, best['val_losses'],   '--', color=COLORS[1], linewidth=2, label='Validation')
axes[0].set_xlabel('Epoch', fontsize=11); axes[0].set_ylabel('Loss', fontsize=11)
axes[0].set_title(f'Best Model ({best_name}): Loss Curve', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=11)

axes[1].plot(ep, best['train_accs'], '-',  color=COLORS[0], linewidth=2, label='Train')
axes[1].plot(ep, best['val_accs'],   '--', color=COLORS[1], linewidth=2, label='Validation')
axes[1].set_xlabel('Epoch', fontsize=11); axes[1].set_ylabel('Accuracy', fontsize=11)
axes[1].set_title(f'Best Model ({best_name}): Accuracy Curve', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=11)
plt.tight_layout()
plt.savefig('fig15_best_model_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig15_best_model_curves.png")

# --- Figure 16: Config comparison bar chart ---
names   = list(results.keys())
val_accs_bar = [results[n]['val_acc'] for n in names]
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(names, val_accs_bar, color=COLORS[:len(names)], edgecolor='white', width=0.5)
for bar, v in zip(bars, val_accs_bar):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{v:.3f}', ha='center', va='bottom', fontsize=10)
ax.set_ylabel('Validation Accuracy', fontsize=12)
ax.set_title('Neural Network: Configuration Comparison', fontsize=13, fontweight='bold')
ax.set_ylim(0, max(val_accs_bar) * 1.1)
plt.xticks(rotation=15, fontsize=10)
plt.tight_layout()
plt.savefig('fig16_config_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig16_config_comparison.png")

# --- Figure 17: Confusion matrix for best model on validation ---
best_model = best['model']
best_model.eval()

@torch.no_grad()
def predict_all(model, X):
    Xt = torch.tensor(X, dtype=torch.float32)
    ds = TensorDataset(Xt)
    dl = DataLoader(ds, batch_size=512)
    preds = []
    for (Xb,) in dl:
        preds.append(model(Xb.to(device)).argmax(1).cpu().numpy())
    return np.concatenate(preds)

y_va_pred = predict_all(best_model, X_va_n)
cm = confusion_matrix(y_va_raw, y_va_pred)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=CIFAR_CLASSES, yticklabels=CIFAR_CLASSES,
            ax=ax, linewidths=0.3, annot_kws={"size": 8})
ax.set_xlabel('Predicted', fontsize=11)
ax.set_ylabel('True', fontsize=11)
ax.set_title(f'Neural Network Confusion Matrix\n{best_name} — Val Acc = {best["val_acc"]:.3f}',
             fontsize=12, fontweight='bold')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('fig17_nn_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig17_nn_confusion_matrix.png")

# =============================================================================
# TEST SET EVALUATION
# =============================================================================
print("\n--- FINAL TEST SET EVALUATION ---")
y_te_pred = predict_all(best_model, X_te_n)
test_acc = (y_te_pred == y_te_raw).mean()
print(f"Best NN ({best_name})  test_acc = {test_acc:.4f}")

print("\nClassification Report (Test):")
print(classification_report(y_te_raw, y_te_pred, target_names=CIFAR_CLASSES))

print("\n=== Part III Complete. All figures saved to  ===")
