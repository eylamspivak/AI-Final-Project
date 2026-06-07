# =============================================================================
# PART II: Classical Classification on CIFAR-10
# =============================================================================
# Sections:
#   3.1 Baseline Models: Logistic Regression, Linear SVM, KNN
#   3.2 Hyperparameter Selection with validation plots
#   3.3 Evaluation: accuracy, confusion matrices, analysis
# =============================================================================

import numpy as np
import pickle, os, struct
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                              classification_report, ConfusionMatrixDisplay)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
COLORS = ['#2563EB', '#DC2626', '#16A34A', '#D97706', '#7C3AED']

CIFAR_CLASSES = ['airplane','automobile','bird','cat','deer',
                 'dog','frog','horse','ship','truck']

# =============================================================================
# LOAD CIFAR-10
# =============================================================================
CIFAR_PATH = r"C:\Users\Eilam\Downloads\cifar-10-python\cifar-10-batches-py"

def load_cifar10_batch(filepath):
    with open(filepath, 'rb') as f:
        d = pickle.load(f, encoding='bytes')
    X = d[b'data'].astype(np.float32)   # (10000, 3072)
    y = np.array(d[b'labels'])
    return X, y

def load_cifar10(data_dir):
    Xs, ys = [], []
    for i in range(1, 6):
        path = os.path.join(data_dir, f'data_batch_{i}')
        X, y = load_cifar10_batch(path)
        Xs.append(X); ys.append(y)
    X_train = np.concatenate(Xs)
    y_train = np.concatenate(ys)
    test_path = os.path.join(data_dir, 'test_batch')
    X_test, y_test = load_cifar10_batch(test_path)
    return X_train, y_train, X_test, y_test

print("Loading CIFAR-10...")
try:
    X_train_full, y_train_full, X_test_full, y_test_full = load_cifar10(CIFAR_PATH)
    print(f"Train: {X_train_full.shape}  Test: {X_test_full.shape}")
    DATA_LOADED = True
except Exception as e:
    print(f"Could not load CIFAR-10 from disk: {e}")
    print("Generating synthetic stand-in data for demonstration...")
    # Synthetic data with same shape — for running the pipeline without the files
    np.random.seed(42)
    N_TR, N_TE = 50000, 10000
    X_train_full = np.random.rand(N_TR, 3072).astype(np.float32) * 255
    y_train_full = np.random.randint(0, 10, N_TR)
    X_test_full  = np.random.rand(N_TE, 3072).astype(np.float32) * 255
    y_test_full  = np.random.randint(0, 10, N_TE)
    DATA_LOADED = False
    print("NOTE: Running on synthetic data. Replace with real CIFAR-10 for actual results.")

# =============================================================================
# SUBSAMPLE for tractability (classical ML on 50k×3072 is very slow)
# Use 10 000 training samples and 2 000 validation samples
# The test set uses 5000 samples (representative)
# =============================================================================
TRAIN_SIZE = 10000
VAL_SIZE   = 2000
TEST_SIZE  = 5000

np.random.seed(42)
idx_tr = np.random.choice(len(X_train_full), TRAIN_SIZE + VAL_SIZE, replace=False)
X_tv = X_train_full[idx_tr]
y_tv = y_train_full[idx_tr]

X_tr_raw, X_va_raw = X_tv[:TRAIN_SIZE], X_tv[TRAIN_SIZE:]
y_tr, y_va = y_tv[:TRAIN_SIZE], y_tv[TRAIN_SIZE:]

idx_te = np.random.choice(len(X_test_full), TEST_SIZE, replace=False)
X_te_raw = X_test_full[idx_te]
y_te     = y_test_full[idx_te]

print(f"Subsampled — Train: {X_tr_raw.shape}  Val: {X_va_raw.shape}  Test: {X_te_raw.shape}")

# =============================================================================
# PREPROCESSING: Normalise + PCA dimensionality reduction
# =============================================================================
# Normalise pixels to [0,1]
X_tr_raw = X_tr_raw / 255.0
X_va_raw = X_va_raw / 255.0
X_te_raw = X_te_raw / 255.0

# Standardise (zero mean, unit variance) — critical for SVM and Logistic Regression
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr_raw)
X_va_s = scaler.transform(X_va_raw)
X_te_s = scaler.transform(X_te_raw)

# PCA: reduce 3072 → 200 dimensions (retains ~80% of variance, makes KNN tractable)
print("Fitting PCA (n_components=200)...")
pca = PCA(n_components=200, random_state=42)
X_tr = pca.fit_transform(X_tr_s)
X_va = pca.transform(X_va_s)
X_te = pca.transform(X_te_s)
print(f"Explained variance (200 PCs): {pca.explained_variance_ratio_.sum()*100:.1f}%")
print(f"Post-PCA shapes — Train: {X_tr.shape}  Val: {X_va.shape}  Test: {X_te.shape}")

# --- Figure 10: Explained variance curve ---
fig, ax = plt.subplots(figsize=(9, 4))
cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
ax.plot(range(1, len(cumvar)+1), cumvar, color=COLORS[0], linewidth=2)
ax.axhline(80, color=COLORS[1], linestyle='--', linewidth=1.5, label='80% threshold')
ax.axhline(90, color=COLORS[2], linestyle='--', linewidth=1.5, label='90% threshold')
ax.set_xlabel('Number of Principal Components', fontsize=12)
ax.set_ylabel('Cumulative Explained Variance (%)', fontsize=12)
ax.set_title('PCA: Cumulative Explained Variance', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('fig10_pca_variance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig10_pca_variance.png")

# =============================================================================
# 3.2 HYPERPARAMETER SELECTION
# =============================================================================

# --- Logistic Regression: regularisation strength C ---
print("\n--- Logistic Regression hyperparameter search ---")
C_values = [0.001, 0.01, 0.1, 1.0, 10.0]
lr_val_accs = []
for C in C_values:
    clf = LogisticRegression(C=C, max_iter=1000, solver='saga',
                             random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    acc = accuracy_score(y_va, clf.predict(X_va))
    lr_val_accs.append(acc)
    print(f"  C={C:6.3f}  val_acc={acc:.4f}")

best_C = C_values[np.argmax(lr_val_accs)]
print(f"Best C: {best_C}")

# --- SVM: regularisation C ---
print("\n--- Linear SVM hyperparameter search ---")
svm_C_values = [0.001, 0.01, 0.1, 1.0, 5.0]
svm_val_accs = []
for C in svm_C_values:
    clf = LinearSVC(C=C, max_iter=2000, random_state=42)
    clf.fit(X_tr, y_tr)
    acc = accuracy_score(y_va, clf.predict(X_va))
    svm_val_accs.append(acc)
    print(f"  C={C:6.3f}  val_acc={acc:.4f}")

best_svm_C = svm_C_values[np.argmax(svm_val_accs)]
print(f"Best SVM C: {best_svm_C}")

# --- KNN: k ---
print("\n--- KNN hyperparameter search ---")
k_values = [1, 3, 5, 7, 10, 15, 20, 30]
knn_val_accs = []
for k in k_values:
    clf = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    acc = accuracy_score(y_va, clf.predict(X_va))
    knn_val_accs.append(acc)
    print(f"  k={k:3d}  val_acc={acc:.4f}")

best_k = k_values[np.argmax(knn_val_accs)]
print(f"Best k: {best_k}")

# --- Figure 11: Hyperparameter selection plots ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].semilogx(C_values, lr_val_accs, 'o-', color=COLORS[0], linewidth=2, markersize=8)
axes[0].axvline(best_C, color=COLORS[1], linestyle='--', linewidth=1.8, label=f'Best C={best_C}')
axes[0].set_xlabel('Regularisation C', fontsize=11)
axes[0].set_ylabel('Validation Accuracy', fontsize=11)
axes[0].set_title('Logistic Regression: C Selection', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10)

axes[1].semilogx(svm_C_values, svm_val_accs, 's-', color=COLORS[2], linewidth=2, markersize=8)
axes[1].axvline(best_svm_C, color=COLORS[1], linestyle='--', linewidth=1.8, label=f'Best C={best_svm_C}')
axes[1].set_xlabel('Regularisation C', fontsize=11)
axes[1].set_ylabel('Validation Accuracy', fontsize=11)
axes[1].set_title('Linear SVM: C Selection', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=10)

axes[2].plot(k_values, knn_val_accs, '^-', color=COLORS[3], linewidth=2, markersize=8)
axes[2].axvline(best_k, color=COLORS[1], linestyle='--', linewidth=1.8, label=f'Best k={best_k}')
axes[2].set_xlabel('Number of Neighbors k', fontsize=11)
axes[2].set_ylabel('Validation Accuracy', fontsize=11)
axes[2].set_title('KNN: k Selection', fontsize=12, fontweight='bold')
axes[2].legend(fontsize=10)

plt.suptitle('Hyperparameter Selection (Validation Accuracy)', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('fig11_hyperparameter_selection.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig11_hyperparameter_selection.png")

# =============================================================================
# 3.1 / 3.3  TRAIN FINAL MODELS & EVALUATE
# =============================================================================
print("\n--- Training final models ---")

lr_clf = LogisticRegression(C=best_C, max_iter=1000, solver='saga',
                             random_state=42, n_jobs=-1)
lr_clf.fit(X_tr, y_tr)

svm_clf = LinearSVC(C=best_svm_C, max_iter=2000, random_state=42)
svm_clf.fit(X_tr, y_tr)

knn_clf = KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1)
knn_clf.fit(X_tr, y_tr)

models = {
    'Logistic Regression': lr_clf,
    'Linear SVM':          svm_clf,
    f'KNN (k={best_k})':   knn_clf,
}

print("\n--- Validation Results ---")
val_accs = {}
for name, clf in models.items():
    acc = accuracy_score(y_va, clf.predict(X_va))
    val_accs[name] = acc
    print(f"  {name:25s}  val_acc={acc:.4f}")

# --- Figure 12: Confusion matrices ---
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
for ax, (name, clf) in zip(axes, models.items()):
    y_pred = clf.predict(X_va)
    cm = confusion_matrix(y_va, y_pred)
    # Normalise by row (true label)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=CIFAR_CLASSES, yticklabels=CIFAR_CLASSES,
                ax=ax, linewidths=0.3, cbar_kws={'shrink': 0.8},
                annot_kws={"size": 7})
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('True', fontsize=10)
    ax.set_title(f'{name}\nVal Acc = {val_accs[name]:.3f}', fontsize=11, fontweight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', rotation=0,  labelsize=8)

plt.suptitle('Normalised Confusion Matrices (Validation Set)', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('fig12_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig12_confusion_matrices.png")

# Per-class F1 for best model on validation
best_val_name = max(val_accs, key=val_accs.get)
best_clf      = models[best_val_name]
print(f"\nBest model: {best_val_name}")
print("\nClassification Report (Validation):")
print(classification_report(y_va, best_clf.predict(X_va), target_names=CIFAR_CLASSES))

# --- Figure 13: Per-class accuracy bar chart ---
cm_best = confusion_matrix(y_va, best_clf.predict(X_va))
per_class_acc = cm_best.diagonal() / cm_best.sum(axis=1)

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(CIFAR_CLASSES, per_class_acc, color=COLORS[0], edgecolor='white', alpha=0.85)
ax.axhline(per_class_acc.mean(), color=COLORS[1], linestyle='--', linewidth=1.8,
           label=f'Mean = {per_class_acc.mean():.3f}')
ax.set_ylabel('Per-class Accuracy', fontsize=12)
ax.set_title(f'Per-class Accuracy — {best_val_name}', fontsize=13, fontweight='bold')
ax.set_ylim(0, 1)
ax.legend(fontsize=11)
for bar, val in zip(bars, per_class_acc):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.2f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig('fig13_per_class_accuracy.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig13_per_class_accuracy.png")

# =============================================================================
# TEST SET EVALUATION (only for selected best model)
# =============================================================================
print("\n--- FINAL TEST SET EVALUATION ---")
test_acc = accuracy_score(y_te, best_clf.predict(X_te))
print(f"  {best_val_name}  test_acc = {test_acc:.4f}")

print("\n=== Part II Complete. All figures saved to  ===")
