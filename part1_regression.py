# =============================================================================
# PART I: Regression Task – Fuel Efficiency Prediction (Auto MPG)
# =============================================================================
# This script covers all required sections:
#   2.1 Exploratory Data Analysis
#   2.2 Preprocessing
#   2.3 Linear Regression Baseline
#   2.4 Polynomial Regression & Model Complexity
#   2.5 KNN Regression
#   2.6 Optimization Behavior
#   2.7 Model Comparison
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
COLORS = ['#2563EB', '#DC2626', '#16A34A', '#D97706', '#7C3AED']

# =============================================================================
# LOAD DATA
# =============================================================================
df = pd.read_csv('auto_mpg.csv')

print("Dataset shape:", df.shape)
print("\nMissing values per column:")
print(df.isnull().sum())
print("\nData types:")
print(df.dtypes)
print("\nBasic statistics:")
print(df.describe())

# =============================================================================
# 2.1 EXPLORATORY DATA ANALYSIS
# =============================================================================

# --- Figure 1: Distribution of target variable (mpg) ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df['mpg'], bins=25, color=COLORS[0], edgecolor='white', alpha=0.85)
axes[0].set_xlabel('MPG', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_title('Distribution of Target Variable (MPG)', fontsize=13, fontweight='bold')

from scipy import stats
stats.probplot(df['mpg'], dist="norm", plot=axes[1])
axes[1].set_title('Q-Q Plot of MPG', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('fig1_mpg_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig1_mpg_distribution.png")

# --- Figure 2: Feature vs target scatter plots ---
features = ['cylinders','displacement','horsepower','weight','acceleration','model_year','origin']
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()
for i, feat in enumerate(features):
    axes[i].scatter(df[feat], df['mpg'], alpha=0.4, color=COLORS[i % len(COLORS)], s=20)
    # Add linear trend line
    mask = df[feat].notna() & df['mpg'].notna()
    z = np.polyfit(df.loc[mask, feat], df.loc[mask, 'mpg'], 1)
    p = np.poly1d(z)
    xline = np.linspace(df[feat].min(), df[feat].max(), 100)
    axes[i].plot(xline, p(xline), color='black', linewidth=1.5, linestyle='--')
    axes[i].set_xlabel(feat, fontsize=10)
    axes[i].set_ylabel('MPG', fontsize=10)
    axes[i].set_title(f'MPG vs {feat}', fontsize=10, fontweight='bold')
axes[-1].axis('off')
plt.suptitle('Feature–Target Relationships', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('fig2_feature_target.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig2_feature_target.png")

# --- Figure 3: Correlation heatmap ---
fig, ax = plt.subplots(figsize=(9, 7))
corr = df.drop(columns=['origin']).corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, ax=ax,
            cbar_kws={'shrink': 0.8})
ax.set_title('Feature Correlation Matrix', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('fig3_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig3_correlation_heatmap.png")

# =============================================================================
# 2.2 PREPROCESSING
# =============================================================================
# Drop rows with missing horsepower values (only 6 rows)
df_clean = df.dropna().copy()
print(f"\nRows after dropping NAs: {len(df_clean)} (dropped {len(df) - len(df_clean)})")

# Features selected: all except origin (categorical with low cardinality — kept)
FEATURES = ['cylinders','displacement','horsepower','weight','acceleration','model_year','origin']
X = df_clean[FEATURES].values
y = df_clean['mpg'].values

# Train / Val / Test split: 70 / 15 / 15
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.15/0.85, random_state=42)

print(f"Train: {X_train.shape[0]}  Val: {X_val.shape[0]}  Test: {X_test.shape[0]}")

# Standardise features (required for gradient descent, KNN, and polynomial regression)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

def metrics(y_true, y_pred, label=''):
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    if label:
        print(f"  [{label}] MSE={mse:.3f}  RMSE={rmse:.3f}  MAE={mae:.3f}  R²={r2:.4f}")
    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}

# =============================================================================
# 2.3 LINEAR REGRESSION BASELINE
# =============================================================================
lr = LinearRegression()
lr.fit(X_train_s, y_train)
y_val_pred_lr = lr.predict(X_val_s)

print("\n--- Linear Regression ---")
lr_train_m = metrics(y_train, lr.predict(X_train_s), 'Train')
lr_val_m   = metrics(y_val,   y_val_pred_lr,          'Val  ')

# Feature coefficients
coef_df = pd.DataFrame({'Feature': FEATURES, 'Coefficient': lr.coef_})
print("\nCoefficients:")
print(coef_df.to_string(index=False))

# --- Figure 4: Prediction vs Ground Truth (Linear) ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
ax.scatter(y_val, y_val_pred_lr, alpha=0.6, color=COLORS[0], s=30, edgecolors='white', linewidth=0.3)
lims = [min(y_val.min(), y_val_pred_lr.min())-1, max(y_val.max(), y_val_pred_lr.max())+1]
ax.plot(lims, lims, 'k--', linewidth=1.5, label='Perfect prediction')
ax.set_xlabel('Ground Truth MPG', fontsize=11)
ax.set_ylabel('Predicted MPG', fontsize=11)
ax.set_title('Linear Regression: Predicted vs Actual', fontsize=12, fontweight='bold')
ax.legend()
ax.text(0.05, 0.92, f"R²={lr_val_m['r2']:.3f}\nRMSE={lr_val_m['rmse']:.3f}", 
        transform=ax.transAxes, fontsize=10,
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

# Residuals
residuals = y_val - y_val_pred_lr
axes[1].scatter(y_val_pred_lr, residuals, alpha=0.6, color=COLORS[1], s=30, edgecolors='white', linewidth=0.3)
axes[1].axhline(0, color='black', linewidth=1.5, linestyle='--')
axes[1].set_xlabel('Predicted MPG', fontsize=11)
axes[1].set_ylabel('Residual (Actual − Predicted)', fontsize=11)
axes[1].set_title('Linear Regression: Residual Plot', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('fig4_linear_regression.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig4_linear_regression.png")

# =============================================================================
# 2.4 POLYNOMIAL REGRESSION & MODEL COMPLEXITY
# =============================================================================
degrees = list(range(1, 9))
train_errors, val_errors = [], []

for deg in degrees:
    pipe = Pipeline([
        ('poly', PolynomialFeatures(degree=deg, include_bias=False)),
        ('scaler', StandardScaler()),
        ('lr', LinearRegression())
    ])
    pipe.fit(X_train_s, y_train)
    train_errors.append(np.sqrt(mean_squared_error(y_train, pipe.predict(X_train_s))))
    val_errors.append(np.sqrt(mean_squared_error(y_val, pipe.predict(X_val_s))))

best_deg = degrees[np.argmin(val_errors)]
print(f"\n--- Polynomial Regression ---")
print(f"Best degree by validation RMSE: {best_deg}")
for d, tr, va in zip(degrees, train_errors, val_errors):
    print(f"  deg={d}  train_RMSE={tr:.3f}  val_RMSE={va:.3f}")

# --- Figure 5: Model Complexity Curve ---
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(degrees, train_errors, 'o-', color=COLORS[0], label='Training RMSE', linewidth=2, markersize=6)
ax.plot(degrees, val_errors,   's-', color=COLORS[1], label='Validation RMSE', linewidth=2, markersize=6)
ax.axvline(best_deg, color=COLORS[2], linestyle='--', linewidth=1.8, label=f'Best degree = {best_deg}')
ax.set_xlabel('Polynomial Degree', fontsize=12)
ax.set_ylabel('RMSE', fontsize=12)
ax.set_title('Model Complexity Curve: Polynomial Regression', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.set_xticks(degrees)
plt.tight_layout()
plt.savefig('fig5_poly_complexity.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig5_poly_complexity.png")

# Best polynomial model
best_poly_pipe = Pipeline([
    ('poly', PolynomialFeatures(degree=best_deg, include_bias=False)),
    ('scaler', StandardScaler()),
    ('lr', LinearRegression())
])
best_poly_pipe.fit(X_train_s, y_train)
y_val_pred_poly = best_poly_pipe.predict(X_val_s)
print(f"\nBest Polynomial (deg={best_deg}) on Validation:")
poly_val_m = metrics(y_val, y_val_pred_poly, 'Val  ')

# =============================================================================
# 2.5 KNN REGRESSION
# =============================================================================
k_values = list(range(1, 31))
knn_train_errors, knn_val_errors = [], []

for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train_s, y_train)
    knn_train_errors.append(np.sqrt(mean_squared_error(y_train, knn.predict(X_train_s))))
    knn_val_errors.append(np.sqrt(mean_squared_error(y_val, knn.predict(X_val_s))))

best_k = k_values[np.argmin(knn_val_errors)]
print(f"\n--- KNN Regression ---")
print(f"Best k by validation RMSE: {best_k}")

# --- Figure 6: KNN error vs k ---
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(k_values, knn_train_errors, 'o-', color=COLORS[0], label='Training RMSE', linewidth=2, markersize=5)
ax.plot(k_values, knn_val_errors,   's-', color=COLORS[1], label='Validation RMSE', linewidth=2, markersize=5)
ax.axvline(best_k, color=COLORS[2], linestyle='--', linewidth=1.8, label=f'Best k = {best_k}')
ax.set_xlabel('Number of Neighbors (k)', fontsize=12)
ax.set_ylabel('RMSE', fontsize=12)
ax.set_title('KNN Regression: Error vs k', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('fig6_knn_error_vs_k.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig6_knn_error_vs_k.png")

best_knn = KNeighborsRegressor(n_neighbors=best_k)
best_knn.fit(X_train_s, y_train)
y_val_pred_knn = best_knn.predict(X_val_s)

print(f"\nKNN (k={best_k}) on Validation:")
knn_val_m = metrics(y_val, y_val_pred_knn, 'Val  ')

# --- Figure 7: KNN Predicted vs Actual ---
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(y_val, y_val_pred_knn, alpha=0.6, color=COLORS[3], s=30, edgecolors='white', linewidth=0.3)
lims = [min(y_val.min(), y_val_pred_knn.min())-1, max(y_val.max(), y_val_pred_knn.max())+1]
ax.plot(lims, lims, 'k--', linewidth=1.5, label='Perfect prediction')
ax.set_xlabel('Ground Truth MPG', fontsize=11)
ax.set_ylabel('Predicted MPG', fontsize=11)
ax.set_title(f'KNN (k={best_k}): Predicted vs Actual', fontsize=12, fontweight='bold')
ax.text(0.05, 0.92, f"R²={knn_val_m['r2']:.3f}\nRMSE={knn_val_m['rmse']:.3f}",
        transform=ax.transAxes, fontsize=10,
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))
ax.legend()
plt.tight_layout()
plt.savefig('fig7_knn_pred_vs_actual.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig7_knn_pred_vs_actual.png")

# =============================================================================
# 2.6 OPTIMIZATION BEHAVIOR
# =============================================================================
# Compare Batch GD (LinearRegression, analytical) vs SGD vs Mini-batch SGD
# We simulate batch/mini-batch by recording loss per epoch manually

def run_sgd_training(X_tr, y_tr, X_va, y_va, batch_size, lr_rate=0.01, epochs=200, seed=42):
    """Train a linear model with gradient descent, tracking train/val loss per epoch."""
    np.random.seed(seed)
    n, d = X_tr.shape
    w = np.zeros(d)
    b = 0.0
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        idx = np.random.permutation(n)
        X_sh, y_sh = X_tr[idx], y_tr[idx]
        for start in range(0, n, batch_size):
            Xb = X_sh[start:start+batch_size]
            yb = y_sh[start:start+batch_size]
            pred = Xb @ w + b
            err  = pred - yb
            grad_w = (2/len(Xb)) * Xb.T @ err
            grad_b = (2/len(Xb)) * err.sum()
            w -= lr_rate * grad_w
            b -= lr_rate * grad_b

        train_pred = X_tr @ w + b
        val_pred   = X_va @ w + b
        train_losses.append(mean_squared_error(y_tr, train_pred))
        val_losses.append(mean_squared_error(y_va, val_pred))

    return train_losses, val_losses

print("\n--- Optimization Behavior ---")
epochs = 200
# Batch GD (batch_size = full dataset)
tl_batch, vl_batch = run_sgd_training(X_train_s, y_train, X_val_s, y_val,
                                       batch_size=len(X_train_s), lr_rate=0.05, epochs=epochs)
# Mini-batch GD (batch_size = 32)
tl_mini, vl_mini = run_sgd_training(X_train_s, y_train, X_val_s, y_val,
                                     batch_size=32, lr_rate=0.05, epochs=epochs)
# SGD (batch_size = 1)
tl_sgd, vl_sgd = run_sgd_training(X_train_s, y_train, X_val_s, y_val,
                                   batch_size=1, lr_rate=0.005, epochs=epochs)

print(f"  Batch GD    final val MSE: {vl_batch[-1]:.3f}")
print(f"  Mini-batch  final val MSE: {vl_mini[-1]:.3f}")
print(f"  SGD         final val MSE: {vl_sgd[-1]:.3f}")

# --- Figure 8: Loss curves ---
ep = list(range(1, epochs+1))
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, tl, vl, label, c in [
    (axes[0], tl_batch, vl_batch, 'Batch GD',    COLORS[0]),
    (axes[0], tl_mini,  vl_mini,  'Mini-batch (32)', COLORS[1]),
    (axes[0], tl_sgd,   vl_sgd,   'SGD (1)',     COLORS[2]),
]:
    ax.plot(ep, tl, '-',  color=c, linewidth=1.5, label=f'{label} Train')
    ax.plot(ep, vl, '--', color=c, linewidth=1.5, label=f'{label} Val')

axes[0].set_xlabel('Epoch', fontsize=11)
axes[0].set_ylabel('MSE Loss', fontsize=11)
axes[0].set_title('All Optimizers: Train & Val Loss', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=8)

# Zoom into mini-batch vs batch (most interesting comparison)
axes[1].plot(ep, tl_batch, '-',  color=COLORS[0], linewidth=2, label='Batch GD – Train')
axes[1].plot(ep, vl_batch, '--', color=COLORS[0], linewidth=2, label='Batch GD – Val')
axes[1].plot(ep, tl_mini,  '-',  color=COLORS[1], linewidth=2, label='Mini-batch – Train')
axes[1].plot(ep, vl_mini,  '--', color=COLORS[1], linewidth=2, label='Mini-batch – Val')
axes[1].set_xlabel('Epoch', fontsize=11)
axes[1].set_ylabel('MSE Loss', fontsize=11)
axes[1].set_title('Batch GD vs Mini-batch GD', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=9)
plt.tight_layout()
plt.savefig('fig8_optimization_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig8_optimization_curves.png")

# =============================================================================
# 2.7 MODEL COMPARISON & TEST SET EVALUATION
# =============================================================================
print("\n--- Model Comparison (Validation Set) ---")
results = {
    'Linear Regression':         metrics(y_val, y_val_pred_lr,   'Linear Reg  '),
    f'Polynomial (deg={best_deg})': metrics(y_val, y_val_pred_poly, f'Poly deg={best_deg}  '),
    f'KNN (k={best_k})':         metrics(y_val, y_val_pred_knn,  f'KNN k={best_k}     '),
}

# Best model on validation is chosen for test set evaluation
# Determine winner
best_model_name = min(results, key=lambda k: results[k]['rmse'])
print(f"\nBest model by val RMSE: {best_model_name}")

# Evaluate on TEST SET (only done once, at the very end)
if 'KNN' in best_model_name:
    y_test_pred = best_knn.predict(X_test_s)
elif 'Poly' in best_model_name:
    y_test_pred = best_poly_pipe.predict(X_test_s)
else:
    y_test_pred = lr.predict(X_test_s)

print("\n--- FINAL TEST SET EVALUATION ---")
test_m = metrics(y_test, y_test_pred, best_model_name)

# --- Figure 9: Bar chart comparison ---
model_names  = list(results.keys())
rmse_vals    = [results[m]['rmse'] for m in model_names]
r2_vals      = [results[m]['r2']   for m in model_names]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
bars = axes[0].bar(model_names, rmse_vals, color=COLORS[:3], edgecolor='white', width=0.5)
axes[0].set_ylabel('RMSE (lower is better)', fontsize=11)
axes[0].set_title('Validation RMSE by Model', fontsize=12, fontweight='bold')
for bar, val in zip(bars, rmse_vals):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=10)

bars2 = axes[1].bar(model_names, r2_vals, color=COLORS[:3], edgecolor='white', width=0.5)
axes[1].set_ylabel('R² (higher is better)', fontsize=11)
axes[1].set_title('Validation R² by Model', fontsize=12, fontweight='bold')
for bar, val in zip(bars2, r2_vals):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('fig9_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig9_model_comparison.png")

print("\n=== Part I Complete. All figures saved to  ===")
