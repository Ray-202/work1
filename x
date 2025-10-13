
# Iris Moving Mean Feature Engineering Notebook

This notebook demonstrates how to use Python's equivalent of MATLAB's `movmean()` for feature engineering with the Iris dataset.

---

## Cell 1: Import Libraries

```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("Libraries imported successfully!")
```

---

## Cell 2: Load the Iris Dataset

```python
print("Loading Iris Dataset...")
iris = load_iris()

# Create a DataFrame for easier manipulation
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:\n{df.head()}")
print(f"\nTarget classes: {iris.target_names}")
```

---

## Cell 3: Define Function to Apply Moving Mean

```python
def apply_moving_mean(data, window_size):
    """
    Apply moving mean to create new features.
    Python equivalent of MATLAB's movmean(x, k, dim)
    
    Parameters:
    -----------
    data : DataFrame
        Input data with features
    window_size : int
        Window size for moving average (k in MATLAB's movmean)
    
    Returns:
    --------
    DataFrame with moving mean features
    """
    feature_cols = [col for col in data.columns if col != 'target']
    
    # Apply rolling mean to each feature column
    # center=True mimics MATLAB's default behavior (centered window)
    # min_periods=1 handles edges like MATLAB (uses available data at boundaries)
    movmean_data = data[feature_cols].rolling(
        window=window_size, 
        center=True, 
        min_periods=1
    ).mean()
    
    # Add suffix to indicate these are moving mean features
    movmean_data.columns = [f'{col}_movmean_w{window_size}' for col in feature_cols]
    
    return movmean_data

print("Moving mean function defined!")
print("This is Python's equivalent to MATLAB's movmean(x, k, dim)")
```

---

## Cell 4: Test Moving Mean on Sample Data

```python
# Let's see what moving mean does to the first feature
sample_window = 5
movmean_sample = apply_moving_mean(df, sample_window)

print(f"\nExample: Original vs Moving Mean (window={sample_window})")
print(f"First 10 rows of '{df.columns[0]}':")
comparison = pd.DataFrame({
    'Original': df[df.columns[0]].head(10),
    f'MovMean_w{sample_window}': movmean_sample.iloc[:10, 0]
})
print(comparison)
```

---

## Cell 5: Define Evaluation Function

```python
def evaluate_window_size(df, window_size, use_original_features=False):
    """
    Create moving mean features and evaluate Naive Bayes classifier.
    
    Parameters:
    -----------
    df : DataFrame
        Original dataset
    window_size : int
        Window size for moving mean
    use_original_features : bool
        If True, combine original + moving mean features
        If False, use only moving mean features
    
    Returns:
    --------
    accuracy, y_test, y_pred, n_features
    """
    # Create moving mean features
    movmean_features = apply_moving_mean(df, window_size)
    
    # Decide which features to use
    if use_original_features:
        # Combine original and moving mean features
        X = pd.concat([df.drop('target', axis=1), movmean_features], axis=1)
    else:
        # Use only moving mean features
        X = movmean_features
    
    y = df['target']
    
    # Split data: 70% train, 30% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train Gaussian Naive Bayes classifier
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    
    # Make predictions
    y_pred = gnb.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy, y_test, y_pred, X.shape[1]

print("Evaluation function defined!")
```

---

## Cell 6: Experiment with Different Window Sizes

```python
print("="*70)
print("EXPERIMENTING WITH DIFFERENT WINDOW SIZES")
print("="*70)

# Test different window sizes
window_sizes = [3, 5, 7, 9, 11, 15, 20]
results = []

for window_size in window_sizes:
    accuracy, y_test, y_pred, n_features = evaluate_window_size(
        df, window_size, use_original_features=False
    )
    results.append({
        'window_size': window_size,
        'accuracy': accuracy,
        'n_features': n_features
    })
    print(f"Window Size: {window_size:2d} | Accuracy: {accuracy:.4f} | Features: {n_features}")

results_df = pd.DataFrame(results)
print("\n" + "="*70)
```

---

## Cell 7: Visualize Results

```python
plt.figure(figsize=(10, 6))
plt.plot(results_df['window_size'], results_df['accuracy'], 
         marker='o', linewidth=2, markersize=8, color='#4F46E5')
plt.xlabel('Window Size', fontsize=12)
plt.ylabel('Classification Accuracy', fontsize=12)
plt.title('Naive Bayes Accuracy vs Moving Mean Window Size', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xticks(window_sizes)
plt.ylim([0.8, 1.0])
plt.tight_layout()
plt.show()

print("Visualization complete!")
```

---

## Cell 8: Find and Analyze Best Window Size

```python
best_window = results_df.loc[results_df['accuracy'].idxmax(), 'window_size']
best_accuracy = results_df['accuracy'].max()

print("="*70)
print(f"BEST WINDOW SIZE: {int(best_window)}")
print(f"BEST ACCURACY: {best_accuracy:.4f}")
print("="*70)

# Get detailed results for best window
accuracy, y_test, y_pred, n_features = evaluate_window_size(
    df, int(best_window), use_original_features=False
)

print(f"\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

---

## Cell 9: Confusion Matrix for Best Window Size

```python
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris.target_names, 
            yticklabels=iris.target_names)
plt.title(f'Confusion Matrix (Window Size = {int(best_window)})', fontsize=14)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.show()
```

---

## Cell 10: Compare with Baseline (Original Features)

```python
print("="*70)
print("BASELINE: Using Original Features (No Moving Mean)")
print("="*70)

X_original = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(
    X_original, y, test_size=0.3, random_state=42, stratify=y
)

gnb_baseline = GaussianNB()
gnb_baseline.fit(X_train, y_train)
y_pred_baseline = gnb_baseline.predict(X_test)
baseline_accuracy = accuracy_score(y_test, y_pred_baseline)

print(f"\nBaseline Accuracy (original features): {baseline_accuracy:.4f}")
print(f"Best Moving Mean Accuracy (window={int(best_window)}): {best_accuracy:.4f}")
print(f"Difference: {best_accuracy - baseline_accuracy:+.4f}")

if best_accuracy > baseline_accuracy:
    print(f"\n✓ Moving mean features IMPROVED classification!")
else:
    print(f"\n✗ Original features performed better")
```

---

## Cell 11 (OPTIONAL): Load Your Own CSV File Instead

```python
# If you want to use your own Iris CSV file, uncomment and modify this:

# df = pd.read_csv('your_iris_file.csv')
# # Make sure the last column is named 'target' or adjust accordingly
# # If your target column has a different name, rename it:
# # df = df.rename(columns={'species': 'target'})
# 
# print(f"Loaded CSV with shape: {df.shape}")
# print(df.head())
```

---

## Summary

This notebook demonstrates:
- **Moving Mean**: Python's `pandas.rolling()` = MATLAB's `movmean()`
- **Feature Engineering**: Creating smoothed features with different window sizes
- **Classification**: Using Gaussian Naive Bayes to evaluate feature quality
- **Comparison**: Finding the optimal window size for best accuracy

Copy each cell's code into your Jupyter notebook and run sequentially!
