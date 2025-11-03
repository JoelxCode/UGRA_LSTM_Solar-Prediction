# Sequential time-based experiment (no MONTH column needed)
import time
from datetime import datetime

print("="*60)
print("SEQUENTIAL TIME-BASED EXPERIMENT: 12 chronological splits")
print("No month column required - uses chronological order")
print("="*60)

# Create sequences (no months array needed)
seq_len = 24
X, y_seq = [], []

for i in range(seq_len, len(solar_ydata)):
    X.append(solar_Xdata[i-seq_len:i, :])
    y_seq.append(solar_ydata[i, 0])

X = np.array(X)
y = np.array(y_seq)

print(f'Total sequences created: {X.shape[0]}')

# Create 12 chronological splits
n_splits = 12
split_size = len(X) // n_splits
all_results = {}

print(f"Split size: ~{split_size} samples per split")
print(f"\nStarting {n_splits}-split experiment (Epochs: {setEpoch})...")

# Run experiment for each split
for split_idx in range(n_splits):
    print(f"\nExperiment {split_idx+1}/{n_splits}: Testing split {split_idx+1}...", end=" ")
    
    start_time = time.time()
    
    # Create chronological split
    test_start = split_idx * split_size
    test_end = test_start + split_size if split_idx < n_splits-1 else len(X)
    
    test_indices = list(range(test_start, test_end))
    train_indices = list(range(0, test_start)) + list(range(test_end, len(X)))
    
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    
    if len(X_test) < 10:
        print(f"SKIPPED (only {len(X_test)} samples)")
        continue
    
    # Build model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_absolute_error')
    
    # Train model
    history = model.fit(X_train, y_train, epochs=setEpoch, batch_size=batch_size, 
                       validation_split=0.1, verbose=0)
    
    # Make predictions
    y_pred = model.predict(X_test, verbose=0)
    y_train_pred = model.predict(X_train, verbose=0)
    
    # Calculate metrics
    test_mae = mean_absolute_error(y_test, y_pred.flatten())
    train_mae = mean_absolute_error(y_train, y_train_pred.flatten())
    
    training_time = time.time() - start_time
    
    # Store results
    all_results[split_idx+1] = {
        'split_name': f'Split_{split_idx+1}',
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'train_mae': float(train_mae),
        'test_mae': float(test_mae),
        'training_time': training_time
    }
    
    print(f"Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f} ({training_time:.1f}s)")
    
    # Clear memory
    del model
    import gc
    gc.collect()

# Display results
print(f"\n" + "="*70)
print("EXPERIMENT RESULTS SUMMARY:")
print("="*70)
print(f"{'Split':<8} {'Train MAE':<12} {'Test MAE':<12} {'Ratio':<8} {'Samples':<10} {'Time(s)':<8}")
print("-" * 70)

mae_results = []
for split_idx in range(1, n_splits+1):
    if split_idx in all_results:
        r = all_results[split_idx]
        ratio = r['test_mae'] / r['train_mae']
        mae_results.append((split_idx, r['split_name'], r['train_mae'], r['test_mae'], ratio))
        print(f"{r['split_name']:<8} {r['train_mae']:<12.4f} {r['test_mae']:<12.4f} {ratio:<8.2f} {r['test_samples']:<10} {r['training_time']:<8.1f}")

# Find best and worst
mae_results.sort(key=lambda x: x[3])
best_split = mae_results[0]
worst_split = mae_results[-1]

print("=" * 70)
print(f"BEST PERFORMANCE:  {best_split[1]} (Test MAE: {best_split[3]:.4f})")
print(f"WORST PERFORMANCE: {worst_split[1]} (Test MAE: {worst_split[3]:.4f})")
print(f"AVERAGE TEST MAE:  {np.mean([r[3] for r in mae_results]):.4f}")

# Save results
results_df = pd.DataFrame([
    {'Split': r[1], 'Train_MAE': r[2], 'Test_MAE': r[3], 'Ratio': r[4]} 
    for r in mae_results
])

csv_path = os.path.join(os.getcwd(), 'sequential_mae_results.csv')
results_df.to_csv(csv_path, index=False)
print(f"\n✓ Results saved to: {csv_path}")

# Update main results
results["experiment_type"] = "12_sequential_splits"
results["best_split"] = best_split[1]
results["worst_split"] = worst_split[1] 
results["best_test_mae"] = best_split[3]
results["worst_test_mae"] = worst_split[3]
results["avg_test_mae"] = np.mean([r[3] for r in mae_results])

print(f"\nExperiment completed!")