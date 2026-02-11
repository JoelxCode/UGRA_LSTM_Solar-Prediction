# Seasonal analysis visualization
import matplotlib.pyplot as plt

# Plot monthly power distributions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Monthly averages
monthly_avg = []
monthly_counts = []
for month in range(1, 13):
    month_mask = (months == month)
    if np.any(month_mask):
        monthly_avg.append(y[month_mask].mean())
        monthly_counts.append(np.sum(month_mask))
    else:
        monthly_avg.append(0)
        monthly_counts.append(0)

# Bar plot of monthly averages
colors = ['red' if m==TEST_MONTH else 'blue' for m in range(1, 13)]
bars = ax1.bar(range(1, 13), monthly_avg, color=colors)
ax1.set_xlabel('Month')
ax1.set_ylabel('Average Power')
ax1.set_title(f'Monthly Average Power (January in red)')
ax1.set_xticks(range(1, 13))
ax1.grid(True, alpha=0.3)

# Add count labels on bars
for i, (bar, count) in enumerate(zip(bars, monthly_counts)):
    if count > 0:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{count}', ha='center', va='bottom', fontsize=8)

# Power distribution comparison
ax2.hist(y_train, bins=50, alpha=0.7, label=f'Training (11 months)', density=True, color='blue')
ax2.hist(y_test, bins=50, alpha=0.7, label=f'Test (January)', density=True, color='red')
ax2.set_xlabel('Power')
ax2.set_ylabel('Density')
ax2.set_title('Power Distribution: Train vs Test')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print challenge assessment
if test_power_mean < train_power_mean * 0.5:
    print("⚠️  CHALLENGE: January has much lower power than training months (winter effect)")
elif test_power_mean > train_power_mean * 1.5:
    print("⚠️  CHALLENGE: January has much higher power than training months")
else:
    print("✓ MODERATE: January power levels similar to training months")

# Create sequences for LSTM with month-based split
# Test on January (month 1), train on other 11 months
TEST_MONTH = 1  # January

seq_len = 24
X, y_seq = [], []
months = []  # track month for each sequence

for i in range(seq_len, len(solar_ydata)):
    X.append(solar_Xdata[i-seq_len:i, :])
    y_seq.append(solar_ydata[i, 0])
    months.append(df.iloc[i]['MONTH'])  # track month for this sequence

X = np.array(X)
y = np.array(y_seq)
months = np.array(months)

print('X shape:', X.shape)
print('y shape:', y.shape)

# Split by month: January for testing, other 11 months for training
test_mask = (months == TEST_MONTH)
train_mask = ~test_mask

X_train = X[train_mask]
y_train = y[train_mask]
X_test = X[test_mask]
y_test = y[test_mask]

print(f'Training on 11 months (excluding January): {X_train.shape[0]} samples')
print(f'Testing on January: {X_test.shape[0]} samples')
print(f'Train/test ratio: {X_train.shape[0]/(X_train.shape[0]+X_test.shape[0]):.2%}')

# Show month distribution in training set
print(f'\nMonth distribution in training set:')
train_months = months[train_mask]
for month in range(1, 13):
    if month != TEST_MONTH:
        count = np.sum(train_months == month)
        print(f'  Month {month:2d}: {count:4d} samples')

# Show seasonal analysis
test_power_mean = y_test.mean()
train_power_mean = y_train.mean()
print(f'\nSeasonal analysis:')
print(f'Average power - January (test): {test_power_mean:.2f}')
print(f'Average power - Other months (train): {train_power_mean:.2f}')
print(f'Power difference: {test_power_mean - train_power_mean:.2f}')