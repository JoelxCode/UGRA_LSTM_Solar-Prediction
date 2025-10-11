# Create a time series plot with all irradiance features on the same graph
plt.figure(figsize=(15, 8))

for i, col in enumerate(available_cols):
    if col in df.columns:
        plt.plot(df.index, df[col], label=col, color=colors[i], alpha=0.7, linewidth=1)

plt.title('All Irradiance Features - Time Series Comparison', fontsize=16, fontweight='bold')
plt.xlabel('Time Index', fontsize=12)
plt.ylabel('Irradiance (W/m²)', fontsize=12)
plt.legend(loc='upper right', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()