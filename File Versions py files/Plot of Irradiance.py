import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import matplotlib.dates as mdates

# Read the data
df = pd.read_csv('BigData.csv')

# Display basic info about the dataset
print("Dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())
print("\nFirst few rows:")
print(df.head())

# Create a proper datetime column from MONTH, DAY, HOUR, MINUTE
# Assuming year 2020 (you can change this if needed)
df['datetime'] = pd.to_datetime({
    'year': 2020,  # You can change this to the actual year
    'month': df['MONTH'],
    'day': df['DAY'],
    'hour': df['HOUR'],
    'minute': df['MINUTE']
})

# Sort by datetime to ensure proper time series
df = df.sort_values('datetime').reset_index(drop=True)

# Check if the target variables exist
target_vars = ['CDNI', 'DHI', 'DNI', 'CDHI']  # Fixed: DNI instead of CNI
existing_vars = [var for var in target_vars if var in df.columns]
print("Available target variables:", existing_vars)
print("Missing variables:", [var for var in target_vars if var not in df.columns])

# Create the main time series plot
plt.figure(figsize=(16, 8))

# Set style for a professional look
sns.set_style("whitegrid")
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

# Plot each variable
variables_to_plot = existing_vars

# Create the plot
for i, var in enumerate(variables_to_plot):
    plt.plot(df['datetime'], df[var], 
             color=colors[i % len(colors)], 
             linewidth=1.5, 
             label=var,
             alpha=0.8)

plt.title('Time Series Analysis: Solar Irradiance Components Throughout 2020', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Date (2020)', fontsize=12, fontweight='bold')
plt.ylabel('Irradiance (W/m²)', fontsize=12, fontweight='bold')

# Format x-axis similar to the heatmap
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=15))

# Customize the plot
plt.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=0)  # Keep month labels horizontal

# Improve layout
plt.tight_layout()

# Show the plot
plt.show()

# Create individual subplots with proper date formatting
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Individual Solar Irradiance Components - 2020', fontsize=16, fontweight='bold')

# Flatten axes for easier iteration
axes = axes.flatten()

for i, var in enumerate(variables_to_plot):
    if i < 4:  # Only plot if we have up to 4 variables
        axes[i].plot(df['datetime'], df[var], 
                    color=colors[i], 
                    linewidth=1.5, 
                    alpha=0.8)
        axes[i].set_title(f'{var} Over Time', fontweight='bold')
        axes[i].set_xlabel('Date (2020)')
        axes[i].set_ylabel(f'{var} (W/m²)')
        axes[i].grid(True, alpha=0.3)
        
        # Format x-axis for each subplot
        axes[i].xaxis.set_major_locator(mdates.MonthLocator())
        axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        axes[i].tick_params(axis='x', rotation=0)

# Hide any unused subplots
for i in range(len(variables_to_plot), 4):
    axes[i].set_visible(False)

plt.tight_layout()
plt.show()

# Create a heatmap-style visualization similar to your example
# Aggregate data by day of year and hour for heatmap
df['day_of_year'] = df['datetime'].dt.dayofyear
df['hour'] = df['HOUR']

# Create pivot table for heatmap (using DNI as an example)
if 'DNI' in df.columns:
    pivot_data = df.pivot_table(values='DNI', index='hour', columns='day_of_year', aggfunc='mean')
    
    plt.figure(figsize=(20, 8))
    sns.heatmap(pivot_data, cmap='YlOrRd', cbar_kws={'label': 'DNI (W/m²)'})
    plt.title('Heatmap of DNI — Hourly vs. Day of Year 2020', fontsize=16, fontweight='bold')
    plt.xlabel('Days of Year 2020', fontsize=12)
    plt.ylabel('Hour of Day (24-hour format)', fontsize=12)
    
    # Format x-axis to show months
    month_starts = [1, 32, 61, 92, 122, 153, 183, 214, 245, 275, 306, 336]
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    plt.xticks(month_starts, month_labels)
    
    plt.tight_layout()
    plt.show()

# Create summary statistics table
if variables_to_plot:
    print("\nSummary Statistics:")
    summary_stats = df[variables_to_plot].describe()
    print(summary_stats.round(4))
    
    # Calculate correlation matrix
    print("\nCorrelation Matrix:")
    correlation_matrix = df[variables_to_plot].corr()
    print(correlation_matrix.round(4))
    
    # Create a correlation heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, 
                annot=True, 
                cmap='coolwarm', 
                center=0,
                square=True,
                linewidths=0.5)
    plt.title('Correlation Matrix: Solar Irradiance Components', fontweight='bold')
    plt.tight_layout()
    plt.show()

# Print some insights about the data
print(f"\nData spans from {df['datetime'].min()} to {df['datetime'].max()}")
print(f"Total number of observations: {len(df)}")
print(f"Data frequency: Every {df['datetime'].diff().mode()[0]} minutes")