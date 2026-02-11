import pandas as pd

# Read the original BigData.csv file
df = pd.read_csv('BigData.csv')

# Remove the day and month columns if they exist
columns_to_remove = ['DAY', 'MONTH']  # Updated to match your CSV column names (uppercase)
existing_columns_to_remove = [col for col in columns_to_remove if col in df.columns]

if existing_columns_to_remove:
    df = df.drop(columns=existing_columns_to_remove)
    print(f"Removed columns: {existing_columns_to_remove}")
else:
    print("Day and month columns not found in the dataset")

# Save to new file called BigData_NoTime.csv
df.to_csv('BigData_NoTime.csv', index=False)

# Display the first few rows to confirm
print("\nFirst 5 rows of the new dataset:")
print(df.head())

# Show column names to verify removal
print(f"\nColumns in the dataset: {list(df.columns)}")
print(f"Total columns: {len(df.columns)}")

print(f"\n✓ New file 'BigData_NoTime.csv' created successfully!")
