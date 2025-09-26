# %%
# Import Libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
import os
import sys
from sklearn.decomposition import PCA

import os, glob
print(os.path.exists(r"C:\Users\nnaji\OneDrive\Documents\GitHub\UGRA_LSTM_Solar-Prediction\File Versions py files\UGRA Onenote"))
print(glob.glob(r"C:\Users\nnaji\OneDrive\Documents\GitHub\UGRA_LSTM_Solar-Prediction\*"))


# %%
#specify the learning rate
learning_rate = 0.0005
#The learning rate has to do with the 
curr_dir = os.getcwd()
print(curr_dir)


#  %%
# avoid changing cwd to a folder without the CSV; pick the CSV explicitly
data_dir = r"C:\Users\nnaji\OneDrive\Documents\GitHub\UGRA_LSTM_Solar-Prediction\File Versions py files"
csv_path = os.path.join(data_dir, "BigData.csv")
if not os.path.isfile(csv_path):
    raise FileNotFoundError(f"BigData.csv not found at {csv_path}")
print("Loading dataset:", csv_path)
df = pd.read_csv(csv_path)


# %%
#Time Series Line Plot
df.plot(subplots=True, figsize=(12, 8), title="Feature Time Series")
plt.tight_layout()
plt.show()

#Correlation Heatmap
import seaborn as sns

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

#Boxplot for Identifying Outliers
for col in df.columns:
    plt.figure(figsize=(6, 3))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()
#At this point, outliers in the data are checked for. The data is completely free of outliers and does not need to be extravagantly normalized. We are
#clear to utilize the MinMaxScalar

# %%
#plotted the feachecking outliers in all of mym coe ?tures to make sure there are not missing data points. 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# %%Load your dataset
df = pd.read_csv(csv_path)  # or your actual filename

# Parameters
features_per_fig = 4
num_features = len(df.columns)
num_figs = (num_features + features_per_fig - 1) // features_per_fig

# Generate a color map
colors = cm.get_cmap('tab20', num_features)  # 'tab20' provides up to 20 distinct colors

# Plot
for i in range(num_figs):
    fig, axs = plt.subplots(features_per_fig, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"Feature Time Series (Part {i+1})", fontsize=16)

    for j in range(features_per_fig):
        idx = i * features_per_fig + j
        if idx >= num_features:
            axs[j].axis('off')
            continue
        col = df.columns[idx]
        axs[j].plot(df[col], label=col, linewidth=1.2, color=colors(idx))
        axs[j].set_ylabel(col)
        axs[j].legend(loc='upper right')
        axs[j].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# %%
# Separate Prediction Variable
solar_Xdata = df.copy()
solar_ydata = solar_Xdata.pop("POWER")


# Reshape Data
solar_Xdata = solar_Xdata.values.reshape(-1,solar_Xdata.shape[1])
solar_ydata = solar_ydata.values.reshape(-1,1)

# %%
# Normalize Data
X_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

solar_XdataN = X_scaler.fit_transform(solar_Xdata)
solar_ydataN = y_scaler.fit_transform(solar_ydata)

# %%
import matplotlib.pyplot as plt

# Convert scaled arrays back to DataFrame for plotting
normalized_df = pd.DataFrame(solar_XdataN, columns=[col for col in df.columns if col != 'POWER'])
normalized_df['POWER'] = solar_ydataN

# Overlay Plot: Irradiance vs. Power
plt.figure(figsize=(12, 5))
for col in ['DNI', 'DHI', 'CDNI', 'CDHI', 'POWER']:
    if col in normalized_df.columns:
        plt.plot(normalized_df[col], label=col, linewidth=1)

plt.title("Normalized Irradiance Features Overlaid with POWER")
plt.xlabel("Time Step")
plt.ylabel("Normalized Value (0–1)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# This plot shows how different sunlight-related features (DNI, DHI, CDNI, CDHI) 
# change over time along with solar power (POWER).
# If the model is learning correctly, solar power should increase when sunlight increases.
# This helps verify that irradiance features are meaningful inputs for prediction.


# Overlay Plot: Time Features vs. Power
plt.figure(figsize=(12, 5))
for col in ['HOUR', 'DAY', 'MONTH', 'POWER']:
    if col in normalized_df.columns:
        plt.plot(normalized_df[col], label=col, linewidth=1)

plt.title("Normalized Time Features Overlaid with POWER")
plt.xlabel("Time Step")
plt.ylabel("Normalized Value (0–1)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# This plot shows how solar power changes throughout the day (HOUR), across days (DAY), and months (MONTH).
# Solar power usually peaks around midday (HOUR), and this pattern should be visible.
# It confirms that time-based features help the model understand daily and seasonal cycles.

# %%
# Create Sequences
X, y = [], []
for i in range(24, len(solar_ydataN)):
    #This is the selected feautres that are going to be used. 
     X.append(solar_XdataN[i-24:i,:])
     y.append(solar_ydataN[i, 0])  # Add the corresponding target
X, y = np.array(X), np.array(y)

#this is a quick check for the sequence lengths
print("X shape:", X.shape)
print("y shape:", y.shape)


# Split data into training and testing sets
train_size = int(0.8 * len(y))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape for LSTM layer
##X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))
##X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))


# %%
# Build LSTM Model
solar_model = Sequential()
solar_model.add(LSTM(units = 50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
solar_model.add(LSTM(units = 50))
solar_model.add(Dense(units=1))

# Create Model
# set Epoch number
setEpoch = 50

# create the optimizer with the desired learning rate
adam_optimizer = Adam(learning_rate=learning_rate)   # use 'learning_rate' not 'lr'
solar_model.compile(optimizer=adam_optimizer, loss='mean_absolute_error')
solar_model.fit(X_train, y_train, epochs=setEpoch, batch_size=32)

# %%
# Run Prediction
y_pred = solar_model.predict(X_test)
y_pred_orig = y_scaler.inverse_transform(y_pred)
y_test_orig = y_scaler.inverse_transform(y_test.reshape(-1,1))

# Evaluate Error (compute MAE before using it in a title)
mae = mean_absolute_error(y_test_orig, y_pred_orig)

# Display / Graph
fig, axs = plt.subplots()
fig.supxlabel("Time [3 days]")
fig.supylabel("DC Power [MW]")

axs.plot(y_test_orig[0:150], label="Actual")
axs.plot(y_pred_orig[0:150], label="Predicted")
axs.legend()

title = f"Train From {X_train.shape[2]} Inputs\nEpochs: {setEpoch}, Avg MAE Error: {mae:.2f}"
fig.suptitle(title)

plt.tight_layout()
plt.show()

# %% Evaluate Error and save a new figure (create fresh figure)
mae = mean_absolute_error(y_test_orig, y_pred_orig)

# Create a new figure for saving/extra display
fig2, ax2 = plt.subplots(figsize=(8,4))
ax2.plot(y_test_orig[0:150], label="Actual")
ax2.plot(y_pred_orig[0:150], label="Predicted")
ax2.legend()
ax2.set_xlabel("Time step")
ax2.set_ylabel("DC Power [MW]")

title = f"Train From {solar_Xdata.shape[1]} Inputs — Epochs: {setEpoch}, Avg MAE: {mae:.2f}"
fig2.suptitle(title)
plt.tight_layout()

# Ensure Images directory exists and save as PNG
out_dir = os.path.join(curr_dir, "Images")
os.makedirs(out_dir, exist_ok=True)
filename = f"Inputs_{solar_Xdata.shape[1]}_Epochs_{setEpoch}.png"
save_path = os.path.join(out_dir, filename)
fig2.savefig(save_path)
print("Saved plot to:", save_path)

plt.show()
plt.close(fig2)

# After normalization
# solar_XdataN = X_scaler.fit_transform(solar_Xdata)
# solar_ydataN = y_scaler.fit_transform(solar_ydata)

# Apply PCA to normalized features (excluding the target)
n_components = 3  # Set to desired number of principal components
pca = PCA(n_components=n_components)
solar_XdataN_pca = pca.fit_transform(solar_XdataN)

print("Explained variance ratio:", pca.explained_variance_ratio_)

# If you want to plot the first two components:
plt.figure(figsize=(8,6))
plt.scatter(solar_XdataN_pca[:, 0], solar_XdataN_pca[:, 1], c=solar_ydataN.flatten(), cmap='viridis', alpha=0.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Projection of Features')
plt.colorbar(label='Normalized POWER')
plt.show()

# Use PCA-transformed features for sequence creation
solar_XdataN_for_seq = solar_XdataN_pca

# ...then continue as before, but use solar_XdataN_for_seq instead of solar_XdataN:
X, y = [], []
for i in range(24, len(solar_ydataN)):
    X.append(solar_XdataN_for_seq[i-24:i, :])
    y.append(solar_ydataN[i, 0])
X, y = np.array(X), np.array(y)

# Split data into training and testing sets
train_size = int(0.8 * len(y))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape for LSTM layer
##X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))
##X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))


# %%
# Build LSTM Model
solar_model = Sequential()
solar_model.add(LSTM(units = 50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
solar_model.add(LSTM(units = 50))
solar_model.add(Dense(units=1))

# Create Model
# set Epoch number
setEpoch = 50

# create the optimizer with the desired learning rate
adam_optimizer = Adam(learning_rate=learning_rate)   # use 'learning_rate' not 'lr'
solar_model.compile(optimizer=adam_optimizer, loss='mean_absolute_error')
solar_model.fit(X_train, y_train, epochs=setEpoch, batch_size=32)

# %%
# Run Prediction
y_pred = solar_model.predict(X_test)
y_pred_orig = y_scaler.inverse_transform(y_pred)
y_test_orig = y_scaler.inverse_transform(y_test.reshape(-1,1))

# Evaluate Error (compute MAE before using it in a title)
mae = mean_absolute_error(y_test_orig, y_pred_orig)

# Display / Graph
fig, axs = plt.subplots()
fig.supxlabel("Time [3 days]")
fig.supylabel("DC Power [MW]")

axs.plot(y_test_orig[0:150], label="Actual")
axs.plot(y_pred_orig[0:150], label="Predicted")
axs.legend()

title = f"Train From {X_train.shape[2]} Inputs\nEpochs: {setEpoch}, Avg MAE Error: {mae:.2f}"
fig.suptitle(title)

plt.tight_layout()
plt.show()

# %% Evaluate Error and save a new figure (create fresh figure)
mae = mean_absolute_error(y_test_orig, y_pred_orig)

# Create a new figure for saving/extra display
fig2, ax2 = plt.subplots(figsize=(8,4))
ax2.plot(y_test_orig[0:150], label="Actual")
ax2.plot(y_pred_orig[0:150], label="Predicted")
ax2.legend()
ax2.set_xlabel("Time step")
ax2.set_ylabel("DC Power [MW]")

title = f"Train From {solar_Xdata.shape[1]} Inputs — Epochs: {setEpoch}, Avg MAE: {mae:.2f}"
fig2.suptitle(title)
plt.tight_layout()

# Ensure Images directory exists and save as PNG
out_dir = os.path.join(curr_dir, "Images")
os.makedirs(out_dir, exist_ok=True)
filename = f"Inputs_{solar_Xdata.shape[1]}_Epochs_{setEpoch}.png"
save_path = os.path.join(out_dir, filename)
fig2.savefig(save_path)
print("Saved plot to:", save_path)

plt.show()
plt.close(fig2)


