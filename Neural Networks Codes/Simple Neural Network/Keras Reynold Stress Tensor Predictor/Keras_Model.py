import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error


# Define file paths
filePaths = ['LM_Channel_0180_prof.csv', 'LM_Channel_0550_prof.csv', 'LM_Channel_1000_prof.csv', 'LM_Channel_2000_prof.csv', 'LM_Channel_5200_prof.csv', 'LM_Couette_R0093_020PI_prof.csv', 'LM_Couette_R0093_100PI_prof.csv', 'LM_Couette_R0220_020PI_prof.csv', 'LM_Couette_R0220_100PI_prof.csv', 'LM_Couette_R0500_020PI_prof.csv', 'LM_Couette_R0500_100PI_prof.csv']

# Corresponding Reynolds numbers
Re = [180, 550, 1000, 2000, 5200, 93, 93, 220, 220, 500, 500]

# Stress tensor components
stressTensorComponents = ['u\'u\'', 'v\'v\'', 'w\'w\'', 'u\'v\'', 'u\'w\'', 'v\'w\'']

# Train a Keras model for a stress tensor component
def trainModel(X_train, y_train):
    model = Sequential()
    model.add(Dense(32, input_dim = 2, activation='relu'))  # Input dimension changed to 2
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adamax')
    model.fit(X_train, y_train, epochs = 50, batch_size = 1, verbose = 1)
    return model

# Initialize lists for training data
X_trainAll = []
y_trainAll = []

# Initialize lists for storing average MSE for each stress tensor component
avg_mse_components = [[] for _ in range(len(stressTensorComponents))]

for i, filePath in enumerate(filePaths):
    # Read CSV file
    data = pd.read_csv(filePath)
    # Extract wall distance
    wallDistance = data['y/delta'].values
    # Extract Reynolds number
    Reynolds = np.full_like(wallDistance, Re[i])
    # Create a figure for Reynolds number
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Reynolds Number: {Re[i]}")
    axs = axs.ravel()
    # Combine mean velocity and Reynolds number as features
    X_data = np.column_stack((wallDistance, Reynolds))
    for j, component in enumerate(stressTensorComponents):
        # Extract stress tensor component values
        stressComponent = data[component].values
        # Split data for training and testing
        X_train, X_test, y_train, y_test = train_test_split(X_data, stressComponent.reshape(-1, 1), test_size=0.2, random_state=42)
        # Scale features
        scaler = StandardScaler()
        X_trainScaled = scaler.fit_transform(X_train)
        X_testScaled = scaler.transform(X_test)
        # Train the model
        model = trainModel(X_trainScaled, y_train)
        # Predict stress tensor component values
        y_pred = model.predict(X_testScaled)
        # Evaluate the model
        # Calculate MSE and store it
        mse = mean_squared_error(y_test, y_pred)
        avg_mse_components[j].append(mse)
        # Plot stress tensor component against mean velocity
        axs[j].scatter(wallDistance, stressComponent, label='Ground Truth')
        axs[j].scatter(X_test[:, 0], y_pred, color='red', label='Predicted')
        axs[j].set_xlabel('Wall Distance (Y)')
        axs[j].set_ylabel(f"{component}")
        axs[j].set_title(f"{component} vs Y")
        axs[j].legend()

    plt.tight_layout()
    plt.show()

# Calculate average MSE for each stress tensor component
avg_mse_per_component = [np.mean(avg_mse) for avg_mse in avg_mse_components]

# Plot average MSE for each stress tensor component
plt.figure(figsize=(10, 6))
bars = plt.bar(np.arange(len(stressTensorComponents)), avg_mse_per_component, color='skyblue')
plt.xlabel('Stress Tensor Component Index')
plt.ylabel('Average Mean Squared Error (MSE)')
plt.title('Average MSE for Each Stress Tensor Component')
plt.xticks(np.arange(len(stressTensorComponents)), stressTensorComponents, rotation=45)
plt.tight_layout()

# Add annotations
for bar, mse in zip(bars, avg_mse_per_component):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{mse:.2f}', ha='center', va='bottom')

plt.show()




