import numpy as np
import matplotlib.pyplot as plt

# Load data
arr = np.load(r"C:\Users\manar\Desktop\GP\data\rest_s20260508_001.npy")

print(arr)  # Should be (timesteps, 11)
labels = [
    "thumb", "pointer", "middle", "ring", "pinky",
    "yaw_R", "pitch_R", "roll_R",
    "yaw_L", "pitch_L", "roll_L"
]

# Create one subplot per sensor
fig, axes = plt.subplots(len(labels), 1, figsize=(14, 18), sharex=True)

for i in range(len(labels)):
    axes[i].plot(arr[:, i])

    axes[i].set_ylabel(labels[i])

    # Grid helps debugging noise/spikes
    axes[i].grid(True)

# X-axis label only on bottom plot
axes[-1].set_xlabel("Timestep")

plt.tight_layout()
plt.show()