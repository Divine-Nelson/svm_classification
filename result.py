import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

models = ['SGD', 'Random Forest', 'KNN', 'SVM_LIN', 'SVM_POLY', 'SVM_RBF']
train_times = [3052.37, 7033.10, 7281.68, 14400, 30959, 88623.25]
predict_times = [0.41, 1.44, 16.32, 22.34, 53.73, 95.43]

# Convert to NumPy for grouped plotting
x = np.arange(len(models))  # X-axis positions
width = 0.4  # Bar width

plt.figure(figsize=(10,7))
sns.set_style("whitegrid")

# Plot bars
train_bars = plt.bar(x - width/2, train_times, width, label='Training Time (s)', color='#0077b6')
predict_bars = plt.bar(x + width/2, predict_times, width, label='Prediction Time (s)', color='#00b4d8')

# Labels and title
plt.title("Training and Prediction Time Comparison of Classifiers", fontsize=13, fontweight='bold')
plt.ylabel("Time (seconds)", fontsize=12)
plt.xlabel("Models", fontsize=12)
plt.xticks(x, models, rotation=20)

# Add text labels above each bar
for i, val in enumerate(train_times):
    plt.text(i - width/2, val + max(train_times)*0.02, f"{val:.0f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
for i, val in enumerate(predict_times):
    plt.text(i + width/2, val + max(train_times)*0.02, f"{val:.2f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

# Optional: log scale for better visibility since training >> prediction
plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.show()
