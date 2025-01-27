# Import required libraries
import matplotlib.pyplot as plt
import numpy as np

# Simulate Metrics with Non-Linear Progression
iterations = np.arange(1, 11)  # Simulate 10 iterations

# Non-linear growth for metrics
accuracy = 0.6 + 0.35 / (1 + np.exp(-0.5 * (iterations - 5)))
recall = 0.5 + 0.4 / (1 + np.exp(-0.4 * (iterations - 5)))
f1_score = 0.55 + 0.38 / (1 + np.exp(-0.45 * (iterations - 5)))
precision = 0.7 + 0.25 / (1 + np.exp(-0.6 * (iterations - 5)))

# Plot Metrics
plt.figure(figsize=(12, 8))

plt.plot(iterations, accuracy, label='Accuracy', marker='o', linestyle='-', color='b')
plt.plot(iterations, recall, label='Recall', marker='s', linestyle='-', color='r')
plt.plot(iterations, f1_score, label='F1 Score', marker='^', linestyle='-', color='g')
plt.plot(iterations, precision, label='Precision', marker='d', linestyle='-', color='m')

plt.title('Simulated Non-Linear Metrics Over Iterations', fontsize=16)
plt.xlabel('Iterations', fontsize=14)
plt.ylabel('Metric Values', fontsize=14)
plt.ylim(0.4, 1.0)  # Metrics range from 0.4 to 1.0
plt.xticks(iterations, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()
