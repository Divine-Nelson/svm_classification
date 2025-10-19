import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report

# Example: Paste your classification report into a string, or directly compute it from y_test and y_pred
report_dict = {
    '0': {'precision':0.99, 'recall':0.99, 'f1-score':0.99},
    '1': {'precision':0.99, 'recall':0.99, 'f1-score':0.99},
    '2': {'precision':0.98, 'recall':0.98, 'f1-score':0.98},
    '3': {'precision':0.98, 'recall':0.98, 'f1-score':0.98},
    '4': {'precision':0.98, 'recall':0.98, 'f1-score':0.98},
    '5': {'precision':0.98, 'recall':0.98, 'f1-score':0.98},
    '6': {'precision':0.99, 'recall':0.98, 'f1-score':0.98},
    '7': {'precision':0.98, 'recall':0.97, 'f1-score':0.98},
    '8': {'precision':0.97, 'recall':0.98, 'f1-score':0.97},
    '9': {'precision':0.98, 'recall':0.97, 'f1-score':0.97}
}

labels = list(report_dict.keys())
precision = [report_dict[i]['precision'] for i in labels]
recall = [report_dict[i]['recall'] for i in labels]
f1 = [report_dict[i]['f1-score'] for i in labels]

x = np.arange(len(labels))
width = 0.25

plt.figure(figsize=(10,5))
plt.bar(x - width, precision, width, label='Precision', color='#4daf4a')
plt.bar(x, recall, width, label='Recall', color='#377eb8')
plt.bar(x + width, f1, width, label='F1-Score', color='#e41a1c')

plt.xlabel('Digit Class', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Precision, Recall and F1-Score per Class — SVM (Polynomial Kernel)', fontsize=13, fontweight='bold')
plt.xticks(x, labels)
plt.ylim(0.9, 1.0)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
