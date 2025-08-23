import matplotlib.pyplot as plt

# Data from your table
iou = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
precision = [0.26, 0.34, 0.52, 0.71, 0.83, 0.86, 0.97, 1.00, 1.00, 1.00]
recall =    [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.84, 0.43]

# ----- ROC Curve -----
fpr_proxy = [1-p for p in precision]  # using 1-precision as proxy for FPR

plt.figure(figsize=(6,6))
plt.plot(fpr_proxy, recall, marker='o', label="ROC Curve")
plt.plot([0,1],[0,1],'--',color='gray',label="Random Guess")
plt.xlabel("False Positive Rate (1 - Precision)")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve (Approx using Precision)")
plt.legend()
plt.grid(True)
plt.show()

# ----- CMC Curve -----
plt.figure(figsize=(6,6))
plt.plot(iou, recall, marker='o')
plt.xlabel("IoU Threshold")
plt.ylabel("Cumulative Match (Recall)")
plt.title("CMC Curve")
plt.grid(True)
plt.show()


# Plot Precision & Recall vs IoU on the same graph
plt.figure(figsize=(8,6))

plt.plot(iou, precision, marker='o', linestyle='-', color='b', label="Precision")
plt.plot(iou, recall, marker='s', linestyle='--', color='g', label="Recall")

plt.xlabel("IoU Threshold")
plt.ylabel("Score")
plt.title("Precision and Recall vs IoU")
plt.legend()
plt.grid(True)
plt.show()
