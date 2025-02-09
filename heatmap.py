import matplotlib.pyplot as plt
import numpy as np

def generate_heatmap(detections):
    if not detections:
        print("No detections to generate heatmap.")
        return

    width, height = 640, 480  # Example dimensions
    heatmap = np.zeros((height, width))

    for _, x1, y1, x2, y2 in detections:
        heatmap[y1:y2, x1:x2] += 1

    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title("Heatmap of Plastic Waste Detections")
    plt.show()
