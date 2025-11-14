import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from datetime import datetime
import pandas as pd

# Save a few prediction samples from the model for quick visual check
# You can look at them in the output folder after training
def save_prediction_sample(model, data_loader, output_dir, num_samples=5):
    os.makedirs(output_dir, exist_ok=True)
    for i, (images, masks) in enumerate(data_loader):
        if i >= num_samples:
            break
        preds = model.predict(images)
        for j in range(len(images)):
            fig, ax = plt.subplots(1, 3, figsize=(12, 4))
            ax[0].imshow(images[j].squeeze(), cmap='gray')
            ax[0].set_title("Image")
            ax[1].imshow(masks[j].squeeze(), cmap='gray')
            ax[1].set_title("Mask")
            ax[2].imshow(preds[j].squeeze(), cmap='hot')
            ax[2].set_title("Prediction")
            for a in ax:
                a.axis('off')
            fig.savefig(os.path.join(output_dir, f"sample_{i}_{j}.png"))
            plt.close()



#Integrate mlflow logging!!!