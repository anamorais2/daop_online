import matplotlib.pyplot as plt
import torchvision
import numpy as np

def debug_visual_batch(loader, save_path="debug_multimodal_batch.png"):
    print("\n🔍 Generating debug image...")
    # Get the first batch
    batch = next(iter(loader))
    
    # images has shape [Batch, 3, 3, 224, 224] 
    # (Batch, 3 views, RGB, H, W)
    images = batch['images'] 
    
    # Get the first patient from the batch (index 0)
    # sample will be [3, 3, 224, 224] -> 3 RGB images
    sample = images[0] 
    
    # Create a grid with the 3 images side by side
    grid = torchvision.utils.make_grid(sample, nrow=3, normalize=True)
    
    plt.figure(figsize=(12, 4))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.title("Debug: Abdomen | Femur | Head")
    plt.savefig(save_path)
    plt.close()
    print(f"Success! Check the file: {save_path}\n")