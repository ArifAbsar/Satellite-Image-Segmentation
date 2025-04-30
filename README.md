# Aerial Image Segmentation

This repository offers a focused demonstration of binary road segmentation from aerial (satellite) imagery, implemented entirely in a Jupyter Notebook using PyTorch and Albumentations. By following this example, users can understand how to prepare data, apply advanced augmentations, perform inference with a trained segmentation model, and visualize results—all within a self-contained environment.

## Repository Contents

- **Aerial_image_segmentation.ipynb**  
  A comprehensive notebook that guides you through:
  1. **Data Loading**: Utilizing a pandas DataFrame to map and load paired satellite images and their binary road masks.  
  2. **Data Augmentation**: Employing Albumentations to apply spatial and color transforms (e.g., flips, random crops, normalization) consistently to both images and masks.  
  3. **Model Inference**: Initializing a custom PyTorch segmentation network, loading pre-trained weights, executing a forward pass, applying sigmoid activation to raw logits, and thresholding to produce binary predictions.  
  4. **Visualization**: Leveraging a helper function (`show_image`) to render the input image, ground-truth mask, and model output side by side, facilitating qualitative assessment of segmentation performance.

- **best_model.pt**  
  A serialized PyTorch state dictionary containing the learned weights of the segmentation network. This file enables immediate inference without re-training.

## Key Features

1. **End-to-End Demonstration**  
   The notebook encapsulates the entire inference pipeline—from reading file paths to displaying final segmentation—allowing users to run the example out of the box.

2. **Advanced Augmentation with Albumentations**  
   Albumentations transforms enhance model robustness by simulating real-world variations in lighting, orientation, and scale. These augmentations are applied on-the-fly during data loading.

3. **PyTorch-Based Segmentation Model**  
   A flexible architecture defined in the notebook supports binary segmentation. Users can adapt this structure to other encoder-decoder backbones or multi-class tasks.

4. **Interactive Visualization**  
   The `show_image` utility automatically handles both PyTorch tensors and NumPy arrays, permuting or transposing channels as needed for Matplotlib display.

## Dependencies

Install the required Python libraries using pip:

```bash
pip install torch torchvision albumentations numpy pandas matplotlib
```
- **torch & torchvision**: Core framework for defining and running the segmentation model.  
- **albumentations**: High-performance image augmentation library.  
- **numpy & pandas**: Efficient array operations and DataFrame-based file mapping.  
- **matplotlib**: Plotting library for result visualization.  


## Data Organization

Arrange your dataset directory (`data_dir`) as follows:

```text
<data_dir>/images/    # Contains raw satellite images (PNG, JPG).
<data_dir>/masks/     # Contains binary masks where road pixels are marked (same size and naming convention).
mapping.csv           # CSV file with two columns: 'images' and 'masks', listing filenames.
```

```python
row = df.iloc[idx]
image_path = os.path.join(data_dir, 'images', row.images)
mask_path  = os.path.join(data_dir, 'masks',  row.masks)

```

## Inference Workflow

Follow these steps in the notebook to perform segmentation inference:

### 1. Load the Pretrained Model

```python
model.load_state_dict(torch.load('best_model.pt'))
model.to(device)
model.eval()
```
Transfers the model weights to the chosen device (CPU/GPU) and sets evaluation mode.


### 2. Prepare a Sample

```python
image, mask = valset[idx]                   # Retrieve a single sample (tensor or array)
image_batch = image.unsqueeze(0).to(device)  # Add batch dimension: (1, C, H, W)
```
### 3. Run Model Inference
```python
logits        = model(image_batch)               # Raw output tensor (logits)
probabilities = torch.sigmoid(logits)            # Convert logits to [0,1] probability map
pred_mask     = (probabilities > 0.5).float()    # Threshold probabilities at 0.5

```
The sigmoid activation maps raw values into a probability-like range, enabling reliable binary thresholding.

### 4. Visualize Predictions
```python
show_image(image, mask, pred_mask.squeeze(0))  # Display original, ground truth, and prediction
```
This utility promotes consistent visualization without manual axis adjustments.

## Getting Started

1. **Clone or Download** this repository to your local machine.  
2. **Install Dependencies** via:
   ```bash
       pip install torch torchvision albumentations numpy pandas matplotlib
   ```
3. **Arrange Data** under ``data_dir`` as described above and update paths in the notebook.
4. **Launch Jupyter Notebook** and open ``Aerial_image_segmentation.ipynb``.
5. **Run All Cells** Sequentially to load data, perform inference with ``best_model.pt``, and visualize segmentation outputs.
