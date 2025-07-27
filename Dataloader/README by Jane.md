# Requirements
- Required to perform right ventricular (RV) segmentation in cardiac MRI
- Training set contained 200 annotated images.  
- Select at least 2 pathologies (>= 2) to train your deep learning models. 
- Select the corresponding pathologies from the testing set to test your models.

# Preprocessing + EDA Findings
- RV is label 3.
- Label mapping (Background = 0, LV = 1, Myo = 2, RV = 3).
- A standard loss function (like CrossEntropy or BCE) will be heavily biased toward predicting label 0 everywhere.
- Do not use CrossEntropyLoss, BCEWithLogitsLoss, 4-class segmentation, full label masks (0-3), because those are only needed for multi-class segmentation, and our assignment doesn’t ask for that.
- Recommended Loss Functions (for binary segmentation):  Dice + BCE
- Use only preprocessed 2D slices from MnM2_preprocessed_2Dslices/{train,val,test}/{images,masks}/ for training, validation, and testing.

# Installation
- Install scikit-image via pip or conda
<pre><code>python -m pip install -U scikit-image</code></pre>
<pre><code>conda install scikit-image</code></pre>
- Install nibabel
<pre><code>pip install nibabel</code></pre>
- Install albumentations via pip or conda
<pre><code>pip install albumentations</code></pre>
<pre><code>conda install -c conda-forge albumentations</code></pre>

# Some Suggestion for loss function
## 1) DiceLoss ✅
- Best choice for imbalanced segmentation tasks like this.
<pre><code>import segmentation_models_pytorch as smp
loss_fn = smp.losses.DiceLoss(mode='binary')</code></pre>
- Why: Focuses on region overlap rather than pixel-wise matching
- Effectiveness: Works especially well when the RV is small compared to the background

## 2) Dice + BCE (Combined Loss) ✅✅ (BEST)
- More robust, combines Dice with Binary Cross Entropy.
<pre><code>import segmentation_models_pytorch as smp
loss_fn = smp.losses.DiceBCELoss(mode='binary')</code></pre>
- Why: Dice captures overlap, BCE stabilizes learning
- Recommended: If you notice instability or poor early training performance

# Some suggestion for model architectures
## 1) Baseline: U-Net (ResNet34)
- No attention
- Widely used

## 2) Feature Pyramid Network (FPN)
- Combines features from multiple scales in a top-down fashion.
- Efficient in both computation and memory.
- No explicit attention mechanisms, but still captures multi-scale context
- Great for anatomical variability in cardiac MRIs.
- Use encoder like resnet18 or mobilenet_v2
<pre><code>import segmentation_models_pytorch as smp
model = smp.FPN(
    encoder_name="resnet18",
    encoder_weights="imagenet",  
    in_channels=1,              
    classes=4
)</code></pre>

## 3) Comparison of Attention-Based Segmentation Models
| Model                        | Type of Attention       | Benefit                                 |
|-----------------------------|-------------------------|------------------------------------------|
| **U-Net + CBAM**             | Channel + Spatial       | Lightweight, boosts relevant features    |
| **Attention U-Net**          | Gated Spatial Attention | Learns to focus on RV-specific features |
| **SE-U-Net**                 | Channel-wise            | Reweights important features             |
| **U-Net + Transformer**      | Global Spatial          | Captures long-range dependencies         |
| **Residual Attention U-Net** | Combined + Residual     | Deep learning with refined attention     |
