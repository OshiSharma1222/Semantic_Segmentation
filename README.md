# Offroad Autonomy Segmentation Challenge - Complete Project Structure

**Duality AI Challenge** - Train a robust semantic segmentation model using synthetic desert environment data from Falcon platform.

## 📋 Project Overview

This project provides a complete workflow for semantic segmentation of desert environments using U-Net deep learning architecture. The dataset contains 10 semantic classes representing different terrain features critical for off-road autonomous navigation.

### Classes
| ID    | Class Name        | ID    | Class Name         |
|-------|-------------------|-------|-------------------|
| 100   | Trees             | 600   | Flowers            |
| 200   | Lush Bushes       | 700   | Logs               |
| 300   | Dry Grass         | 800   | Rocks              |
| 500   | Dry Bushes        | 7100  | Landscape (Ground) |
| 550   | Ground Clutter    | 10000 | Sky                |

## 📂 Project Structure

```
Semantic_Segmentation/
├── 1_EDA_Analysis.ipynb                    # Exploratory Data Analysis
├── 2_Feature_Engineering.ipynb             # Data Preprocessing & Augmentation
├── 3_Model_Training.ipynb                  # U-Net Model Training
├── 4_Model_Evaluation_Testing.ipynb        # Evaluation & Performance Analysis
├── offroad-env/                            # Python virtual environment
├── checkpoints/                            # Model weights
│   ├── best_model.pth                      # Best checkpoint during training
│   └── final_model.pth                     # Final trained model
├── logs/                                   # Training logs
├── Offroad_Segmentation_Training_Dataset/  # Training data
│   ├── train/
│   │   ├── Color_Images/
│   │   └── Segmentation/
│   └── val/
│       ├── Color_Images/
│       └── Segmentation/
├── Offroad_Segmentation_testImages/        # Test data
│   ├── Color_Images/
│   └── Segmentation/
└── README.md                               # This file
```

## 🚀 Quick Start Guide

### Step 1: Activate Environment
```cmd
offroad-env\Scripts\activate.bat
```

### Step 2: Run Notebooks in Order

#### Notebook 1: Exploratory Data Analysis (1_EDA_Analysis.ipynb)
- **Duration**: 15-30 minutes
- **Outputs**: 
  - Dataset statistics and class distribution
  - Sample visualizations with color-coded masks
  - Image dimension analysis
  - Data quality assessment
- **Files Generated**:
  - `class_distribution.png`
  - `sample_visualizations.png`
  - `image_dimensions.png`

**Key Insights**:
- Analyze dataset composition and class balance
- Identify challenging classes with low representation
- Understand image properties and quality

---

#### Notebook 2: Feature Engineering & Preprocessing (2_Feature_Engineering.ipynb)
- **Duration**: 10-20 minutes
- **Outputs**:
  - Normalized dataset with augmentation pipeline
  - Data loaders for training
  - Class weights for imbalanced dataset
- **Files Generated**:
  - `preprocessing_config.json` - Configuration for training
  - `class_weights.npy` - Class weights for loss function
  - `augmented_samples.png` - Augmentation examples

**Key Steps**:
- Calculate normalization parameters (mean, std)
- Implement data augmentation (rotation, flip, brightness, noise)
- Create PyTorch DataLoaders
- Calculate class weights for weighted loss

---

#### Notebook 3: Model Training (3_Model_Training.ipynb)
- **Duration**: 2-6 hours (depends on GPU and data)
- **Outputs**:
  - Trained U-Net model
  - Training history and metrics
- **Files Generated**:
  - `checkpoints/best_model.pth` - Best model checkpoint
  - `checkpoints/final_model.pth` - Final model weights
  - `logs/training_history.json` - Training metrics
  - `training_history.png` - Loss and IoU curves

**Model Architecture**:
- **U-Net** with encoder-decoder structure
- 5 downsampling blocks, 5 upsampling blocks
- Batch normalization and skip connections
- Total parameters: ~31M

**Training Configuration**:
- **Epochs**: 100 (with early stopping)
- **Learning Rate**: 0.001 (adaptive with scheduler)
- **Batch Size**: 16
- **Loss Function**: Combined CrossEntropy + Dice
- **Optimizer**: Adam with weight decay
- **Early Stopping**: 15 epochs patience

---

#### Notebook 4: Model Evaluation & Testing (4_Model_Evaluation_Testing.ipynb)
- **Duration**: 10-20 minutes
- **Outputs**:
  - Performance metrics (IoU, Dice)
  - Per-class analysis
  - Prediction visualizations
  - Failure case analysis
- **Files Generated**:
  - `test_metrics.csv` - Per-class metrics
  - `per_class_metrics.png` - IoU and Dice charts
  - `predictions_visualization.png` - Sample predictions
  - `final_summary_statistics.png` - Comprehensive summary
  - `final_evaluation_report.txt` - Complete report

**Metrics Calculated**:
- **IoU (Intersection over Union)**: Primary metric for segmentation
- **Dice Score**: F1-like metric for binary/multi-class segmentation
- **Per-class Analysis**: Performance breakdown by semantic class
- **Error Analysis**: Identification of challenging cases

## 📊 Expected Performance

### Typical Metrics
- **Mean IoU**: 0.60-0.80 (depends on class complexity)
- **Mean Dice**: 0.70-0.85
- **Best Classes**: Sky, Landscape, Rocks (high IoU)
- **Challenging Classes**: Flowers, Logs, Ground Clutter (variable representation)

### Performance by Dataset Type
- **Landscape Class**: High accuracy (dominant in most images)
- **Sky Class**: High accuracy (distinctive blue color)
- **Small Objects**: Lower accuracy (flowers, logs)
- **Similar Classes**: Moderate accuracy (dry grass vs. landscape)

## 🔧 Configuration & Customization

### Modify Training Parameters
Edit in `3_Model_Training.ipynb`:
```python
TRAIN_CONFIG = {
    'epochs': 100,              # Increase for better convergence
    'learning_rate': 0.001,     # Lower = slower, more stable
    'weight_decay': 1e-5,       # L2 regularization strength
    'patience': 15,             # Early stopping patience
}
```

### Adjust Data Augmentation
Edit in `2_Feature_Engineering.ipynb`:
```python
train_transform = A.Compose([
    A.Resize(512, 512),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.Rotate(limit=45, p=0.5),
    # Add more augmentations as needed
])
```

### Change Model Architecture
Modify U-Net in `3_Model_Training.ipynb`:
```python
class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        # Adjust channel numbers and depth
```

## 📈 Workflow Visualization

```
1. EDA_Analysis.ipynb
   ↓ (Understand data)
   ├─ class_distribution.png
   ├─ sample_visualizations.png
   └─ image_dimensions.png
   
2. Feature_Engineering.ipynb
   ↓ (Prepare data)
   ├─ preprocessing_config.json
   ├─ class_weights.npy
   └─ augmented_samples.png
   
3. Model_Training.ipynb
   ↓ (Train model)
   ├─ checkpoints/best_model.pth
   ├─ logs/training_history.json
   └─ training_history.png
   
4. Model_Evaluation_Testing.ipynb
   ↓ (Evaluate results)
   ├─ test_metrics.csv
   ├─ per_class_metrics.png
   ├─ predictions_visualization.png
   ├─ final_summary_statistics.png
   └─ final_evaluation_report.txt
```

## 💡 Tips for Better Results

### 1. Data Augmentation
- Increase augmentation probability for overfitting
- Add specialized augmentations (weather effects, brightness)
- Use albumentations for GPU-accelerated augmentation

### 2. Class Imbalance
- Use weighted loss functions (already implemented)
- Oversample minority classes
- Use focal loss for difficult samples
- Apply class-specific thresholds

### 3. Model Improvements
- Increase model depth/width for better capacity
- Use pre-trained encoders (ResNet, EfficientNet)
- Implement attention mechanisms (CBAM, Squeeze-Excitation)
- Try different architectures (DeepLab, FCN, PSPNet)

### 4. Training Strategy
- Use progressive resizing (start small, increase size)
- Implement learning rate scheduling (cosine, warmup)
- Use mixed precision training for faster convergence
- Accumulate gradients for larger effective batch size

### 5. Post-processing
- Apply CRF (Conditional Random Field) for spatial smoothness
- Morphological operations (opening, closing)
- Ensemble predictions from multiple models
- Test-time augmentation (TTA)

## 🎯 Key Deliverables

### 1. Trained Model ✓
- Location: `checkpoints/final_model.pth`
- Architecture: U-Net with 10 semantic classes
- Input: 512x512 RGB images
- Output: 512x512 segmentation masks

### 2. Performance Report ✓
- Location: `final_evaluation_report.txt`
- Includes: IoU, Dice, per-class analysis
- Insights: Best/worst performing classes
- Recommendations: Improvement strategies

### 3. Documentation & Visualizations ✓
- Loss curves and training history
- Per-class performance charts
- Sample predictions with overlays
- Error maps and failure analysis

## 📝 Troubleshooting

### Common Issues

**Issue**: Out of memory during training
- **Solution**: Reduce batch size in TRAIN_CONFIG
```python
BATCH_SIZE = 8  # Instead of 16
```

**Issue**: Model not converging
- **Solution**: Adjust learning rate
```python
'learning_rate': 0.0005,  # Try lower
```

**Issue**: Poor performance on specific class
- **Solution**: Adjust class weights
```python
class_weights[class_idx] *= 2.0  # Increase weight
```

**Issue**: Slow data loading
- **Solution**: Increase num_workers (if available)
```python
num_workers=4  # Parallel data loading
```

## 🔍 Monitoring Training

Check metrics during training:
1. **Loss should decrease** over time
2. **IoU should increase** over time
3. **No significant gap** between train and val loss (indicates good generalization)
4. **Early stopping** triggered when validation IoU plateaus

## 🎓 Learning Resources

- **U-Net Paper**: [Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)
- **Semantic Segmentation**: [Review paper](https://arxiv.org/abs/2012.14101)
- **PyTorch Documentation**: https://pytorch.org/docs/
- **Albumentations**: https://albumentations.ai/

## 📬 Support & Questions

For challenges or improvements:
1. Review the commented code in notebooks
2. Check loss curves and metrics visualization
3. Analyze failure cases in evaluation notebook
4. Experiment with hyperparameters systematically

## ✅ Checklist for Hackathon Submission

- [x] Complete EDA with visualizations
- [x] Implement feature engineering with augmentation
- [x] Train U-Net model with proper monitoring
- [x] Evaluate on test set with comprehensive metrics
- [x] Generate performance report
- [x] Save trained model weights
- [x] Document findings and insights
- [x] Create presentation-ready visualizations

## 🏆 Final Notes

This project demonstrates professional ML development practices:
- Modular, well-documented code
- Comprehensive EDA before modeling
- Proper data preprocessing and augmentation
- Rigorous evaluation and metrics
- Clear documentation and reproducibility

Good luck with the Offroad Autonomy Segmentation Challenge! 🚀

---

**Last Updated**: December 2024
**Framework**: PyTorch
**Python Version**: 3.10+
**Dependencies**: See requirements in notebooks
