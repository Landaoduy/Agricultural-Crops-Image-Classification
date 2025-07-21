
![UTA-DataScience-Logo](https://github.com/user-attachments/assets/6d626bcc-5430-4356-927b-97764939109d)

# Agricultural-Crops-Image-Classification

* This project aims to classify images of 30 different agricultural crops using deep learning techniques. I built and evaluated a customed CNN Model, multiple pre-trained transfer learning models including **MobileNetV2**, **ResNet50V2**, and **EfficientNetB3**.
  
* The goal is to explore model performance across architectures and improve classification accuracy for agricultural applications.

* Kaggle Dataset Link: https://www.kaggle.com/datasets/mdwaquarazam/agricultural-crops-image-classification

## Overview

* **Task Definition:** Develop an automated crop identification system capable of classifying 30 different agricultural crop types from images. This computer vision task addresses critical needs in precision agriculture, crop monitoring, and agricultural automation by enabling rapid, accurate identification of crops in various field conditions.

* **My Approach:**
  * **Multi-Model Strategy:**
    Implemented and compared 3 different architectures:
    * MobileNetV2 for efficient mobile deployment
    * ResNet50V2 for robust feature extraction
    * EfficientNetB3 for state-of-the-art performance
      
  * **Transfer Learning Pipeline:** Leveraged ImageNet pre-trained weights with custom classification heads, frozen backbone training followed by fine-tuning optimization
    
  * **Data Enhancement:** Comprehensive augmentation strategy including geometric transformations, color adjustments, and noise injection to improve model robustness and generalization
    
  * **Evaluation:** Rigorous performance assessment using accuracy, precision, recall and F1-score metrics with detailed per-class analysis

* **Summary of The Performance Achieved:** **EfficientNetB3** achieved the highest performance with **81% validation accuracy**, outperforming **ResNet50V2 (78%)**, **MobileNetV2 (72%)**, demonstrating the effectiveness of transfer learning for agricultural crop classification.

## Summary of Workdone:

### Data:

* Type: Image Dataset
  * Input: RGB crop images in JPG/JPEG format with diverse agricultural scenes
  * Target: Multi-class labels for 30 agricultural crop types

* Size:
  * Total images: 759 images
  * Training set: 608 images (80%)
  * Validation set: 151 images (20%)
  * Image resolution: 224×224 pixels (300×300 for EfficientNet)
 
### Preprocessing:

* **Image Cleaning**
  * Removed corrupted or unreadable files using a lightweight JFIF header check:
    
    * Opened each image in binary mode and verified that "JFIF" was present in the header
      
    * Invalid or unreadable images were skipped and deleted to prevent model training issues
      
      ```sh
      with open(fpath, "rb") as fobj:
        is_jfif = b"JFIF" in fobj.peek(10)
      ```
    * This resulted in a clean dataset of 759 usable images

* **Image Loading and Resizing**
  * Used image_dataset_from_directory to:
    
    * Automatically infer labels from subfolder names
      
    * Shuffle and split into 80% training and 20% validation using validation_split
      
    * Resize images to:
      * 224×224 for MobileNetV2 and ResNet50V2
        
      * 300×300 for EfficientNetB3
        
    * Batch size set to:
      * 16 for MobileNetV2 and ResNet50V2
      * 10 for EfficientNetB3

### Data Visualization

* **Image and Labels of crops**
<img width="793" height="812" alt="image" src="https://github.com/user-attachments/assets/19955d4a-0f7d-40ab-b24f-82417fecb28a" />

* **Bar plot of Class Distribution**
<img width="1189" height="590" alt="image" src="https://github.com/user-attachments/assets/e7f3264c-cb6d-4345-8b6e-e5ce1bdde93e" />

### Problem Formulation

#### Data Augmentation
  * Applied on-the-fly augmentation using a Keras Sequential pipeline to improve generalization:
    
    ```sh
    data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.1),
    layers.GaussianNoise(0.02),
    ])
    ```
  * Applied dynamically to each training batch using the .map() method
    
  * Ensured diverse representations of crops under different lighting, positioning, and noise conditions

#### Prefetching
  * Used tf.data.AUTOTUNE to prefetch batches and optimize GPU utilization
    
  * This minimizes input pipeline bottlenecks during training:
    ```sh
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    validation_ds = validation_ds.prefetch(tf.data.AUTOTUNE)
    ```
#### Model Architecture
* **Custom CNN Model (baseline)**
  * Inspired by Xception-style separable convolutions with residual connections
    
  * Includes:
    * Rescaling layer
    * SeparableConv2D → BatchNorm → ReLU blocks
    * MaxPooling + residual path via Conv2D
    * Global Average Pooling + Dropout
    * Final classifier: Dense → Dropout → Softmax (30 classes)
      
  * Accuracy: ~30%

* **MobileNetV2**
  * Lightweight architecture ideal for mobile/edge deployment
    
  * ImageNet pre-trained, frozen base with custom classifier head
    
  * Classifier includes:
    * Global Average Pooling
    * Dense (512 to 128)
    * BatchNorm, Dropout
    * Softmax for 30 classes
      
  * Validation Accuracy: 72%
<img width="576" height="455" alt="image" src="https://github.com/user-attachments/assets/757d9165-288d-4792-ae74-35d82969e8f1" />

* **ResNet50V2**
  * Deep residual network with skip connections
    
  * Frozen base backbone; only custom classification layers trained
    
  * Strong spatial feature learning via deeper layers
<img width="574" height="455" alt="image" src="https://github.com/user-attachments/assets/1072cf36-6dc7-47db-9b40-dee22f7a4f29" />

* **EfficientNetB3**
  *State-of-the-art architecture with compound scaling (depth, width, resolution)

  * Input size increased to 300×300 to leverage resolution scaling
    
  * Custom classification head with:
    * Dense (512 → 256), BatchNorm, Dropout
      
  * **Best performing model**: 81% accuracy, robust across all classes
<img width="585" height="455" alt="image" src="https://github.com/user-attachments/assets/259bbd05-2348-4ea2-8acc-01248c5dc3de" />

### Training

#### **Approach**
* Pretrained on ImageNet

* Initially froze backbone to train only classifier head

#### **Hyperparameters**
| Parameter             | Value                         |
| --------------------- | ----------------------------- |
| Optimizer             | Adam                          |
| Initial Learning Rate | 3e-4                          |
| Loss Function         | SparseCategoricalCrossentropy |
| Epochs                | 50–60                         |
| Batch Size            | 10 (Mobile, ResNet), 16 (EfficientNet)   |

#### **Learning Scheduling and Regularization**

* **ReduceLROnPlateau:** Halve LR when validation loss plateaus

* **EarlyStopping:** Stop training when no improvement after 7–8 epochs

* **ModelCheckpoint:** Save the best model (lowest val_loss)

### Performance Comparison

| Model              | Validation Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) |
| ------------------ | ------------------- | -------------------- | ----------------- | ------------------- |
| **EfficientNetB3** | **81%**             | **82%**              | **81%**           | **80%**             |
| ResNet50V2         | 78%                 | 78%                  | 78%               | 77%                 |
| MobileNetV2        | 72%                 | 76%                  | 72%               | 72%                 |

Top Performing Classes:

Perfect Classification: Coconut, Mustard-oil, Sunflower (100% accuracy)
Strong Performance: Lemon, Clove, Tea, Pineapple (>85% accuracy)
Challenging Classes: Sugarcane, Vigna-radiati, Tomato

<img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/39864308-551b-4bf7-9462-0b28b2ee4f05" />

### Conclusion

* MobileNetV2 offered solid performance, ideal for mobile or embedded systems

* ResNet50V2 showed strong feature extraction capabilities and improved class recall

* EfficientNetB3 outperformed all other models, achieving 81% validation accuracy, with high F1-scores and balanced performance across classes
