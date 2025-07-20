![UTA-DataScience-Logo](https://github.com/user-attachments/assets/6d626bcc-5430-4356-927b-97764939109d)

# Agricultural-Crops-Image-Classification

* This project aims to classify images of 30 different agricultural crops using deep learning techniques. I built and evaluated multiple pre-trained transfer learning models including **MobileNetV2**, **ResNet50V2**, and **EfficientNetB3**.
  
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
    
  * **Evaluation:** Rigorous performance assessment using accuracy and F1-score metrics with detailed per-class analysis

* **Summary of the performance achieved:** **EfficientNetB3** achieved the highest performance with **81% validation accuracy**, outperforming **ResNet50V2 (78%)**, **MobileNetV2 (72%)**, demonstrating the effectiveness of transfer learning for agricultural crop classification.

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

* **Data Visualization**
  
