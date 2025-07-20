![UTA-DataScience-Logo](https://github.com/user-attachments/assets/6d626bcc-5430-4356-927b-97764939109d)

# Agricultural-Crops-Image-Classification

* This project aims to classify images of 30 different agricultural crops using deep learning techniques. I built and evaluated both a custom CNN and multiple pre-trained transfer learning models including **MobileNetV2**, **ResNet50V2**, and **EfficientNetB3**.
  
* The goal is to explore model performance across architectures and improve classification accuracy for agricultural applications.

* Kaggle Dataset Link: https://www.kaggle.com/datasets/mdwaquarazam/agricultural-crops-image-classification

## Overview

**Task Definition:** Develop an automated crop identification system capable of classifying 30 different agricultural crop types from images. This computer vision task addresses critical needs in precision agriculture, crop monitoring, and agricultural automation by enabling rapid, accurate identification of crops in various field conditions.

**My Approach:** 

**1. Multi-Model Strategy:**
Implemented and compared 4 different architectures:
   * Custom CNN with ResNet-inspired residual connections
   * MobileNetV2 for efficient mobile deployment
   * ResNet50V2 for robust feature extraction
   * EfficientNetB3 for state-of-the-art performance

**2. Transfer Learning Pipeline:** Leveraged ImageNet pre-trained weights with custom classification heads, frozen backbone training followed by fine-tuning optimization
