# Automated Fruit Counting Orchards Using Drones for Yield Estimation

## Overview
This project implements a computer vision-based system to automate fruit counting in orchards using drone-captured imagery, aiding in yield estimation for precision agriculture. The solution leverages OpenCV for image processing, including HSV color segmentation, adaptive thresholding, morphological operations, edge detection, and contour analysis to detect and count fruits (e.g., apples). It also estimates yield in kilograms per hectare and evaluates performance using the F1-score. Designed for educational and practical use, this project supports orchard management by providing actionable data from aerial surveys.

## Features
- Detects and counts fruits in drone images using advanced image processing techniques.
- Estimates yield based on fruit count, tree density, and average fruit weight.
- Includes performance evaluation with F1-score against ground truth data.
- Visualizes detection results with masks and annotated images.
- Scalable for multiple images to cover entire orchards.

## Requirements
- Python 3.7+
- Required Libraries:
  - `opencv-python` (>=4.5.0)
  - `matplotlib` (>=3.5.0)
  - `numpy` (>=1.21.0)
- Hardware: A computer with sufficient memory (recommended 8GB RAM) and, optionally, a drone with a camera (e.g., DJI Mavic).

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/sajee-sajee/Counting-apples-using-image-processing.git
   cd automated-fruit-counting
2. Install dependencies
   ```bash
   pip install -r requirements.txt
3. Input parameters
   ```bash
   Place your drone images in the drone_images folder.
   

  Edit main.py to set ground_truth_count (manual fruit count for F1-score) and adjust parameters like num_trees, trees_per_hectare, and apple_weight_kg in the count_fruits function.
