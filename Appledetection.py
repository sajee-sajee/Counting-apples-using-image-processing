import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def process_image(image_path, ground_truth_count=None):
    ###### Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return 0, None, None
    
    ####Resize image for consistency (adjust based on drone altitude)
    image = cv2.resize(image, (640, 480))
    
    ##### Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # ###Define color range for trees (green) - Adjusted for broader range
    lower_green = np.array([25, 20, 20])  
    upper_green = np.array([90, 255, 255])
    tree_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    ##### Define color range for fruits (red apples) - Fine-tuned for your image
    lower_red1 = np.array([0, 20, 60])    #####Broader range for red/yellow hues
    upper_red1 = np.array([20, 255, 255])
    lower_red2 = np.array([160, 20, 60])  #### Adjusted upper red
    upper_red2 = np.array([180, 255, 255])
    fruit_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    fruit_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    fruit_mask = fruit_mask1 + fruit_mask2
    
    # ####Apply morphological operations to clean up the fruit mask
    kernel = np.ones((7, 7), np.uint8)  # Larger kernel for drone images
    fruit_mask = cv2.morphologyEx(fruit_mask, cv2.MORPH_OPEN, kernel)
    fruit_mask = cv2.dilate(fruit_mask, kernel, iterations=2)
    
    ####### Apply fruit detection within tree regions
    fruit_mask = cv2.bitwise_and(fruit_mask, tree_mask)
    
    # #####Find contours
    contours, _ = cv2.findContours(fruit_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"{image_path} - Number of contours detected: {len(contours)}")
    
    ##### Filter contours by area and circularity
    min_area = 15          
    circularity_threshold = 0.3  
    fruits = []
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            print(f"Contour {i}: Area={area:.1f}, Perimeter={perimeter:.1f}, Circularity={circularity:.3f}")
            if area > min_area and circularity > circularity_threshold:
                fruits.append(contour)
    
    fruit_count = len(fruits)
    print(f"{image_path} - Number of apples after filtering: {fruit_count}")
    
   
    metrics = None
    if ground_truth_count is not None:
       
        tp = min(fruit_count, ground_truth_count)  
        fp = max(fruit_count - ground_truth_count, 0) 
        fn = max(ground_truth_count - fruit_count, 0)  
        
        # Compute precision, recall, and F-score
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": precision,
            "recall": recall,
            "f_score": f_score
        }
        print(f"{image_path} - Metrics: Precision={precision:.3f}, Recall={recall:.3f}, F1-Score={f_score:.3f}")
    
    ##### Visualize detections
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for cnt in fruits:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(image_rgb, center, radius, (0, 255, 0), 2)
    
    # ###Add count and metrics text
    cv2.putText(image_rgb, f"Fruit Count: {fruit_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    if metrics:
        cv2.putText(image_rgb, f"F1-Score: {metrics['f_score']:.3f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    return fruit_count, image_rgb, metrics

def estimate_yield(image_paths, ground_truth=None, avg_fruit_weight=0.15):
    
    total_fruit_count = 0
    total_yield_kg = 0
    all_metrics = []
    
    for img_path in image_paths:
        gt_count = ground_truth.get(img_path) if ground_truth else None
        count, image_rgb, metrics = process_image(img_path, gt_count)
        total_fruit_count += count
        
        if metrics:
            all_metrics.append(metrics)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(image_rgb)
        title = f"Detected Fruits (Count: {count})"
        if metrics:
            title += f", F1-Score: {metrics['f_score']:.3f}"
        plt.title(title)
        plt.axis('off')
        plt.show()
    
    total_yield_kg = total_fruit_count * avg_fruit_weight
    
    
    avg_metrics = None
    if all_metrics:
        avg_precision = np.mean([m["precision"] for m in all_metrics])
        avg_recall = np.mean([m["recall"] for m in all_metrics])
        avg_f_score = np.mean([m["f_score"] for m in all_metrics])
        avg_metrics = {
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "avg_f_score": avg_f_score
        }
        print(f"Average Metrics - Precision: {avg_precision:.3f}, Recall: {avg_recall:.3f}, F1-Score: {avg_f_score:.3f}")
    
    return {
        "total_fruit_count": total_fruit_count,
        "total_yield_kg": total_yield_kg,
        "metrics": avg_metrics
    }

def main():
    #### Example image paths and ground truth
    image_folder = "images"
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.jpg')]
    
    #### Example ground-truth data 
    ground_truth = {
        os.path.join(image_folder, "images/aplessss.jpg"):7,
        os.path.join(image_folder, "images/app.jpg"): 6,
        os.path.join(image_folder, "images/appl.jpeg"): 5,
        os.path.join(image_folder, "images/apples.jpeg"): 4,
        os.path.join(image_folder, "images/apples.jpg"): 3,
        os.path.join(image_folder, "images/applesss.jpg"): 2,
        os.path.join(image_folder, "images/appppp.jpg"): 1,
       
    }
    
    if not image_paths:
        print("No images found in the folder. Please add drone images to 'images' folder.")
        return
    
    # Estimate yield with evaluation
    results = estimate_yield(image_paths, ground_truth=ground_truth)
    print(f"Total Fruit Count Across All Images: {results['total_fruit_count']}")
    print(f"Estimated Yield: {results['total_yield_kg']:.2f} kg")
    if results["metrics"]:
        print(f"Average F1-Score: {results['metrics']['avg_f_score']:.3f}")

if __name__ == "__main__":
    main()
