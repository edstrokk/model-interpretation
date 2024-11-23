import os
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

def process_images_with_yolo_seg(input_folder, model_path, output_folder):
    """
    Processes images using a YOLO segmentation model, normalizes polygon points,
    and adjusts drawing polygons using normalized coordinates.

    Args:
        input_folder (str): Path to the folder containing input images.
        model_path (str): Path to the trained YOLO segmentation model (.pt file).
        output_folder (str): Path to the folder where the output images with polygons will be saved.

    Returns:
        dict: A dictionary with normalized points and bounding box for the card class.
    """
    # Load the YOLO model
    model = YOLO(model_path)

    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    results_dict = {}

    # Iterate through all image files in the input folder
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)

        # Ensure the file is an image
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue

        # Load the image
        image = cv2.imread(file_path)
        if image is None:
            print(f"Failed to read image: {file_path}")
            continue

        img_height, img_width = image.shape[:2]

        print("image shape: ", img_height, img_width)
        # Run inference
        results = model(image)

        for result in results:
            if result.masks is None:
                print(f"No masks detected for {filename}")
                continue

            for idx, cls in enumerate(result.boxes.cls):
                cls_id = int(cls)  # Class ID
                mask = result.masks.data[idx].cpu().numpy()  # Extract the raw mask as a NumPy array
                mask_height, mask_width = mask.shape
                print("mask shape: ", mask_height, mask_width)
                # Convert mask to binary
                binary_mask = (mask > 0.5).astype(np.uint8) * 255

                # Detect contours
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Approximate contours to polygons and normalize points
                for contour in contours:
                    epsilon = 0.002 * cv2.arcLength(contour, True)
                    polygon = cv2.approxPolyDP(contour, epsilon, True)
                    print("polygon shape: ", polygon.shape)
                    # Normalize the polygon points
                    normalized_points = [
                        [point[0][0] / mask_width, point[0][1] / mask_height]
                        for point in polygon
                    ]

                    # Draw polygon using normalized coordinates

                    scaled_polygon = np.array([
                        [int(point[0] * img_width), int(point[1] * img_height)]
                        for point in normalized_points
                    ], dtype=np.int32)

                    # Draw polygon based on class
                    if cls_id == 0:  # Class 0: card
                        cv2.polylines(image, [scaled_polygon], True, (0, 0, 255), 2)  # Red for card
                        # Get bounding box for card
                        x, y, w, h = cv2.boundingRect(scaled_polygon)
                        results_dict[filename] = {
                            "seg": scaled_polygon,
                            "bbox": [x, y, w, h]
                        }
                    elif cls_id == 1:  # Class 1: slab
                        cv2.polylines(image, [scaled_polygon], True, (255, 0, 0), 2)  # Blue for slab

        # Save the image with drawn polygons
        output_image_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_image_path, image)
        print(f"Processed and saved: {output_image_path}")

    return results_dict




if __name__ == "__main__":
    input_folder = "/Users/eduardstrok/PycharmProjects/yolo11_interp/images"
    model_path = "/Users/eduardstrok/PycharmProjects/yolo11_interp/model.pt"
    output_folder = "/Users/eduardstrok/PycharmProjects/yolo11_interp/output"
    results = process_images_with_yolo_seg(input_folder, model_path, output_folder)

    # Print results for all processed images
    for filename, result in results.items():
        print(f"{filename}: Normalized Points - {result['seg']}, Bounding Box - {result['bbox']}")
