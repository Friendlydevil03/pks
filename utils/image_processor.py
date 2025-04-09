import cv2
import numpy as np
from PIL import Image, ImageTk


def preprocess_frame_for_parking_detection(img):
    """Preprocess a frame for parking space detection"""
    # Convert to grayscale
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    # Apply threshold
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 25, 16)
    # Apply blur again to smooth edges
    imgBlur = cv2.medianBlur(imgThreshold, 5)
    # Dilate to fill in holes
    kernel = np.ones((3, 3), np.uint8)
    imgDilate = cv2.dilate(imgBlur, kernel, iterations=1)
    return imgDilate


def process_parking_spaces(img_pro, img, pos_list, threshold, debug=False):
    """Process and mark parking spaces in the image"""
    space_counter = 0

    # Add debug info
    if debug:
        img_height, img_width = img.shape[:2]
        cv2.putText(img, f"Image size: {img_width}x{img_height}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    for i, (x, y, w, h) in enumerate(pos_list):
        # Ensure coordinates are within image bounds
        if (y >= 0 and y + h < img_pro.shape[0] and x >= 0 and x + w < img_pro.shape[1]):
            # Add box number and coordinates in debug mode
            if debug:
                coord_text = f"Box {i}: ({x},{y})"
                cv2.putText(img, coord_text, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

            # Draw ID number for each space
            cv2.putText(img, str(i), (x + 5, y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            img_crop = img_pro[y:y + h, x:x + w]
            count = cv2.countNonZero(img_crop)

            if count < threshold:
                color = (0, 255, 0)  # Green for free
                space_counter += 1
            else:
                color = (0, 0, 255)  # Red for occupied

            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, str(count), (x, y + h - 3), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)

    free_spaces = space_counter
    total_spaces = len(pos_list)
    occupied_spaces = total_spaces - free_spaces

    return img, free_spaces, occupied_spaces, total_spaces


# Replace the old vehicle detection function with your new code
def detect_vehicles_traditional(current_frame, prev_frame, line_height, min_contour_width, min_contour_height, offset,
                                matches, vehicles_count):
    """
    Detect vehicles using traditional computer vision methods based on the provided code
    """
    # Create a copy of the current frame for displaying results
    display_frame = current_frame.copy()

    # Calculate absolute difference between frames
    d = cv2.absdiff(prev_frame, current_frame)
    grey = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)

    # Apply blur and threshold
    blur = cv2.GaussianBlur(grey, (5, 5), 0)
    ret, th = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

    # Apply dilation and morphology operations
    dilated = cv2.dilate(th, np.ones((3, 3)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, h = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Make a deep copy of matches list to avoid modifying during iteration
    matches_copy = matches.copy()

    # Draw detection line
    cv2.line(display_frame, (0, line_height), (display_frame.shape[1], line_height), (0, 255, 0), 2)

    # Process each contour
    for (i, c) in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(c)
        contour_valid = (w >= min_contour_width) and (h >= min_contour_height)

        if not contour_valid:
            continue

        # Draw rectangle around vehicle
        cv2.rectangle(display_frame, (x - 10, y - 10), (x + w + 10, y + h + 10), (255, 0, 0), 2)

        # Calculate centroid
        cx = x + w // 2
        cy = y + h // 2
        centroid = (cx, cy)

        # Add centroid to matches list
        matches_copy.append(centroid)

        # Draw centroid
        cv2.circle(display_frame, centroid, 5, (0, 255, 0), -1)

    # Count vehicles crossing the line
    new_vehicles_count = vehicles_count
    new_matches = []

    for (x, y) in matches_copy:
        # Check if centroid is near the line
        if line_height - offset < y < line_height + offset:
            new_vehicles_count += 1
        else:
            # Keep centroids that haven't crossed the line
            new_matches.append((x, y))

    # Display vehicle count
    cv2.putText(display_frame, f"Total Vehicle Detected: {new_vehicles_count}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 170, 0), 2)

    return display_frame, new_matches, new_vehicles_count


# Fix for utils/image_processor.py - complete process_ml_detections function

def process_ml_detections(frame, detections, line_height, offset, matches, vehicles_count, class_names):
    """Process detections from ML model"""
    display_frame = frame.copy()

    # Draw detection line
    cv2.line(display_frame, (0, line_height), (display_frame.shape[1], line_height), (0, 255, 0), 2)

    # Make a deep copy of matches list
    matches_copy = matches.copy() if matches is not None else []

    # Handle case where detections might be None
    if detections is None:
        detections = []

    # Process each detection
    for detection in detections:
        # Ensure detection has the expected format
        if len(detection) < 5:  # We need at least box, score, label
            continue

        box, score, label = detection[:5]  # Unpack the first 5 elements

        # Ensure box is valid
        if len(box) < 4:
            continue

        x1, y1, x2, y2 = box

        # Draw bounding box
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Calculate centroid
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        centroid = (cx, cy)

        # Add label
        class_name = class_names[label] if label < len(class_names) else f"Class {label}"
        cv2.putText(display_frame, f"{class_name}: {score:.2f}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Add centroid
        matches_copy.append(centroid)

        # Draw centroid
        cv2.circle(display_frame, centroid, 5, (0, 0, 255), -1)

    # Count vehicles crossing the line
    new_vehicles_count = vehicles_count
    new_matches = []

    for (x, y) in matches_copy:
        if line_height - offset < y < line_height + offset:
            new_vehicles_count += 1
        else:
            new_matches.append((x, y))

    # Display vehicle count
    cv2.putText(display_frame, f"Total Vehicle Detected: {new_vehicles_count}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 170, 0), 2)

    # Return all required values
    return display_frame, new_matches, new_vehicles_count