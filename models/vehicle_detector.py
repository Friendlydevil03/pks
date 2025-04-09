import torch
from torchvision.models import detection
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
import numpy as np

class VehicleDetector:
    def __init__(self, confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Force CPU usage to avoid CUDA issues
        self.device = torch.device('cpu')
        print(f"Using device: {self.device}")

        try:
            # Load pre-trained model with proper error handling
            print("Loading ML model...")
            self.model = detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully")

            # COCO class names (we're interested in vehicles)
            self.classes = [
                'background', 'person', 'bicycle', 'car', 'motorcycle',
                'airplane', 'bus', 'train', 'truck', 'boat'
            ]
            self.vehicle_classes = [2, 3, 5, 6, 7, 8]  # Indices of vehicle classes
        except Exception as e:
            print(f"Error loading ML model: {str(e)}")
            raise

    # Fix for models/vehicle_detector.py - detect_vehicles method

    def detect_vehicles(self, frame):
        """Detect vehicles in a frame"""
        # Handle invalid input
        if frame is None or frame.size == 0:
            print("Error: Empty frame received")
            return []

        try:
            # Convert frame to tensor
            img = torch.from_numpy(frame.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
            img = img.to(self.device)

            with torch.no_grad():
                predictions = self.model(img)

            # Extract detections
            boxes = predictions[0]['boxes'].cpu().numpy().astype(int)
            scores = predictions[0]['scores'].cpu().numpy()
            labels = predictions[0]['labels'].cpu().numpy()

            # Filter by confidence and vehicle classes
            vehicle_detections = []
            for box, score, label in zip(boxes, scores, labels):
                if score > self.confidence_threshold and label in self.vehicle_classes:
                    # Return in expected format: (box, score, label)
                    vehicle_detections.append(([box[0], box[1], box[2], box[3]], float(score), int(label)))

            return vehicle_detections

        except Exception as e:
            print(f"Error in vehicle detection: {str(e)}")
            # Return empty list on error to prevent crashing
            return []