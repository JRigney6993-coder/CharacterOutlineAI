import cv2
import torch

# Load YOLOv5 model 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', force_reload=True).to(device)

def detect_objects(frame, confidence_threshold=0.4):
    """Detect objects in the given frame using YOLOv5."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    frame_resized = cv2.resize(frame_rgb, (640, 360))
    results = model(frame_resized)
    detections = results.xyxy[0].cpu().numpy()
    detections = detections[detections[:, 4] > confidence_threshold]
    scale_x = frame.shape[1] / 640
    scale_y = frame.shape[0] / 360
    detections[:, :4] *= [scale_x, scale_y, scale_x, scale_y]
    return detections

def adjust_confidence_for_distance(detections):
    """Dynamically adjust confidence thresholds based on object size."""
    adjusted_detections = []
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        box_width = x2 - x1
        box_height = y2 - y1

        if box_width > 50 and box_height > 50:
            adjusted_detections.append(det)
        elif box_width > 30 and box_height > 30 and conf > 0.35:
            adjusted_detections.append(det)
        elif box_width > 15 and box_height > 15 and conf > 0.25:
            adjusted_detections.append(det)

    return adjusted_detections
