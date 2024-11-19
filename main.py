import cv2
import warnings
from concurrent.futures import ThreadPoolExecutor
from capture import capture_screen
from detection import detect_objects, adjust_confidence_for_distance
from config import load_config

# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

def main():
    # Load configuration
    cfg = load_config()

    # Extract values from config
    capture_region = cfg["capture_region"]
    crosshair_x = cfg["crosshair"]["x"]
    crosshair_y = cfg["crosshair"]["y"]
    confidence_threshold = cfg["confidence_threshold"]

    with ThreadPoolExecutor(max_workers=3) as executor:
        while True:
            # Capture frame
            future_frame = executor.submit(capture_screen, capture_region)
            frame = future_frame.result()
            display_frame = frame.copy()

            # Detect objects
            future_detections = executor.submit(detect_objects, frame, confidence_threshold)
            detections = future_detections.result()

            detections = adjust_confidence_for_distance(detections)

            print(f"Detections: {len(detections)}")

            # Draw crosshair
            cv2.circle(display_frame, (crosshair_x, crosshair_y), 10, (255, 0, 0), 2)

            for detection in detections[:10]:
                x1, y1, x2, y2, conf, cls = detection[:6]
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                head_height = int((y2 - y1) * 0.15)
                head_x1 = x1
                head_y1 = y1
                head_x2 = x2
                head_y2 = y1 + head_height

                head_center_x = (head_x1 + head_x2) // 2
                head_center_y = (head_y1 + head_y2) // 2

                cv2.rectangle(display_frame, (head_x1, head_y1), (head_x2, head_y2), (0, 255, 255), 2)
                cv2.circle(display_frame, (head_center_x, head_center_y), 5, (0, 0, 255), -1)

                label = f"{conf:.2f}"
                cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                if abs(head_center_x - crosshair_x) < 50 and abs(head_center_y - crosshair_y) < 50:
                    print("Crosshair aligned with head!")
                    cv2.putText(
                        display_frame, "Head Detected!", (crosshair_x - 50, crosshair_y - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
                    )

            cv2.imshow("Detection", display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
