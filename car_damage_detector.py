import cv2
import math
import cvzone
from ultralytics import YOLO

class CarDamageDetector:
    def __init__(self, model_path):
        """
        Initialize the Car Damage Detector
        Args:
            model_path (str): Path to the YOLO model weights
        """
        self.model = YOLO(model_path)
        self.class_labels = [
            'Bodypanel-Dent', 'Front-Windscreen-Damage', 'Headlight-Damage',
            'Rear-windscreen-Damage', 'RunningBoard-Dent', 'Sidemirror-Damage',
            'Signlight-Damage', 'Taillight-Damage', 'bonnet-dent', 'boot-dent',
            'doorouter-dent', 'fender-dent', 'front-bumper-dent', 'pillar-dent',
            'quaterpanel-dent', 'rear-bumper-dent', 'roof-dent'
        ]
        self.confidence_threshold = 0.3

    def detect_damage(self, image):
        """
        Detect car damage in the given image
        Args:
            image: Input image (numpy array)
        Returns:
            tuple: Processed image and list of detections
        """
        results = self.model(image)
        detections = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                
                # Get confidence and class
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls_idx = int(box.cls[0])

                if conf > self.confidence_threshold:
                    detection = {
                        'bbox': (x1, y1, w, h),
                        'class': self.class_labels[cls_idx],
                        'confidence': conf
                    }
                    detections.append(detection)
                    
                    # Draw bounding box and label
                    cvzone.cornerRect(image, (x1, y1, w, h), t=2)
                    cvzone.putTextRect(
                        image,
                        f'{self.class_labels[cls_idx]} {conf}',
                        (x1, y1 - 10),
                        scale=0.8,
                        thickness=1,
                        colorR=(255, 0, 0)
                    )

        return image, detections

def main():
    # Initialize detector
    detector = CarDamageDetector("Weights/best.pt")
    
    # Load and process image
    image_path = "Media/dent_1.jpg"
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # Detect damage
    processed_img, detections = detector.detect_damage(img)
    
    # Display results
    cv2.imshow("Car Damage Detection", processed_img)
    
    # Print detected damages
    print("\nDetected Damages:")
    for i, det in enumerate(detections, 1):
        print(f"{i}. {det['class']} (Confidence: {det['confidence']:.2f})")

    # Wait for 'q' to quit
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cv2.waitKey(1)  # Additional waitKey to handle buffer

if __name__ == "__main__":
    main()

detector = CarDamageDetector("Weights/best.pt")
image = cv2.imread("Media/dent_1.jpg")
processed_img, detections = detector.detect_damage(image)