import cv2
from fer import FER

# ✅ Load emotion detector
detector = FER(mtcnn=True)

# ✅ Direct mapping of emotion → image file
reaction_images = {
    "happy": "happy.jpg",
    "sad": "sad.png",
    "angry": "disgust.jpg",
    "surprise": "suprise.png",
    "neutral": "neutral.jpg",
    "disgust": "disgust.jpg",
    "fear": "fear.jpg"
}

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect emotions
    result = detector.detect_emotions(frame)

    if result:
        emotions = result[0]["emotions"]
        emotion = max(emotions, key=emotions.get)  # Most confident emotion

        print("Detected emotion:", emotion)  # Debug output ✅

        # ✅ Load correct reaction image safely
        img_path = "reaction_images/" + reaction_images.get(emotion, "neutral.jpg")
        img = cv2.imread(img_path)

        if img is not None:
            img = cv2.resize(img, (300, 300))
            cv2.imshow("Reaction Output", img)
        else:
            print("⚠ Image missing:", img_path)

        # Show detected emotion text
        cv2.putText(frame, emotion.upper(), (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Live Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()
