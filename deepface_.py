import cv2
from deepface import DeepFace

# ✅ Direct mapping of emotion → image file
reaction_images = {
    "happy": "happy.jpg",
    "sad": "sad.png",
    "angry": "angry.jpg",
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

    try:
        # DeepFace emotion analysis
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']

        print("Detected emotion:", emotion)

        # ✅ Load mapped reaction image
        img_path = "reaction_images/" + reaction_images.get(emotion, "neutral.jpg")
        img = cv2.imread(img_path)

        if img is not None:
            img = cv2.resize(img, (300, 300))
            cv2.imshow("Reaction Output", img)
        else:
            print("⚠ Image missing:", img_path)

        # Show detected emotion text on webcam
        cv2.putText(frame, emotion.upper(), (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    except Exception as e:
        print("Error in detection:", e)

    cv2.imshow("Live Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
