import cv2

imgpath = "reaction_images/angry.jpg"
img = cv2.imread(imgpath)

if img is None:
    print("❌ Image not found! Check path:", imgpath)
else:
    img = cv2.resize(img, (300, 300))
    cv2.imshow("Reaction Output", img)

    cv2.waitKey(0)  # 👈 keeps window open until key press
    cv2.destroyAllWindows()
