import cv2

cascade_classifier = cv2.CascadeClassifier("C:\\Users\\uppada satwik\\Downloads\\haarcascades\\haarcascade_frontalface_alt_tree.xml")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    gray = cv2.cvtColor(frame, 0)
    detections = cascade_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(detections) > 0:
        (x, y, w, h) = detections[0]
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("Image Classifier", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
