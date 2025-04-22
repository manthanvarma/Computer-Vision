import cv2

def detect_face_eyes_smile():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

    cap = cv2.VideoCapture(0) # webcam

    if not cap.isOpened():
        print("error: failed to capture webcam")
        return
    
    print("press 'q' to exit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("failed to grab frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5) # face detection

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2) # draw face rectangle

            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            # detect eyes
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255,0,0), 2)

            # detect smile
            smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=20)
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(roi_color, (sx,sy), (sx+sw, sy+sh), (0,0,255), 2)
            
        cv2.imshow('Face, Eyes, Smile Detection', frame) # display

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_face_eyes_smile()

