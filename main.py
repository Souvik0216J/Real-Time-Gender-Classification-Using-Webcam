from keras.models import load_model
from time import sleep
from tensorflow.keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
gender_classifier = load_model(r'best_model.h5')

gender_labels = ['Female', 'Male']

cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    _, frame = cap.read()
    labels = []
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_classifier.detectMultiScale(rgb)

    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi_rgb = rgb[y:y + h, x:x + w]
        roi_rgb = cv2.resize(roi_rgb, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_rgb]) != 0:
            # Preprocess the ROI for the gender classifier
            roi_rgb = roi_rgb.astype('float') / 255.0
            roi_rgb = img_to_array(roi_rgb)
            roi_rgb = np.expand_dims(roi_rgb, axis=0)

            # Make prediction using the gender classifier
            prediction = gender_classifier.predict(roi_rgb)[0]
            label = gender_labels[int(round(prediction[0]))]
            # Add the label and display it on the frame
            label_position = (x, y - 10)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        else:
            # Display 'No Faces' if no faces are detected
            cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Gender Classifier', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()