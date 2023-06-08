import cv2



# Load image

test_image = cv2.imread('group.jpg')



# Convert to grayscale

test_image_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)



# Initialize the face detector

face_classifier = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

face_cascade = cv2.CascadeClassifier(face_classifier)



# Detect the faces in the image

faces = face_cascade.detectMultiScale(test_image_gray)



# Print number of faces detected

print(f"{len(faces)} faces detected in the image")



# Draw a blue rectangle over every face

for x, y, width, height in faces:

    cv2.rectangle(test_image, (x, y), (x + width, y + height), color=(255,0, 0), thickness=2)



# Save the new image with rectangles drawn

cv2.imwrite("group_detected.jpg", test_image)
