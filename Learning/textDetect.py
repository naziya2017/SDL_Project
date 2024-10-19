import pytesseract
from PIL import Image
import cv2
import numpy as np
import os

# Specify the path to the Tesseract-OCR executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

# Specify the full path to the image file
image_path = r'C:\Users\91772\OneDrive\Desktop\SDL PROJECT\extra\Learning\cropped_image_page_4.png'

# Check if the image file exists
if not os.path.exists(image_path):
    print(f"Error: The file at {image_path} does not exist.")
else:
    try:
        # Open the image file
        img = Image.open(image_path)

        # Convert the image to grayscale using OpenCV
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)

        # Apply thresholding to enhance text contrast
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # Perform OCR to detect text and digits
        text = pytesseract.image_to_string(thresh, lang='eng')
        print("Extracted text:", text)

        # Detecting characters and getting bounding box coordinates
        boxes = pytesseract.image_to_boxes(img)
        hImg, wImg = img.size

        # Convert the image to a numpy array for drawing with OpenCV
        img_array = np.array(img)

        for b in boxes.splitlines():
            b = b.split(' ')
            char, x, y, w, h = b[0], int(b[1]), int(b[2]), int(b[3]), int(b[4])

            # Draw a rectangle around the detected character
            cv2.rectangle(img_array, (x, hImg), (w, hImg), (0, 0, 255), 2)
            # Optionally, add text label (character) near the rectangle
            cv2.putText(img_array, char, (x, hImg - y + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255), 2)

        # Show the image with bounding boxes
        cv2.imshow('Result', img_array)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error occurred: {e}")
