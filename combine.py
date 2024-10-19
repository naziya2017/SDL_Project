import os
import sys
import io
import cv2
import numpy as np

# import tensorflow as tf
import pytesseract
from flask import Flask, render_template, request
from pdf2image import convert_from_path
from torchvision import models

# Set up Flask app
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Set tesseract path if needed
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

# Load MNIST model
model_path = 'handwritten_model.h5'
model = tf.keras.models.load_model(model_path)

# Load YOLOv3 model
yolo_model = models.detection.yolo.YOLOv3('yolov3.pt')

# Function to preprocess and predict digit from image
def predict_digit(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = np.invert(img)  # Invert the image for MNIST model format
    img = img / 255.0  # Normalize pixel values
    img = img.reshape(1, 28, 28, 1)  # Reshape for MNIST model input
    prediction = model.predict(img)
    return np.argmax(prediction)

def process_pdf(file):
    pdf_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(pdf_path)

    images = convert_from_path(pdf_path, dpi=300, poppler_path=r'C:\Users\91772\Downloads\poppler-0.68.0_x86\poppler-0.68.0\bin')

    marks_per_page = []
    overall_total_marks = 0

    for i, image in enumerate(images):
        width, height = image.size
        print(f"Processing Page {i+1}, Dimensions: {width}x{height}")

        if i == 0:
            # Crop marks for units (a-e)
            unit_regions = [
                (475, 700, 1340, 800),  # Unit 1 marks
                (475, 800, 1340, 900),  # Unit 2 marks
                (475, 900, 1340, 1000),  # Unit 3 marks
                (475, 1100, 1340, 1200),  # Unit 4 marks
                (475, 1200, 1340, 1300),  # Unit 5 marks
            ]

            page_marks = []
            page_total = 0

            for j, region in enumerate(unit_regions):
                cropped_image = image.crop(region)
                cropped_path = f'unit_{j+1}_page_{i+1}.png'
                cropped_image.save(cropped_path)
                # Resize for MNIST model
                resized_image = cropped_image.resize((28, 28))
                resized_image_path = f'resized_unit_{j+1}_page_{i+1}.png'
                resized_image.save(resized_image_path)
                # Predict digit for each unit
                predicted_digit = predict_digit(resized_image_path)
                page_marks.append(predicted_digit)
                page_total += predicted_digit
                # Append page results
                marks_per_page.append({
                    "marks": page_marks,
                    "total": page_total,
                })

                overall_total_marks += page_total
        else:
            # Crop upper-right corner for marks
            upper_right_crop_area = (1400, 200, 2500, 800) 
            upper_right_cropped_image = image.crop(upper_right_crop_area)
            upper_right_cropped_path = f'upper_right_page_{i+1}.png'
            upper_right_cropped_image.save(upper_right_cropped_path)

            # Perform object detection using YOLOv3
            image_path = upper_right_cropped_path
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            outputs = yolo_model(image)
            for output in outputs:
                for detection in output:
                    scores = detection['scores']
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5 and class_id == 0:  # Filter by class ID (digit) and confidence
                        x, y, w, h = detection['bbox']
                        digit_image = image[y:y+h, x:x+w]
                        break

            # Resize the detected digit for MNIST recognition
            digit_image = cv2.resize(digit_image, (28, 28))

            # Predict the digit using MN 
                        # Normalize and reshape the digit image for the MNIST model
            digit_image = np.invert(digit_image)  # Invert for MNIST
            digit_image = digit_image / 255.0  # Normalize pixel values
            digit_image = digit_image.reshape(1, 28, 28, 1)  # Reshape for model input

            # Predict digit using the MNIST model
            predicted_digit = model.predict(digit_image)
            predicted_digit = np.argmax(predicted_digit)

            # Save the marks detected in the cropped area for the page
            marks_per_page.append({
                "page": i + 1,
                "predicted_digit": predicted_digit
            })

            overall_total_marks += predicted_digit

    return {
        "marks_per_page": marks_per_page,
        "overall_total_marks": overall_total_marks
    }

# Define route for the upload and PDF processing
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            result = process_pdf(file)
            return render_template('result.html', result=result)

    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
