import os
import pytesseract
from flask import Flask, render_template, request
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageFilter

app = Flask(__name__)

# Set the upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Tesseract executable path (modify this if needed) // as per your pc wheree this tesseract.exe is saved...
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part", 400

        file = request.files['file']

        if file.filename == '':
            return "No selected file", 400

        if file:
            # Save the uploaded PDF to a designated directory
            pdf_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(pdf_path)

            # Convert the PDF to a list of images
            images = convert_from_path(pdf_path, dpi=300, poppler_path=r'C:\Users\91772\Downloads\poppler-0.68.0_x86\poppler-0.68.0\bin')

            marks_per_page = []
            overall_total_marks = 0

            # Process each page from the PDF
            for i, image in enumerate(images):
                full_image_path = f'full_page_image_{i}.png'
                image.save(full_image_path)  # Save full-page image for inspection
                print(f"Saved full-page image: {full_image_path}")

                width, height = image.size
                print(f"Page {i} - Image dimensions: {width}x{height}")

                # Define crop area for the marks region (adjust based on image dimensions)
                crop_area = (1400, 200, 1900, 600)  # Coordinates for the region containing marks
                cropped_image = image.crop(crop_area)

                cropped_image_path = f'cropped_image_page_{i}.png'
                cropped_image.save(cropped_image_path)  # Save cropped image
                print(f"Saved cropped image: {cropped_image_path}")

                # Convert to grayscale and enhance image contrast
                gray_image = cropped_image.convert('L')

                # Apply denoising filter and sharpen the image
                gray_image = gray_image.filter(ImageFilter.MedianFilter())  # Denoising
                enhancer = ImageEnhance.Contrast(gray_image)
                enhanced_image = enhancer.enhance(2)  # Increase contrast

                # Apply sharpening filter to make edges sharper
                enhanced_image = enhanced_image.filter(ImageFilter.SHARPEN)
                
                enhanced_image_path = f'enhanced_image_page_{i}_enhanced.png'
                enhanced_image.save(enhanced_image_path)  # Save enhanced image
                print(f"Saved enhanced image: {enhanced_image_path}")

                # Perform OCR to extract digits
                # custom_config = r'--psm 7 -c tessedit_char_whitelist=0123456789'  # Mode for single digits
                custom_config = r'--oem 1 --psm 7'
                text = pytesseract.image_to_string(enhanced_image, config=custom_config)
                
                print(f"OCR Result for page {i}: {text.strip()}")  # Debugging OCR result

                # Post-process the OCR output to filter digits (1-70)
                page_marks = []
                page_total = 0
                try:
                    extracted_mark = int(text.strip())
                    if 1 <= extracted_mark <= 70:  # Constrain the valid marks range
                        page_marks.append(extracted_mark)
                        page_total += extracted_mark
                    else:
                        page_marks.append("-1")  # Invalid mark outside the range
                except ValueError:
                    page_marks.append("N/A")  # OCR did not return a valid integer

                # Store the marks per page and the total
                marks_per_page.append({"page": i + 1, "marks": page_marks, "total": page_total})
                overall_total_marks += page_total

            # Prepare the result string
            result = f"Marks per Page: {marks_per_page}, Overall Total Marks: {overall_total_marks}"

            # Render the result.html template with the result string
            return render_template('result.html', result=result)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

