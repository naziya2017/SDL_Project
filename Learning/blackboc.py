import cv2
import pytesseract
from pdf2image import convert_from_path
import os
import numpy as np

# Set up pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

def extract_marks_from_image(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply thresholding to enhance text contrast
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    # Perform OCR, detection of text+digit
    text = pytesseract.image_to_string(Image.fromarray(thresh), config='--psm 6')
    
    # Extract marks from the text
    marks = 0
    for line in text.splitlines():
        if line.isdigit():
            marks = int(line)
            break
    
    return marks

def process_page(image):
    # Get the upper right corner of the page
    height, width = image.shape[:2]
    upper_right_corner = image[0:int(height*0.2), int(width*0.8):width]
    
    # Extract marks from the upper right corner
    marks = extract_marks_from_image(upper_right_corner)
    
    return marks

def main(pdf_path):
    # Convert PDF to images
    pages = convert_from_path(pdf_path, 300)
    
    total_marks = 0
    for i, page in enumerate(pages):
        image_path = f"page_{i}.png"
        page.save(image_path, 'PNG')
        
        # Read image
        image = cv2.imread(image_path)
        
        # Process the page
        marks = process_page(image)
        total_marks += marks
        
        # Cleanup the image file
        os.remove(image_path)
    
    print(f"Total Marks: {total_marks}")

if __name__ == "__main__":
    pdf_path = './samplePdf.pdf'
    main(pdf_path)