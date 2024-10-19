import cv2
import pytesseract
import pdf2image
import os
import tempfile

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

def extract_text_from_image(image_path):
    try:
        img = cv2.imread(image_path)
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(img, config=custom_config)
        return text
    except Exception as e:
        print(f"Error extracting text from image: {e}")
        return ""

def extract_text_from_pdf_page(pdf_path, page_number):
    try:
        images = pdf2image.convert_from_path(pdf_path)
        if page_number < 1 or page_number > len(images):
            print(f"Page number {page_number} is out of range")
            return ""
        image_path = f'page_{page_number}.jpg'
        images[page_number - 1].save(image_path, 'JPEG')
        text = extract_text_from_image(image_path)
        os.remove(image_path)
        return text
    except Exception as e:
        print(f"Error extracting text from PDF page: {e}")
        return ""

def extract_text_from_pdf_range(pdf_path, start_page, end_page):
    try:
        if start_page > end_page:
            print("Start page cannot be greater than end page")
            return ""
        total_text = ''
        for page_number in range(start_page, end_page + 1):
            text = extract_text_from_pdf_page(pdf_path, page_number)
            total_text += text + '\n'
        return total_text
    except Exception as e:
        print(f"Error extracting text from PDF range: {e}")
        return ""

pdf_path = './samplePdf.pdf'
start_page = 2
end_page = 10 
text = extract_text_from_pdf_range(pdf_path, start_page, end_page)
print(f'Text from page {start_page} to {end_page}: {text}')


