import pytesseract
import cv2

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

# Set the path to the image
image_path = r'C:\Users\91772\OneDrive\Desktop\SDL PROJECT\extra\Learning\danger.png'

# Read the image
img = cv2.imread(image_path)

# Check if the image was loaded successfully
if img is None:
    print("Error: Could not read the image. Check the file path.")
else:
    # Display the image (will not work in Jupyter)
    cv2.imshow('sample', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Extract text from the image
    custom_config = r'--oem 1 --psm 7'
    text = pytesseract.image_to_string(img, config=custom_config)
    #text = pytesseract.image_to_string(img)
    # Print extracted text
    print("Extracted Text:\n", text)
