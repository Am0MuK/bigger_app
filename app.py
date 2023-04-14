from flask import Flask, render_template, request, redirect, url_for
import cv2
import pytesseract
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Define allowed image file extensions
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

# Set the language and config parameters for fine-tuning
lang = "ron" # The language of the text
config = "--psm 6" # Page segmentation mode: Assume a single uniform block of text
config += " --oem 1" # OCR engine mode: LSTM only
config += " -c tessedit_char_whitelist=AĂăBCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890.-," # Character whitelist: Only recognize these characters

def allowed_file(filename):
    """Check if a file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text(img):
    """Extract text from an image"""
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to remove noise
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Perform adaptive thresholding to binarize the image
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Perform morphological operations to remove noise and fill gaps in the text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find contours in the image
    contours, hierarchy = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by their x-coordinate to process from left to right
    contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])

    # Initialize an empty list to store the recognized text
    text_list = []

    # Loop over the contours and recognize the text
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Crop the image to the contour coordinates
        roi = morph[y:y+h, x:x+w]

        # Run Tesseract OCR on the cropped image
        text = pytesseract.image_to_string(roi, lang=lang, config=config)

        # Append the recognized text to the list
        text_list.append(text)

    # Convert the text list to a pandas dataframe
    df = pd.DataFrame(text_list, columns=["text"])
    return df

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/extract_text', methods=['POST'])
def extract_text_route():
    # Check if an image file was uploaded
    if 'image' not in request.files:
        return redirect(request.url)

    file = request.files['image']

    # Check if the file has an allowed extension
    if not allowed_file(file.filename):
        return redirect(request.url)

    # Read the image file and extract the text
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    df = extract_text(img)

   # Save the text to an excel file
    output_file = os.path.join(app.root_path, 'static', 'output_file.xlsx')
    df.to_excel(output_file, index=False)

# Render the extracted text on a new page
    return render_template('result.html', text=df['text'].tolist())