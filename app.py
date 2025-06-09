import cv2
import pytesseract
import numpy as np
import re
from pytesseract import Output
from flask import Flask, request, render_template, jsonify
import os
from werkzeug.utils import secure_filename
import base64
from pyzbar.pyzbar import decode
from PIL import Image
import requests

# Flask app setup
app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Helper Functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def encode_image(img):
    """Encode an image to base64 for JSON response."""
    if img is None:
        return None
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

# --------------------- [1] Emblem Detection via Mirrored Template Matching ---------------------
def detect_emblem_template_match(emblem_path, bill_path):
    try:
        emblem = cv2.imread(emblem_path, cv2.IMREAD_GRAYSCALE)
        bill = cv2.imread(bill_path)
        bill_gray = cv2.cvtColor(bill, cv2.COLOR_BGR2GRAY) if len(bill.shape) == 3 else bill

        if emblem is None or bill is None:
            return False, "Emblem or bill image not found.", encode_image(bill)

        if emblem.shape[0] > bill.shape[0] or emblem.shape[1] > bill.shape[1]:
            return False, "Emblem template too large.", encode_image(bill)

        emblem_mirrored = cv2.flip(emblem, 1)
        h, w = bill_gray.shape
        bill_roi = bill_gray[0:int(h * 0.2), w//2 - int(w * 0.15):w//2 + int(w * 0.15)]

        if emblem_mirrored.shape[0] > bill_roi.shape[0] or emblem_mirrored.shape[1] > bill_roi.shape[1]:
            return False, "Mirrored emblem too large for ROI.", encode_image(bill)

        res = cv2.matchTemplate(bill_roi, emblem_mirrored, cv2.TM_CCOEFF_NORMED)
        _, conf, _, loc = cv2.minMaxLoc(res)

        result_img = bill.copy()
        if conf > 0.7:
            top_left = (loc[0] + w//2 - int(w * 0.15), loc[1])
            bottom_right = (top_left[0] + emblem_mirrored.shape[1], top_left[1] + emblem_mirrored.shape[0])
            cv2.rectangle(result_img, top_left, bottom_right, (0, 255, 0), 2)
            return True, f"Confidence: {conf:.4f}", encode_image(result_img)
        return False, "Emblem mismatch", encode_image(result_img)
    except Exception as e:
        return False, f"Error in emblem detection: {str(e)}", encode_image(cv2.imread(bill_path))

# --------------------- [2] GSTIN Extraction and Validation ---------------------
def validate_gstin(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            return False, "Image not found.", encode_image(image)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        config = r'--oem 3 --psm 6'
        data = pytesseract.image_to_data(thresh, output_type=Output.DICT, config=config)

        gstin_pattern = r'[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}'
        for i, word in enumerate(data['text']):
            if word.strip():
                candidate = word.strip().replace(" ", "").replace("\n", "").upper()
                if re.fullmatch(gstin_pattern, candidate):
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(image, "GSTIN", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    return True, f"Valid GSTIN: {candidate}", encode_image(image)
        return False, "GSTIN not found or invalid.", encode_image(image)
    except Exception as e:
        return False, f"Error in GSTIN validation: {str(e)}", encode_image(cv2.imread(image_path))

# --------------------- [3] From Date Alignment Verification ---------------------
def check_from_date_alignment(image_path):
    try:
        true_x, true_y, true_w, true_h = 527, 933, 94, 26
        tolerance = 15

        img = cv2.imread(image_path)
        if img is None:
            return False, "Image not found.", encode_image(img)

        data = pytesseract.image_to_data(img, output_type=Output.DICT)
        for i in range(len(data['text']) - 1):
            if data['text'][i].strip().lower() == "from":
                for j in range(i + 1, min(i + 5, len(data['text']))):
                    next_word = data['text'][j].strip()
                    if next_word and any(char.isdigit() for char in next_word):
                        x, y, w, h = data['left'][j], data['top'][j], data['width'][j], data['height'][j]
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(img, "From Date", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        if (
                            abs(x - true_x) > tolerance or abs(y - true_y) > tolerance or
                            abs(w - true_w) > tolerance or abs(h - true_h) > tolerance
                        ):
                            return False, f"Misaligned From Date: ({x},{y},{w},{h}) Expected: ({true_x},{true_y},{true_w},{true_h})", encode_image(img)
                        return True, "From Date is correctly aligned.", encode_image(img)
                break
        return False, "From Date not found", encode_image(img)
    except Exception as e:
        return False, f"Error in From Date alignment: {str(e)}", encode_image(cv2.imread(image_path))

# --------------------- [4] Signature Keyword Position Check ---------------------
def preprocess(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None, None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return img, thresh
    except Exception as e:
        return None, None

def extract_signature_keywords(thresh_img, original_img):
    try:
        if thresh_img is None:
            return {}, original_img

        data = pytesseract.image_to_data(thresh_img, output_type=Output.DICT)
        keywords = [
            "signature not verifiedby",
            "digitally signed by",
            "ds goods and services tax",
            "network 07",
            "date:verified"
        ]
        boxes = {}
        result_img = original_img.copy()

        text_list = [w.strip().lower() for w in data['text']]
        for key in keywords:
            key_words = key.split()
            for i in range(len(text_list) - len(key_words) + 1):
                if text_list[i:i+len(key_words)] == key_words:
                    x_vals = [data['left'][j] for j in range(i, i+len(key_words))]
                    y_vals = [data['top'][j] for j in range(i, i+len(key_words))]
                    w_vals = [data['width'][j] for j in range(i, i+len(key_words))]
                    h_vals = [data['height'][j] for j in range(i, i+len(key_words))]
                    x, y = min(x_vals) // 2, min(y_vals) // 2
                    w = (max([x_vals[j] + w_vals[j] for j in range(len(w_vals))]) - x) // 2
                    h = (max([y_vals[j] + h_vals[j] for j in range(len(h_vals))]) - y) // 2
                    boxes[key] = (x, y, w, h)
                    cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(result_img, key, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    break
        return boxes, result_img
    except Exception as e:
        return {}, original_img

def compare_positions(ref_boxes, test_boxes, test_path, tolerance=200):
    try:
        test_img = cv2.imread(test_path)
        result_img = test_img.copy()
        for key in test_boxes:
            x, y, w, h = test_boxes[key]
            cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(result_img, key, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        for key in ref_boxes:
            if key not in test_boxes:
                return False, f"Keyword '{key}' not found", encode_image(result_img)
            rx, ry, _, _ = ref_boxes[key]
            tx, ty, _, _ = test_boxes[key]
            if abs(rx - tx) > tolerance or abs(ry - ty) > tolerance:
                return False, f"Keyword '{key}' misaligned: Ref ({rx},{ry}) vs Test ({tx},{ty})", encode_image(result_img)
        return True, "All key signature elements are aligned correctly.", encode_image(result_img)
    except Exception as e:
        return False, f"Error in signature comparison: {str(e)}", encode_image(cv2.imread(test_path))

# --------------------- [5] GST Watermark Detection ---------------------
def detect_watermark_GST(main_image_path, template_path, threshold=0.3):
    try:
        main_img = cv2.imread(main_image_path)
        main_img_gray = cv2.cvtColor(main_img, cv2.COLOR_BGR2GRAY) if len(main_img.shape) == 3 else main_img
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

        if main_img is None or template is None:
            return False, "Image or template not loaded.", encode_image(main_img)

        if template.shape[0] > main_img.shape[0] or template.shape[1] > main_img.shape[1]:
            scale_factor = min(main_img.shape[0] / template.shape[0], main_img.shape[1] / template.shape[1], 0.5)
            template = cv2.resize(template, (0, 0), fx=scale_factor, fy=scale_factor)

        h, w = main_img_gray.shape
        header_roi = main_img_gray[int(h * 0.05):int(h * 0.15), int(w * 0.1):int(w * 0.5)]
        result = cv2.matchTemplate(header_roi, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        result_img = main_img.copy()
        if max_val >= threshold:
            top_left = (max_loc[0] + int(w * 0.1), max_loc[1] + int(h * 0.05))
            bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
            cv2.rectangle(result_img, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(result_img, "GST Watermark", (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            return True, f"Watermark 'GST' detected. Confidence: {max_val:.4f}", encode_image(result_img)
        return False, "Watermark 'GST' not found", encode_image(result_img)
    except Exception as e:
        return False, f"Error in GST watermark detection: {str(e)}", encode_image(cv2.imread(main_image_path))

# --------------------- [6] QR Code Detection and Validation ---------------------
def detect_and_decode_qr(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            return False, "Could not load the image.", encode_image(image)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        scale_percent = 150
        width = int(gray.shape[1] * scale_percent / 100)
        height = int(gray.shape[0] * scale_percent / 100)
        resized = cv2.resize(thresh, (width, height), interpolation=cv2.INTER_AREA)

        qr_codes = decode(resized)
        if not qr_codes:
            return False, "No QR code found in the image.", encode_image(image)

        qr_data = qr_codes[0].data.decode('utf-8')
        result_img = image.copy()
        for qr in qr_codes:
            points = qr.polygon
            if len(points) > 4:
                hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
                points = hull
            pts = np.array([(point.x // (scale_percent // 100), point.y // (scale_percent // 100)) for point in points], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(result_img, [pts], True, (0, 255, 0), 2)
            cv2.putText(result_img, "QR Code", (pts[0][0][0], pts[0][0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        url_pattern = re.compile(r'^(https?://[^\s/$.?#].[^\s]*)$')
        if url_pattern.match(qr_data):
            try:
                response = requests.head(qr_data, timeout=5)
                if response.status_code == 200:
                    return True, f"Valid QR Code: URL {qr_data} is accessible.", encode_image(result_img)
                return False, f"Invalid QR Code: URL {qr_data} returned status code {response.status_code}.", encode_image(result_img)
            except requests.RequestException:
                return False, f"Invalid QR Code: URL {qr_data} is not accessible.", encode_image(result_img)
        return True, f"Valid QR Code: Data = {qr_data} (Non-URL, assumed valid if readable).", encode_image(result_img)
    except Exception as e:
        return False, f"Error processing QR code: {str(e)}", encode_image(cv2.imread(image_path))

# --------------------- [7] Legal and Trade Name Extraction ---------------------
def extract_text_from_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return "Could not load image.", img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(thresh, config=config)
        return text, img
    except Exception as e:
        return f"Error in text extraction: {str(e)}", cv2.imread(image_path)

def clean_line(line):
    cleaned = re.sub(r'^[^a-zA-Z0-9]+', '', line)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def extract_value(line):
    cleaned_line = re.sub(r'cs\s*legal\s*name\s*\|?|legal\s*name\s*\|?|\(?\s*if\s*any\s*\)?|P41\.\s*\|?', '', line, flags=re.IGNORECASE).strip()
    parts = re.split(r'[=:,]', cleaned_line, maxsplit=1)
    return parts[1].strip() if len(parts) > 1 else cleaned_line.strip()

def find_relevant_lines(text, img):
    legal_line = None
    trade_line = None
    result_img = img.copy()
    lines = text.split('\n')
    for line in lines:
        line_clean = clean_line(line).lower()
        if re.search(r'legal\s*name|legetneme', line_clean):
            legal_line = extract_value(line)
        elif re.search(r'trade\s*name.*if\s*any|ea\s*trade\s*name', line_clean):
            trade_line = extract_value(line)
    return legal_line, trade_line, result_img

# --------------------- Master Function ---------------------
def run_all_checks(test_path, emblem_path="uploads/emblem.jpg", reference_path="Uploads/2.jpg", 
                  gst_template_path="Uploads/chervic__.jpg"):
    results = []
    failed_checks = []

    # Emblem Detection
    result, msg, img = detect_emblem_template_match(emblem_path, test_path)
    results.append({"check": "Emblem Detection", "result": result, "message": msg, "image": img})
    if not result:
        failed_checks.append("Emblem Detection")

    # GSTIN Validation
    result, msg, img = validate_gstin(test_path)
    results.append({"check": "GSTIN Validation", "result": result, "message": msg, "image": img})
    if not result:
        failed_checks.append("GSTIN Validation")

    # From Date Alignment
    result, msg, img = check_from_date_alignment(test_path)
    results.append({"check": "From Date Alignment", "result": result, "message": msg, "image": img})
    if not result:
        failed_checks.append("From Date Alignment")

    # Signature Keywords Check
    ref_img, ref_thresh = preprocess(reference_path)
    test_img, test_thresh = preprocess(test_path)
    ref_boxes, _ = extract_signature_keywords(ref_thresh, ref_img)
    test_boxes, test_img = extract_signature_keywords(test_thresh, test_img)
    result, msg, img = compare_positions(ref_boxes, test_boxes, test_path)
    results.append({"check": "Signature Keywords Check", "result": result, "message": msg, "image": img})
    if not result:
        failed_checks.append("Signature Keywords Check")

    # GST Watermark Detection
    result, msg, img = detect_watermark_GST(test_path, gst_template_path)
    results.append({"check": "Watermark GST Detection", "result": result, "message": msg, "image": img})
    if not result:
        failed_checks.append("Watermark GST Detection")

    # QR Code Detection
    result, msg, img = detect_and_decode_qr(test_path)
    results.append({"check": "QR Code Detection", "result": result, "message": msg, "image": img})
    if not result:
        failed_checks.append("QR Code Detection")

    # Legal and Trade Name Extraction
    text, img = extract_text_from_image(test_path)
    legal_line, trade_line, result_img = find_relevant_lines(text, img)
    result = bool(legal_line or trade_line)
    results.append({
        "check": "Legal and Trade Name Extraction",
        "result": result,
        "message": f"Legal Name: {legal_line or 'Not found'}, Trade Name: {trade_line or 'Not found'}",
        "image": encode_image(result_img)
    })
    if not result:
        failed_checks.append("Legal and Trade Name Extraction")

    # Verdict
    verdict = f"FAKE - Failed checks: {', '.join(failed_checks)}" if failed_checks else "Document appears to be GENUINE."

    return results, verdict

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        emblem_path = os.path.join(app.config['UPLOAD_FOLDER'], 'emblem.jpg')
        reference_path = os.path.join(app.config['UPLOAD_FOLDER'], '2.jpg')
        gst_template_path = os.path.join(app.config['UPLOAD_FOLDER'], 'chervic__.jpg')

        for path in [emblem_path, reference_path, gst_template_path]:
            if not os.path.exists(path):
                return jsonify({'error': f'Reference file not found: {path}'}), 400

        results, verdict = run_all_checks(file_path, emblem_path, reference_path, gst_template_path)
        return jsonify({'results': results, 'verdict': verdict})
    return jsonify({'error': 'Invalid file format'}), 400

if __name__ == "__main__":
    app.run(debug=True)