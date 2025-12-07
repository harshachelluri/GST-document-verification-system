# GST-document-verification-system
This is a **Flask-based web application** designed to perform **document verification** on uploaded images. The application checks various aspects of the document to determine its authenticity, such as verifying the presence of an emblem, validating GSTIN, checking date alignment, detecting signature keywords, and more. The app uses **OpenCV**, **pytesseract**, **deep learning**, and other libraries to process and analyze images.

**Note:** This system was developed and deployed as part of a **real client project**, ensuring high standards and real-world applicability.

---  

## Features

### 1. **Emblem Detection**
   - Detects the presence of an emblem by using mirrored template matching techniques to compare the document's emblem with a reference image.

### 2. **GSTIN Validation**
   - Extracts and validates the **GSTIN** (Goods and Services Tax Identification Number) from the document using **OCR (Optical Character Recognition)**.

### 3. **From Date Alignment**
   - Verifies if the "From Date" text is properly aligned within the document, following predefined alignment standards.

### 4. **Signature Keyword Position Check**
   - Detects keywords related to **digital signatures**, such as "Digitally Signed," and checks if they appear in the expected position on the certificate.

### 5. **GST Watermark Detection**
   - Identifies if the document contains a **GST watermark**, ensuring the document has the necessary official mark.

### 6. **QR Code Validation**
   - Detects and decodes any **QR code** present in the document and verifies the URL for accessibility and authenticity.

### 7. **Legal and Trade Name Extraction**
   - Extracts **legal** and **trade names** from the document using OCR and checks them for correctness and expected formatting.

---

## Output

After processing the document, the system will display the results of each verification step along with an **overall verdict**. The possible outcomes are:

- **Genuine**: If the document passes all checks.
- **Fake**: If any check fails. A list of the failed checks will be provided.

### Example Outputtext
GST Certification Verification:

- **Emblem Detected**: Yes
- **GSTIN Validated**: Yes (GSTIN: 27AAAAA0000A1Z5)
- **From Date Alignment**: Yes
- **Signature Position Check**: Yes
- **GST Watermark Detected**: Yes
- **QR Code Validated**: Yes
- **Legal Name Extracted**: XYZ Corporation
- **Trade Name Extracted**: XYZ Traders

**Verdict**: Genuine

![Image](https://github.com/user-attachments/assets/c2e4db7f-06c5-40dd-81c9-40ce0b5caa7c)
