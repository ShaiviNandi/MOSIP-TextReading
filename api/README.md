# MOSIP Text Reading API

A comprehensive OCR and data verification API that supports both handwritten and printed text extraction from documents, with advanced verification capabilities.

## Features

### API 1: OCR Extraction API (`/api/v1/ocr/extract`)
- ✅ Accepts scanned PDFs and images (PNG, JPG, JPEG, TIFF)
- ✅ Supports English language text extraction
- ✅ Handles both handwritten and printed text
- ✅ Extracts structured form fields (name, address, phone, email, etc.)
- ✅ Returns confidence scores for each extracted field
- ✅ Multi-page PDF support

### API 2: Data Verification API (`/api/v1/verification/compare`)
- ✅ Compares form-filled data against original scanned documents
- ✅ Provides field-by-field match analysis
- ✅ Returns similarity scores and confidence levels
- ✅ Highlights mismatches and potential OCR errors
- ✅ Overall verification status (VERIFIED/PARTIALLY_VERIFIED/VERIFICATION_FAILED)

## Supported Document Types

- **ID Cards**: Identity documents with personal information
- **Forms**: Application forms with structured fields
- **Certificates**: Educational or professional certificates
- **Any document** with the following field types:
  - First Name, Middle Name, Last Name
  - Gender, Date of Birth
  - Address (Line 1, Line 2, City, State, Pin Code)
  - Phone Number, Email ID

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ShaiviNandi/MOSIP-TextReading.git
   cd MOSIP-TextReading/api
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the server**:
   ```bash
   python api.py
   ```

The server will start on `http://localhost:5000`

## API Endpoints

### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "ocr_ready": true,
  "timestamp": "2025-09-19T10:30:00"
}
```

### 1. OCR Extraction API

**Endpoint:** `POST /api/v1/ocr/extract`

**Description:** Extract text and form fields from uploaded documents

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: Form data with file upload

**Parameters:**
- `file` (required): Document file (PDF, PNG, JPG, JPEG, TIFF)
- `include_raw` (optional): Set to "true" to include raw OCR regions in response

**Example using curl:**
```bash
curl -X POST \
  http://localhost:5000/api/v1/ocr/extract \
  -F "file=@sample_form.pdf" \
  -F "include_raw=true"
```

**Example using Python:**
```python
import requests

url = "http://localhost:5000/api/v1/ocr/extract"
with open("sample_form.pdf", "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)
    result = response.json()
    print(result)
```

**Response:**
```json
{
  "status": "success",
  "extracted_fields": {
    "First Name": {
      "value": "John",
      "confidence": 0.85,
      "type": "extracted"
    },
    "Last Name": {
      "value": "Doe",
      "confidence": 0.90,
      "type": "extracted"
    },
    "Email ID": {
      "value": "john.doe@email.com",
      "confidence": 0.75,
      "type": "extracted"
    }
  },
  "metadata": {
    "total_regions_detected": 15,
    "fields_extracted": 8,
    "processing_timestamp": "2025-09-19T10:35:00"
  }
}
```

### 2. Data Verification API

**Endpoint:** `POST /api/v1/verification/compare`

**Description:** Compare submitted form data with extracted data from original document

**Method 1: File Upload + Form Data**
```bash
curl -X POST \
  http://localhost:5000/api/v1/verification/compare \
  -F "file=@original_document.pdf" \
  -F 'submitted_data={"First Name": "John", "Last Name": "Doe", "Email ID": "john.doe@email.com"}'
```

**Method 2: JSON Payload**
```bash
curl -X POST \
  http://localhost:5000/api/v1/verification/compare \
  -H "Content-Type: application/json" \
  -d '{
    "submitted_data": {
      "First Name": "John",
      "Last Name": "Doe",
      "Email ID": "john.doe@email.com"
    },
    "extracted_data": {
      "First Name": "John",
      "Last Name": "Doe", 
      "Email ID": "john.doe@gmail.com"
    }
  }'
```

**Example using Python:**
```python
import requests
import json

# Method 1: With file upload
url = "http://localhost:5000/api/v1/verification/compare"
submitted_data = {
    "First Name": "John",
    "Last Name": "Doe",
    "Email ID": "john.doe@email.com"
}

with open("original_document.pdf", "rb") as f:
    files = {"file": f}
    data = {"submitted_data": json.dumps(submitted_data)}
    response = requests.post(url, files=files, data=data)
    result = response.json()

# Method 2: JSON only
payload = {
    "submitted_data": submitted_data,
    "extracted_data": {
        "First Name": "John",
        "Last Name": "Doe",
        "Email ID": "john.doe@gmail.com"  # Slight difference
    }
}
response = requests.post(url, json=payload)
result = response.json()
```

**Response:**
```json
{
  "status": "success",
  "verification_result": {
    "overall_status": "PARTIALLY_VERIFIED",
    "overall_match_percentage": 75.0,
    "fields_matched": 2,
    "total_fields": 3,
    "field_comparisons": {
      "First Name": {
        "submitted_value": "John",
        "extracted_value": "John",
        "extracted_field_key": "First Name",
        "match_status": true,
        "similarity_score": 1.0,
        "confidence_score": 1.0,
        "details": "Similarity: 1.00"
      },
      "Last Name": {
        "submitted_value": "Doe",
        "extracted_value": "Doe",
        "extracted_field_key": "Last Name", 
        "match_status": true,
        "similarity_score": 1.0,
        "confidence_score": 1.0,
        "details": "Similarity: 1.00"
      },
      "Email ID": {
        "submitted_value": "john.doe@email.com",
        "extracted_value": "john.doe@gmail.com",
        "extracted_field_key": "Email ID",
        "match_status": false,
        "similarity_score": 0.85,
        "confidence_score": 0.85,
        "details": "Similarity: 0.85 (Close match - possible OCR error)"
      }
    }
  },
  "metadata": {
    "total_extracted_fields": 8,
    "processing_timestamp": "2025-09-19T10:40:00"
  }
}
```

### Supported Fields

**Endpoint:** `GET /api/v1/fields/supported`

**Response:**
```json
{
  "status": "success",
  "supported_fields": [
    "First Name",
    "Middle Name", 
    "Last Name",
    "Gender",
    "Date of Birth",
    "Address Line 1",
    "Address Line 2",
    "City",
    "State",
    "Pin Code",
    "Phone Number",
    "Email ID"
  ],
  "field_count": 12
}
```

## Response Codes

- `200 OK`: Request successful
- `400 Bad Request`: Invalid input (missing file, invalid format, etc.)
- `413 Payload Too Large`: File size exceeds 16MB limit
- `500 Internal Server Error`: Processing failed

## Error Handling

All error responses follow this format:
```json
{
  "error": "Error description",
  "status": "error"
}
```

## Configuration

The API supports the following configuration:
- **Maximum file size**: 16MB
- **Supported formats**: PDF, PNG, JPG, JPEG, TIFF, TIF
- **OCR Languages**: English
- **Port**: 5000 (configurable)

## Technical Details

### OCR Technology Stack
- **Text Detection**: EasyOCR for region detection
- **Handwriting Recognition**: Microsoft TrOCR (Transformer-based OCR)
- **PDF Processing**: PyMuPDF (fitz)
- **Image Processing**: OpenCV + PIL

### Verification Algorithm
- **String Similarity**: Difflib SequenceMatcher
- **Field Matching**: Fuzzy name matching with 60% threshold
- **Confidence Scoring**: Multi-factor analysis including format validation
- **Match Thresholds**: 80% similarity for text fields, exact match for numeric fields

## Sample Test Files

The repository includes sample documents in `MOSIP-TextReading/Images/` for testing:
- ID cards with personal information
- Application forms
- Certificates

## Performance Notes

- **Processing time**: 2-5 seconds per page depending on complexity
- **Memory usage**: ~2GB RAM recommended for optimal performance
- **GPU support**: Automatically detects and uses CUDA if available
- **Concurrent requests**: Supports multiple simultaneous API calls

## Troubleshooting

### Common Issues

1. **"OCR processor not initialized"**
   - Ensure all dependencies are installed
   - Check if CUDA drivers are properly installed (for GPU acceleration)
   - Restart the server

2. **"File type not allowed"**
   - Check file extension is in supported list
   - Ensure file is not corrupted

3. **Poor OCR accuracy**
   - Ensure document image is high resolution (300+ DPI recommended)
   - Check if document is clearly visible and not blurry
   - Try preprocessing the image (contrast adjustment, noise reduction)

### Logs

The server provides detailed logging:
- OCR processing steps
- Field extraction results
- Error details with stack traces

## License

This project is part of the MOSIP (Modular Open Source Identity Platform) ecosystem.