from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import sys
import json
import tempfile
import warnings

# Suppress protobuf warnings
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf.runtime_version')

# Suppress TensorFlow warnings (optional)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info/warning messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations to avoid numerical differences
import shutil
from typing import Dict, List, Any, Optional
import cv2
import numpy as np
from PIL import Image
import fitz  # PyMuPDF for PDF processing
import difflib
from datetime import datetime
import re

# Add the parent directory to the path to import TrOCR_Handwritten
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'MOSIP-TextReading'))

try:
    from TrOCR_Handwritten import GenericHandwritingOCR
except ImportError as e:
    print(f"Error importing TrOCR_Handwritten: {e}")
    print("Please ensure the TrOCR_Handwritten.py file is in the correct location")
    sys.exit(1)

app = Flask(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'pdf', 'tiff', 'tif'}

# Initialize OCR processor
ocr_processor = None

def init_ocr():
    """Initialize the OCR processor"""
    global ocr_processor
    try:
        ocr_processor = GenericHandwritingOCR()
        print("✅ OCR processor initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize OCR processor: {e}")
        return False

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def pdf_to_images(pdf_path: str) -> List[str]:
    """Convert PDF to images and return list of image paths"""
    images = []
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            mat = fitz.Matrix(2.0, 2.0)  # Scale factor for better quality
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            # Save image to temporary file
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], f"page_{page_num}.png")
            with open(img_path, "wb") as f:
                f.write(img_data)
            images.append(img_path)
        doc.close()
        return images
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        return []

def calculate_field_confidence(extracted_value: str, expected_type: str = "") -> float:
    """Calculate confidence score for extracted field based on format validation"""
    if not extracted_value or len(extracted_value.strip()) == 0:
        return 0.0
    
    # Base confidence
    confidence = 0.5
    
    # Length-based confidence (reasonable field lengths)
    length = len(extracted_value.strip())
    if 2 <= length <= 50:
        confidence += 0.2
    elif length > 50:
        confidence -= 0.1
    
    # Pattern-based confidence for specific field types
    if expected_type:
        if expected_type.lower() in ['phone', 'phone number'] and re.match(r'^\d{10}$', extracted_value.strip()):
            confidence += 0.3
        elif expected_type.lower() in ['email', 'email id'] and '@' in extracted_value:
            confidence += 0.3
        elif expected_type.lower() in ['date', 'birth'] and re.match(r'\d{1,2}[-/]\d{1,2}[-/]\d{4}', extracted_value):
            confidence += 0.3
        elif expected_type.lower() == 'pin code' and re.match(r'^\d{6}$', extracted_value.strip()):
            confidence += 0.3
    
    return min(confidence, 1.0)

def normalize_field_name(field_name: str) -> str:
    """Normalize field names for comparison"""
    return re.sub(r'[^\w\s]', '', field_name.lower().strip())

def compare_values(extracted: str, submitted: str) -> Dict[str, Any]:
    """Compare extracted and submitted values and return match details"""
    if not extracted or not submitted:
        return {
            'match': False,
            'similarity': 0.0,
            'confidence': 0.0,
            'details': 'One or both values are empty'
        }
    
    # Normalize values for comparison
    extracted_norm = extracted.strip().lower()
    submitted_norm = submitted.strip().lower()
    
    # Calculate similarity using difflib
    similarity = difflib.SequenceMatcher(None, extracted_norm, submitted_norm).ratio()
    
    # Determine match based on similarity threshold
    match_threshold = 0.8
    is_match = similarity >= match_threshold
    
    # Calculate confidence based on similarity and other factors
    confidence = similarity
    
    # Special handling for specific data types
    if re.match(r'^\d+$', extracted.strip()) and re.match(r'^\d+$', submitted.strip()):
        # Numeric fields - exact match required
        is_match = extracted.strip() == submitted.strip()
        confidence = 1.0 if is_match else 0.0
    
    details = f"Similarity: {similarity:.2f}"
    if not is_match:
        if similarity > 0.5:
            details += " (Close match - possible OCR error)"
        else:
            details += " (Significant difference)"
    
    return {
        'match': is_match,
        'similarity': similarity,
        'confidence': confidence,
        'details': details
    }

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    global ocr_processor
    return jsonify({
        'status': 'healthy',
        'ocr_ready': ocr_processor is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/v1/ocr/extract', methods=['POST'])
def extract_ocr():
    """
    API 1: OCR Extraction API
    Accepts a scanned PDF/image and uses OCR to extract relevant fields
    """
    global ocr_processor
    
    if ocr_processor is None:
        return jsonify({
            'error': 'OCR processor not initialized',
            'status': 'error'
        }), 500
    
    # Check if file is present in request
    if 'file' not in request.files:
        return jsonify({
            'error': 'No file provided',
            'status': 'error'
        }), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({
            'error': 'No file selected',
            'status': 'error'
        }), 400
    
    if not allowed_file(file.filename):
        return jsonify({
            'error': f'File type not allowed. Supported: {", ".join(app.config["ALLOWED_EXTENSIONS"])}',
            'status': 'error'
        }), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Process based on file type
        if filename.lower().endswith('.pdf'):
            # Convert PDF to images
            image_paths = pdf_to_images(file_path)
            if not image_paths:
                return jsonify({
                    'error': 'Failed to convert PDF to images',
                    'status': 'error'
                }), 500
            
            # Process all pages and combine results
            all_extracted_fields = {}
            all_raw_regions = {}
            total_regions = 0
            
            for i, img_path in enumerate(image_paths):
                try:
                    result = ocr_processor.process_image(img_path)
                    if 'error' not in result:
                        # Add page prefix to avoid conflicts
                        page_prefix = f"page_{i+1}_"
                        
                        # Merge form fields
                        for field, value in result.get('form_fields', {}).items():
                            field_key = f"{page_prefix}{field}"
                            all_extracted_fields[field_key] = value
                        
                        # Merge raw regions
                        for region, data in result.get('raw_regions', {}).items():
                            region_key = f"{page_prefix}{region}"
                            all_raw_regions[region_key] = data
                        
                        total_regions += result.get('total_regions', 0)
                except Exception as e:
                    print(f"Error processing page {i+1}: {e}")
                    continue
            
            # Clean up temporary image files
            for img_path in image_paths:
                try:
                    os.remove(img_path)
                except:
                    pass
            
            extracted_data = {
                'form_fields': all_extracted_fields,
                'raw_regions': all_raw_regions,
                'total_regions': total_regions,
                'pages_processed': len(image_paths)
            }
        else:
            # Process single image
            result = ocr_processor.process_image(file_path)
            if 'error' in result:
                return jsonify({
                    'error': result['error'],
                    'status': 'error'
                }), 500
            
            extracted_data = result
        
        # Clean up uploaded file
        try:
            os.remove(file_path)
        except:
            pass
        
        # Calculate confidence scores for each field
        enriched_fields = {}
        for field_name, field_value in extracted_data.get('form_fields', {}).items():
            confidence = calculate_field_confidence(field_value, field_name)
            enriched_fields[field_name] = {
                'value': field_value,
                'confidence': confidence,
                'type': 'extracted'
            }
        
        response = {
            'status': 'success',
            'extracted_fields': enriched_fields,
            'metadata': {
                'total_regions_detected': extracted_data.get('total_regions', 0),
                'fields_extracted': len(enriched_fields),
                'processing_timestamp': datetime.now().isoformat()
            }
        }
        
        # Include raw regions if requested
        if request.args.get('include_raw') == 'true':
            response['raw_regions'] = extracted_data.get('raw_regions', {})
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'error': f'Processing failed: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/api/v1/verification/compare', methods=['POST'])
def verify_data():
    """
    API 2: Data Verification API
    Accepts form-filled data and original scanned document
    Compares extracted values against submitted data
    """
    global ocr_processor
    
    if ocr_processor is None:
        return jsonify({
            'error': 'OCR processor not initialized',
            'status': 'error'
        }), 500
    
    try:
        # Parse request data
        if request.content_type and 'multipart/form-data' in request.content_type:
            # File upload with form data
            if 'file' not in request.files:
                return jsonify({
                    'error': 'No document file provided',
                    'status': 'error'
                }), 400
            
            file = request.files['file']
            if not allowed_file(file.filename):
                return jsonify({
                    'error': f'File type not allowed. Supported: {", ".join(app.config["ALLOWED_EXTENSIONS"])}',
                    'status': 'error'
                }), 400
            
            # Get submitted form data
            submitted_data_str = request.form.get('submitted_data')
            if not submitted_data_str:
                return jsonify({
                    'error': 'No submitted_data provided',
                    'status': 'error'
                }), 400
            
            try:
                submitted_data = json.loads(submitted_data_str)
            except json.JSONDecodeError:
                return jsonify({
                    'error': 'Invalid JSON format in submitted_data',
                    'status': 'error'
                }), 400
            
            # Save and process uploaded file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Extract data from document
            if filename.lower().endswith('.pdf'):
                image_paths = pdf_to_images(file_path)
                if not image_paths:
                    return jsonify({
                        'error': 'Failed to convert PDF to images',
                        'status': 'error'
                    }), 500
                
                # Process all pages and combine results
                all_extracted_fields = {}
                for i, img_path in enumerate(image_paths):
                    try:
                        result = ocr_processor.process_image(img_path)
                        if 'error' not in result:
                            all_extracted_fields.update(result.get('form_fields', {}))
                    except Exception as e:
                        print(f"Error processing page {i+1}: {e}")
                        continue
                
                # Clean up temporary image files
                for img_path in image_paths:
                    try:
                        os.remove(img_path)
                    except:
                        pass
            else:
                result = ocr_processor.process_image(file_path)
                if 'error' in result:
                    return jsonify({
                        'error': result['error'],
                        'status': 'error'
                    }), 500
                all_extracted_fields = result.get('form_fields', {})
            
            # Clean up uploaded file
            try:
                os.remove(file_path)
            except:
                pass
            
        else:
            # JSON request
            data = request.get_json()
            if not data:
                return jsonify({
                    'error': 'No JSON data provided',
                    'status': 'error'
                }), 400
            
            submitted_data = data.get('submitted_data', {})
            all_extracted_fields = data.get('extracted_data', {})
            
            if not submitted_data:
                return jsonify({
                    'error': 'No submitted_data provided',
                    'status': 'error'
                }), 400
        
        # Perform field-by-field comparison
        comparison_results = {}
        overall_match_count = 0
        total_fields = 0
        
        # Normalize field names for better matching
        normalized_extracted = {}
        for field, value in all_extracted_fields.items():
            normalized_key = normalize_field_name(field)
            normalized_extracted[normalized_key] = {'original_key': field, 'value': value}
        
        for submitted_field, submitted_value in submitted_data.items():
            total_fields += 1
            normalized_submitted_field = normalize_field_name(submitted_field)
            
            # Try to find matching extracted field
            extracted_value = None
            original_extracted_key = None
            
            # Direct match
            if normalized_submitted_field in normalized_extracted:
                extracted_info = normalized_extracted[normalized_submitted_field]
                extracted_value = extracted_info['value']
                original_extracted_key = extracted_info['original_key']
            else:
                # Fuzzy match for field names
                best_match = None
                best_similarity = 0
                for norm_key, info in normalized_extracted.items():
                    similarity = difflib.SequenceMatcher(None, normalized_submitted_field, norm_key).ratio()
                    if similarity > best_similarity and similarity > 0.6:  # 60% similarity threshold
                        best_similarity = similarity
                        best_match = info
                
                if best_match:
                    extracted_value = best_match['value']
                    original_extracted_key = best_match['original_key']
            
            # Compare values
            if extracted_value is not None:
                comparison = compare_values(extracted_value, str(submitted_value))
                if comparison['match']:
                    overall_match_count += 1
                
                comparison_results[submitted_field] = {
                    'submitted_value': submitted_value,
                    'extracted_value': extracted_value,
                    'extracted_field_key': original_extracted_key,
                    'match_status': comparison['match'],
                    'similarity_score': comparison['similarity'],
                    'confidence_score': comparison['confidence'],
                    'details': comparison['details']
                }
            else:
                comparison_results[submitted_field] = {
                    'submitted_value': submitted_value,
                    'extracted_value': None,
                    'extracted_field_key': None,
                    'match_status': False,
                    'similarity_score': 0.0,
                    'confidence_score': 0.0,
                    'details': 'Field not found in extracted data'
                }
        
        # Calculate overall verification score
        overall_match_percentage = (overall_match_count / total_fields * 100) if total_fields > 0 else 0
        
        # Determine verification status
        if overall_match_percentage >= 90:
            verification_status = 'VERIFIED'
        elif overall_match_percentage >= 70:
            verification_status = 'PARTIALLY_VERIFIED'
        else:
            verification_status = 'VERIFICATION_FAILED'
        
        response = {
            'status': 'success',
            'verification_result': {
                'overall_status': verification_status,
                'overall_match_percentage': round(overall_match_percentage, 2),
                'fields_matched': overall_match_count,
                'total_fields': total_fields,
                'field_comparisons': comparison_results
            },
            'metadata': {
                'total_extracted_fields': len(all_extracted_fields),
                'processing_timestamp': datetime.now().isoformat()
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'error': f'Verification failed: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/api/v1/fields/supported', methods=['GET'])
def get_supported_fields():
    """Get list of supported form fields that can be extracted"""
    supported_fields = [
        'First Name',
        'Middle Name', 
        'Last Name',
        'Gender',
        'Date of Birth',
        'Address Line 1',
        'Address Line 2',
        'City',
        'State',
        'Pin Code',
        'Phone Number',
        'Email ID'
    ]
    
    return jsonify({
        'status': 'success',
        'supported_fields': supported_fields,
        'field_count': len(supported_fields)
    })

@app.errorhandler(413)
def file_too_large(error):
    return jsonify({
        'error': 'File too large. Maximum size allowed is 16MB.',
        'status': 'error'
    }), 413

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'status': 'error',
        'available_endpoints': [
            'GET /health',
            'POST /api/v1/ocr/extract',
            'POST /api/v1/verification/compare',
            'GET /api/v1/fields/supported'
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'status': 'error'
    }), 500

if __name__ == '__main__':
    print("🚀 Starting MOSIP Text Reading API Server...")
    
    # Initialize OCR processor
    if not init_ocr():
        print("❌ Failed to initialize OCR processor. Exiting...")
        sys.exit(1)
    
    print("✅ API Server ready!")
    print("\nAvailable endpoints:")
    print("📊 GET  /health - Health check")
    print("🔍 POST /api/v1/ocr/extract - OCR extraction from PDF/image")
    print("✅ POST /api/v1/verification/compare - Data verification")
    print("📋 GET  /api/v1/fields/supported - Supported field types")
    print("\n🌐 Server starting on http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)