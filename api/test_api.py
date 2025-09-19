#!/usr/bin/env python3
"""
Test script for MOSIP Text Reading API
Tests both OCR extraction and data verification endpoints
"""

import requests
import json
import os
import sys
from typing import Dict, Any

# API base URL
BASE_URL = "http://localhost:5000"

def test_health_check():
    """Test the health check endpoint"""
    print("🏥 Testing health check endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
            return data.get('ocr_ready', False)
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_supported_fields():
    """Test the supported fields endpoint"""
    print("\n📋 Testing supported fields endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/v1/fields/supported")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Supported fields: {data['field_count']} fields")
            print(f"Fields: {', '.join(data['supported_fields'][:5])}...")
            return True
        else:
            print(f"❌ Supported fields failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Supported fields error: {e}")
        return False

def test_ocr_extraction(image_path: str = None):
    """Test OCR extraction endpoint"""
    print("\n🔍 Testing OCR extraction endpoint...")
    
    # Try to find a test image
    if not image_path:
        possible_paths = [
            "../MOSIP-TextReading/Images/Handwriting/image.png",
            "../MOSIP-TextReading/images/handwriting/image.png", 
            "test_image.png",
            "sample.pdf"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                image_path = path
                break
    
    if not image_path or not os.path.exists(image_path):
        print("⚠️  No test image found. Skipping OCR extraction test.")
        print("   Please provide a test image or PDF file.")
        return None
    
    print(f"📄 Using test file: {image_path}")
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{BASE_URL}/api/v1/ocr/extract", files=files)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ OCR extraction successful!")
            print(f"   Fields extracted: {data['metadata']['fields_extracted']}")
            print(f"   Total regions: {data['metadata']['total_regions_detected']}")
            
            # Show first few extracted fields
            fields = data.get('extracted_fields', {})
            if fields:
                print("   Sample fields:")
                for i, (field, info) in enumerate(list(fields.items())[:3]):
                    print(f"     {field}: '{info['value']}' (confidence: {info['confidence']:.2f})")
            
            return data
        else:
            print(f"❌ OCR extraction failed: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error: {error_data.get('error', 'Unknown error')}")
            except:
                print(f"   Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ OCR extraction error: {e}")
        return None

def test_data_verification(extracted_data: Dict[str, Any] = None):
    """Test data verification endpoint"""
    print("\n✅ Testing data verification endpoint...")
    
    # Sample submitted data for testing
    sample_submitted_data = {
        "First Name": "John",
        "Last Name": "Doe", 
        "Email ID": "john.doe@email.com",
        "Phone Number": "9876543210"
    }
    
    # If we have extracted data from OCR test, use it
    if extracted_data and 'extracted_fields' in extracted_data:
        # Convert extracted fields format for verification
        extracted_fields = {}
        for field, info in extracted_data['extracted_fields'].items():
            extracted_fields[field] = info['value']
        
        payload = {
            "submitted_data": sample_submitted_data,
            "extracted_data": extracted_fields
        }
    else:
        # Use mock extracted data for testing
        mock_extracted_data = {
            "First Name": "John",
            "Last Name": "Doe",
            "Email ID": "john.doe@gmail.com",  # Slight difference 
            "Phone Number": "9876543210"
        }
        
        payload = {
            "submitted_data": sample_submitted_data,
            "extracted_data": mock_extracted_data
        }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/verification/compare",
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            data = response.json()
            result = data['verification_result']
            print(f"✅ Data verification successful!")
            print(f"   Overall status: {result['overall_status']}")
            print(f"   Match percentage: {result['overall_match_percentage']}%")
            print(f"   Fields matched: {result['fields_matched']}/{result['total_fields']}")
            
            # Show comparison details
            print("   Field comparisons:")
            for field, comparison in result['field_comparisons'].items():
                status = "✓" if comparison['match_status'] else "✗"
                print(f"     {status} {field}: {comparison['similarity_score']:.2f}")
            
            return data
        else:
            print(f"❌ Data verification failed: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error: {error_data.get('error', 'Unknown error')}")
            except:
                print(f"   Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Data verification error: {e}")
        return None

def test_file_upload_verification(image_path: str = None):
    """Test data verification with file upload"""
    print("\n📤 Testing data verification with file upload...")
    
    if not image_path:
        possible_paths = [
            "../MOSIP-TextReading/Images/Handwriting/image.png",
            "../MOSIP-TextReading/images/handwriting/image.png"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                image_path = path
                break
    
    if not image_path or not os.path.exists(image_path):
        print("⚠️  No test image found. Skipping file upload verification test.")
        return None
    
    sample_submitted_data = {
        "First Name": "Abigail",
        "Last Name": "Smith", 
        "Email ID": "abigail@email.com"
    }
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            data = {'submitted_data': json.dumps(sample_submitted_data)}
            response = requests.post(
                f"{BASE_URL}/api/v1/verification/compare",
                files=files,
                data=data
            )
        
        if response.status_code == 200:
            data = response.json()
            result = data['verification_result']
            print(f"✅ File upload verification successful!")
            print(f"   Overall status: {result['overall_status']}")
            print(f"   Match percentage: {result['overall_match_percentage']}%")
            return data
        else:
            print(f"❌ File upload verification failed: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ File upload verification error: {e}")
        return None

def main():
    """Run all API tests"""
    print("🧪 MOSIP Text Reading API Test Suite")
    print("=" * 50)
    
    # Test 1: Health check
    if not test_health_check():
        print("❌ Server is not healthy. Please start the API server first.")
        print("   Run: python api.py")
        sys.exit(1)
    
    # Test 2: Supported fields
    test_supported_fields()
    
    # Test 3: OCR extraction
    extracted_data = test_ocr_extraction()
    
    # Test 4: Data verification (JSON)
    test_data_verification(extracted_data)
    
    # Test 5: Data verification (file upload)
    test_file_upload_verification()
    
    print("\n" + "=" * 50)
    print("🎉 API test suite completed!")
    print("\n📖 For detailed API documentation, see README.md")

if __name__ == "__main__":
    main()