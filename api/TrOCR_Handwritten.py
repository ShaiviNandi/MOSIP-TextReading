from transformers import VisionEncoderDecoderModel, TrOCRProcessor
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import numpy as np
import cv2
import re
import os
import json
from typing import Dict, List, Tuple, Optional
from transformers import pipeline
import easyocr

class GenericHandwritingOCR:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.load_models()
    
    def load_models(self):
        """Load text detection and handwritten OCR models"""
        try:
            print("Loading handwritten OCR model...")
            self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
            self.model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten').to(self.device)
            print("✅ TrOCR model loaded")
            
            print("Loading text detection model...")
            self.text_detector = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
            print("✅ Text detection model loaded")
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            self.model = None
            self.text_detector = None
    
    def detect_text_regions(self, image_path: str) -> List[Dict]:
        """Use Hugging Face model to detect text locations in the image"""
        if self.text_detector is None:
            return []
        
        try:
            # Read image
            image = cv2.imread(image_path)
            
            # Use EasyOCR to detect text regions
            results = self.text_detector.readtext(image_path, detail=1, paragraph=False)
            
            text_regions = []
            for i, (bbox, text, confidence) in enumerate(results):
                # Convert bbox to standard format (x1, y1, x2, y2)
                bbox = np.array(bbox)
                x1, y1 = bbox.min(axis=0).astype(int)
                x2, y2 = bbox.max(axis=0).astype(int)
                
                # Add padding for better OCR
                padding = 10
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(image.shape[1], x2 + padding)
                y2 = min(image.shape[0], y2 + padding)
                
                text_regions.append({
                    'id': f'text_region_{i}',
                    'bbox': (x1, y1, x2, y2),
                    'detected_text': text,
                    'confidence': confidence
                })
            
            print(f"🔍 Detected {len(text_regions)} text regions")
            return text_regions
            
        except Exception as e:
            print(f"❌ Text detection failed: {e}")
            return []
    
    def advanced_preprocessing(self, image: Image.Image) -> List[Image.Image]:
        """Apply multiple preprocessing techniques for better handwriting recognition"""
        if isinstance(image, str):
            image = Image.open(image)
        
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        variants = []
        img_array = np.array(image)
        
        # Variant 1: Enhanced CLAHE
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4,4))
        clahe_img = clahe.apply(img_array)
        clahe_enhanced = cv2.convertScaleAbs(clahe_img, alpha=1.3, beta=10)
        variants.append(Image.fromarray(clahe_enhanced).convert('RGB'))
        
        # Variant 2: Denoising + contrast
        denoised = cv2.fastNlMeansDenoising(img_array, h=10)
        enhanced = cv2.convertScaleAbs(denoised, alpha=1.8, beta=30)
        variants.append(Image.fromarray(enhanced).convert('RGB'))
        
        # Variant 3: Morphological preprocessing
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
        morph = cv2.morphologyEx(img_array, cv2.MORPH_CLOSE, kernel)
        sharp_kernel = np.array([[-1,-1,-1,-1,-1], 
                                [-1,2,2,2,-1], 
                                [-1,2,8,2,-1], 
                                [-1,2,2,2,-1], 
                                [-1,-1,-1,-1,-1]]) / 8.0
        sharpened = cv2.filter2D(morph, -1, sharp_kernel)
        variants.append(Image.fromarray(np.clip(sharpened, 0, 255).astype(np.uint8)).convert('RGB'))
        
        return variants
    
    def recognize_handwriting_with_trocr(self, region_image: Image.Image) -> str:
        """Use TrOCR to recognize handwritten text in the region"""
        if self.model is None:
            return ""
        
        variants = self.advanced_preprocessing(region_image)
        results = []
        
        for i, variant in enumerate(variants):
            try:
                pixel_values = self.processor(images=variant, return_tensors="pt").pixel_values.to(self.device)
                
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        pixel_values,
                        max_length=50,
                        num_beams=6,
                        early_stopping=True,
                        do_sample=False,
                        no_repeat_ngram_size=3,
                        length_penalty=1.0,
                        repetition_penalty=1.1
                    )
                
                text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                if text and len(text) > 0:
                    results.append((text, i))
                    
            except Exception as e:
                print(f"TrOCR attempt {i} failed: {e}")
                continue
        
        if not results:
            return ""
        
        # Return the best result
        best_result = max(results, key=lambda x: len(x[0]) if x[0] else 0)
        return best_result[0]
    
    def clean_text(self, text: str) -> str:
        """Enhanced text cleaning for values"""
        if not text or len(text.strip()) == 0:
            return ""
        
        # Basic cleaning
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single space
        
        # Remove trailing periods that are common OCR artifacts
        text = re.sub(r'\s*\.\s*$', '', text)
        
        # Clean up common OCR errors for specific patterns
        # Phone numbers
        if re.match(r'[\d\-\s]+', text):
            text = re.sub(r'[^\d\-]', '', text)
        
        # Email addresses
        if '@' in text:
            text = re.sub(r'[^\w@.-]', '', text)
        
        # Dates
        if re.match(r'[\d\-/\s]+', text):
            text = re.sub(r'[^\d\-/]', '', text)
        
        # General cleanup - keep alphanumeric, spaces, and common punctuation
        text = re.sub(r'[^\w\s@.-]', '', text)
        
        return text.strip()
    
    def create_debug_visualization(self, image_path: str, regions: List[Dict]):
        """Create debug image showing detected regions"""
        image = Image.open(image_path)
        debug_image = image.copy()
        draw = ImageDraw.Draw(debug_image)
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 
                 'pink', 'gray', 'olive', 'navy', 'cyan', 'magenta']
        
        for i, region in enumerate(regions):
            color = colors[i % len(colors)]
            x1, y1, x2, y2 = region['bbox']
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            region_id = region.get('id', f'region_{i}')
            draw.text((x1, y1-15), region_id, fill=color)
        
        debug_path = 'debug_detection.png'
        debug_image.save(debug_path)
        print(f"🔍 Debug image saved: {debug_path}")
    
    def pair_labels_with_values(self, regions: List[Dict]) -> Dict[str, str]:
        """Pair label regions with their corresponding value regions based on spatial proximity"""
        
        # Sort regions by position (top to bottom, left to right)
        sorted_regions = sorted(regions, key=lambda x: (x['position']['y'], x['position']['x']))
        
        key_value_pairs = {}
        used_regions = set()
        
        # First pass: identify potential labels and values
        labels = []
        values = []
        
        for i, region in enumerate(sorted_regions):
            text = region['text'].lower().strip()
            
            # Enhanced label detection
            label_keywords = ['name', 'first', 'middle', 'last', 'gender', 'date', 'birth', 
                            'address', 'line', 'city', 'state', 'phone', 'email', 'code', 'pin', 'plin']
            
            # Check if this region looks like a label
            is_label = any(keyword in text for keyword in label_keywords)
            
            # Additional checks for labels
            if not is_label:
                # Check if it's a short text that could be a label
                if len(text.split()) <= 3 and any(char.isalpha() for char in text):
                    # Check if it's positioned on the left side of the image
                    if region['position']['x'] < 200:  # Assuming labels are on the left
                        is_label = True
            
            if is_label:
                labels.append((i, region))
            else:
                values.append((i, region))
        
        print(f"🏷️  Found {len(labels)} potential labels and {len(values)} potential values")
        
        # Second pass: pair labels with values
        for label_idx, label_region in labels:
            if label_idx in used_regions:
                continue
            
            label_text = label_region['text'].lower().strip()
            label_center_x = label_region['position']['x'] + label_region['position']['width'] / 2
            label_center_y = label_region['position']['y'] + label_region['position']['height'] / 2
            label_right = label_region['position']['x'] + label_region['position']['width']
            
            best_value = None
            best_value_idx = -1
            min_distance = float('inf')
            
            for value_idx, value_region in values:
                if value_idx in used_regions:
                    continue
                
                value_text = value_region['text'].lower().strip()
                value_center_x = value_region['position']['x'] + value_region['position']['width'] / 2
                value_center_y = value_region['position']['y'] + value_region['position']['height'] / 2
                value_left = value_region['position']['x']
                
                # Skip if value also looks like a label
                if any(keyword in value_text for keyword in ['name', 'first', 'middle', 'last', 'date', 'birth', 'address', 'line', 'city', 'state', 'phone', 'email']):
                    continue
                
                # Calculate distances
                horizontal_distance = abs(value_center_x - label_center_x)
                vertical_distance = abs(value_center_y - label_center_y)
                
                # Prefer values that are:
                # 1. On the same row (small vertical distance)
                # 2. To the right of the label
                # 3. Close horizontally
                
                is_same_row = vertical_distance < 50  # Allow some tolerance for same row
                is_to_the_right = value_left > label_right - 50  # Value should be to the right
                
                if is_same_row and is_to_the_right:
                    # Same row pairing - prioritize horizontal distance
                    distance = horizontal_distance + vertical_distance * 0.1
                elif vertical_distance < 150:  # Different row but close vertically
                    # Check if they're roughly aligned (label above value)
                    horizontal_alignment = abs(label_center_x - value_center_x)
                    if horizontal_alignment < 100:  # Reasonably aligned
                        distance = vertical_distance + horizontal_alignment * 0.5
                    else:
                        continue  # Skip if not aligned
                else:
                    continue  # Too far apart
                
                if distance < min_distance:
                    min_distance = distance
                    best_value = value_region
                    best_value_idx = value_idx
            
            # Pair the label with the best value found
            if best_value and min_distance < 300:  # Reasonable maximum distance
                clean_label = self.format_label(label_region['text'])
                clean_value = self.clean_text(best_value['text'])
                
                if clean_label and clean_value:
                    key_value_pairs[clean_label] = clean_value
                    used_regions.add(label_idx)
                    used_regions.add(best_value_idx)
                    print(f"✅ Paired: '{clean_label}' -> '{clean_value}'")
        
        return key_value_pairs
    
    def format_label(self, text: str) -> str:
        """Format label text into proper field names"""
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        
        label_mappings = {
            'first name': 'First Name',
            'first': 'First Name',
            'middle name': 'Middle Name', 
            'middle': 'Middle Name',
            'midde': 'Middle Name',  # Common OCR error
            'manne': 'Middle Name',  # Common OCR error
            'last name': 'Last Name',
            'last': 'Last Name',
            'date of birth': 'Date of Birth',
            'birth': 'Date of Birth',
            'date': 'Date of Birth',
            'address line 1': 'Address Line 1',
            'line 1': 'Address Line 1',
            'address': 'Address Line 1',
            'address line 2': 'Address Line 2', 
            'line 2': 'Address Line 2',
            'city': 'City',
            'state': 'State',
            'pin code': 'Pin Code',
            'plin code': 'Pin Code',  # Common OCR error
            'phone number': 'Phone Number',
            'phone': 'Phone Number',
            'mumbers': 'Phone Number',  # Common OCR error
            'email id': 'Email ID',
            'email': 'Email ID',
            'gender': 'Gender'
        }
        
        # Try exact match first
        if text in label_mappings:
            return label_mappings[text]
        
        # Try partial matches for multi-word labels
        for key, value in label_mappings.items():
            if len(key.split()) > 1:  # Multi-word labels
                key_words = key.split()
                text_words = text.split()
                if len(set(key_words) & set(text_words)) >= len(key_words) - 1:  # Allow one word difference
                    return value
        
        # Try single word matches
        for key, value in label_mappings.items():
            if key in text or text in key:
                return value
        
        # Fallback: capitalize each word
        return ' '.join(word.capitalize() for word in text.split())

    def process_image(self, image_path: str) -> Dict:
        """Main processing function: detect text regions and recognize handwriting"""
        print(f"\n🔍 Processing: {os.path.basename(image_path)}")
        
        if self.model is None or self.text_detector is None:
            return {'error': 'Models not loaded'}
        
        # Step 1: Detect text regions using Hugging Face model
        text_regions = self.detect_text_regions(image_path)
        
        if not text_regions:
            return {'error': 'No text regions detected'}
        
        # Step 2: Process each detected region with TrOCR
        image = Image.open(image_path)
        processed_regions = []
        
        for region_info in text_regions:
            region_id = region_info['id']
            x1, y1, x2, y2 = region_info['bbox']
            
            print(f"Processing {region_id}...", end=" ")
            
            # Extract region from image
            region_image = image.crop((x1, y1, x2, y2))
            
            # Recognize handwriting with TrOCR
            recognized_text = self.recognize_handwriting_with_trocr(region_image)
            
            # Clean the text (basic cleaning only)
            if recognized_text:
                cleaned_text = self.clean_text(recognized_text)
                if cleaned_text:
                    processed_regions.append({
                        'id': region_id,
                        'text': cleaned_text,
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(region_info.get('confidence', 0.0)),
                        'position': {
                            'x': int(x1),
                            'y': int(y1),
                            'width': int(x2 - x1),
                            'height': int(y2 - y1)
                        }
                    })
                    print(f"'{cleaned_text}'")
                else:
                    print("❌ (empty after cleaning)")
            else:
                print("❌ (no text recognized)")
        
        # Create debug visualization
        self.create_debug_visualization(image_path, processed_regions)
        
        key_value_pairs = self.pair_labels_with_values(processed_regions)
        
        output_data = {
            'image_path': image_path,
            'total_regions': len(processed_regions),
            'form_fields': key_value_pairs,  # Changed from 'extracted_text' to 'form_fields'
            'raw_regions': {}  # Added raw regions for debugging
        }
        
        # Also include raw regions for debugging
        sorted_regions = sorted(processed_regions, key=lambda x: (x['position']['y'], x['position']['x']))
        
        for i, region in enumerate(sorted_regions):
            key = f"text_field_{i+1}"
            output_data['raw_regions'][key] = {
                'value': region['text'],
                'confidence': region['confidence'],
                'bbox': region['bbox'],
                'position': region['position']
            }
        
        return output_data

def main():
    """Main function to process handwritten text and output to JSON"""
    print("🎯 Generic Handwriting OCR with Text Detection + TrOCR")
    
    ocr = GenericHandwritingOCR()
    
    if ocr.model is None or ocr.text_detector is None:
        print("❌ Cannot proceed without models")
        return
    
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # Running in Jupyter notebook or interactive environment
        script_dir = os.getcwd()
    
    possible_paths = [
        os.path.join(script_dir, "Images", "Handwriting", "image.png"),
        os.path.join(script_dir, "images", "handwriting", "image.png"),
        os.path.join(script_dir, "image.png"),
        "Images/Handwriting/image.png",
        "images/handwriting/image.png",
        "image.png"
    ]
    
    image_path = None
    for path in possible_paths:
        if os.path.exists(path):
            image_path = path
            break
    
    if not image_path:
        print(f"❌ Image not found in any of these locations:")
        for path in possible_paths:
            print(f"   {path}")
        return
    
    print(f"📸 Found image: {image_path}")
    
    # Process the image
    results = ocr.process_image(image_path)
    
    if 'error' in results:
        print(f"❌ Error: {results['error']}")
        return
    
    output_file = 'output.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n📋 EXTRACTED FORM FIELDS:")
    print("=" * 50)
    
    if results.get('form_fields'):
        for field_name, field_value in results['form_fields'].items():
            print(f"{field_name}: {field_value}")
    else:
        print("No form fields detected")
    
    print("=" * 50)
    print(f"✅ Successfully processed {results['total_regions']} text regions")
    print(f"📁 Results saved to: {output_file}")

if __name__ == "__main__":
    main()
