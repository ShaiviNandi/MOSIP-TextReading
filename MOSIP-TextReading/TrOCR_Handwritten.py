from transformers import VisionEncoderDecoderModel, TrOCRProcessor
import torch
from PIL import Image
import os
import json
import re
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# --- SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model with different processor options
try:
    print("Loading TrOCR model...")
    # Try with fast processor first
    try:
        processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten', use_fast=True)
        print("Using fast processor")
    except:
        processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
        print("Using standard processor")
    
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten').to(device)
    print("Model loaded successfully")
except Exception as e:
    print(f"Failed to load model: {e}")
    exit(1)

def process_image(image_path):
    """Process image with multiple fallback methods"""
    try:
        print(f"\nProcessing: {os.path.basename(image_path)}")
        
        # Open image with explicit format handling
        with Image.open(image_path) as img:
            # Try three different processing approaches
            for attempt in range(1, 4):
                if attempt == 1:
                    # Method 1: Original approach
                    pixel_values = processor(images=img.convert("RGB"), return_tensors="pt").pixel_values.to(device)
                elif attempt == 2:
                    # Method 2: Try with different image format
                    pixel_values = processor(images=img.convert("L").convert("RGB"), return_tensors="pt").pixel_values.to(device)
                else:
                    # Method 3: Alternative processing
                    pixel_values = processor(images=img.convert("RGB").transpose(Image.TRANSPOSE), return_tensors="pt").pixel_values.to(device)
                
                generated_ids = model.generate(
                    pixel_values,
                    max_length=300,
                    num_beams=8,
                    early_stopping=True,
                    no_repeat_ngram_size=2,
                    temperature=0.7
                )
                
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                print(f"Attempt {attempt} Output: {generated_text}")
                
                # Validate output isn't just numbers/symbols
                if len(generated_text) > 3 and any(c.isalpha() for c in generated_text):
                    return extract_information(generated_text)
            
            print("All processing attempts failed")
            return None
            
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None

def extract_information(input_text):
    """Improved information extraction with more flexible patterns"""
    PATTERNS = {
        'first_name': r'(?:First\s*Name|Given\s*Name|First)[\s:]*([A-Za-z]+)',
        'middle_name': r'(?:Middle\s*Name|Middle)[\s:]*([A-Za-z]+)',
        'last_name': r'(?:Last\s*Name|Surname|Family\s*Name|Last)[\s:]*([A-Za-z]+)',
        'gender': r'(?:Gender|Sex)[\s:]+(Male|Female|M|F|Other)\b',
        'date_of_birth': r'(?:Date\s*of\s*Birth|DOB|D\.O\.B\.)[\s:]*([\d]{1,2}[-/][\d]{1,2}[-/][\d]{2,4})',
        'address_line_1': r'(?:Address|Addr\.?)[\s:]+(.*?)(?:\n|City|State|Pin|Zip)',
        'city': r'(?:City|Town)[\s:]+([A-Za-z\s]+)',
        'state': r'(?:State|Province)[\s:]+([A-Za-z\s]+)',
        'pin_code': r'(?:Pin|Zip|Postal)[\s:Code]*[\s:]*(\d{5,6})',
        'phone_number': r'(?:Phone|Mobile|Contact)[\s:No\.]*[\s:]*([\+\d\s-]{10,15})',
        'email_id': r'(?:Email|E-mail|Mail)[\s:ID]*[\s:]*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
    }

    extracted_data = {}
    for key, pattern in PATTERNS.items():
        try:
            matches = re.finditer(pattern, input_text, re.IGNORECASE)
            extracted_data[key] = " ".join([m.group(1).strip() for m in matches]) or None
        except Exception as e:
            print(f"Error extracting {key}: {e}")
            extracted_data[key] = None
    
    return extracted_data

def main():
    """Main function with comprehensive error handling"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(script_dir, "Images", "Handwriting")
    output_file = os.path.join(script_dir, 'output.json')
    
    os.makedirs(folder_path, exist_ok=True)
    
    try:
        image_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        if not image_files:
            print(f"No images found in {folder_path}")
            return
    except Exception as e:
        print(f"Error reading directory: {e}")
        return

    results = []
    for filename in image_files:
        image_path = os.path.join(folder_path, filename)
        result = process_image(image_path)
        if result:
            result['source_file'] = filename
            results.append(result)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    print("Starting OCR processing...")
    main()
    print("Processing completed")