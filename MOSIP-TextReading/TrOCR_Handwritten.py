from transformers import VisionEncoderDecoderModel, TrOCRProcessor
import torch
from PIL import Image
import os
import json
import re

# --- 1. SETUP: Load model and determine device (GPU or CPU) ---

# Check if a GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the processor and model from Hugging Face
# Move the model to the selected device (GPU/CPU) for faster processing
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten').to(device)

def extract_information(input_text):
    """
    Extracts personal information from a string using regular expressions.
    This function remains the same as your original, as it's well-structured.
    """
    PATTERNS = {
        'first_name': r'(?:First Name|Given Name|First)[\s:]+([A-Za-z]+)',
        'middle_name': r'(?:Middle Name|Middle)[\s:]+([A-Za-z]+)',
        'last_name': r'(?:Last Name|Surname|Family Name|Last)[\s:]+([A-Za-z]+)',
        'gender': r'(?:Gender|Sex)[\s:]+(Male|Female|M|F|Other)\b',
        'date_of_birth': r'(?:Date of Birth|DOB)[\s:]+([\d]{1,2}[-/][\d]{1,2}[-/][\d]{2,4})',
        'address_line_1': r'Address[\s:]+(.*?)(?:\n|City|State|Pin)',
        'city': r'City[\s:]+([A-Za-z\s]+)',
        'state': r'State[\s:]+([A-Za-z\s]+)',
        'pin_code': r'(?:Pin|Zip|Postal)[\s:Code]*(\d{6})',
        'phone_number': r'(?:Phone|Mobile|Contact)[\s:No]*([\+\d\s-]{10,15})',
        'email_id': r'Email[\s:ID]*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
    }

    extracted_data = {}
    for key, pattern in PATTERNS.items():
        match = re.search(pattern, input_text, re.IGNORECASE)
        if match:
            extracted_data[key] = match.group(1).strip()
        else:
            extracted_data[key] = None
    return extracted_data

def process_image(image_path):
    """
    Processes a single image: loads it, performs OCR, and extracts information.
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"ERROR: Could not read or process file '{image_path}'. Reason: {e}")
        return None

    # Process the image tensor and move it to the same device as the model
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    
    # Generate text from the image
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Extract structured data from the generated text
    extracted_data = extract_information(generated_text)
    
    # Add the source filename and the raw text for reference
    extracted_data['source_file'] = os.path.basename(image_path)
    extracted_data['raw_text'] = generated_text
    
    print(f"Successfully processed: {os.path.basename(image_path)}")
    print(f"-> Raw Text: {generated_text[:70]}...") # Print a snippet
    return extracted_data

def main():
    """
    Main function to orchestrate the processing of all images in a folder.
    """
    # Use os.path.join for better cross-platform compatibility (Windows, macOS, Linux)
    folder_path = os.path.join("..", "Images", "Handwriting")
    output_file = 'output.json'
    
    # --- Collect all results in a list first ---
    all_results = []

    if not os.path.isdir(folder_path):
        print(f"Error: Folder not found at '{folder_path}'")
        return

    for filename in os.listdir(folder_path):
        # Process only common image file types
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_path = os.path.join(folder_path, filename)
            result = process_image(image_path)
            if result:
                all_results.append(result)

    # --- Write the entire list to a valid JSON file at once ---
    # This creates a properly formatted JSON file (a list of objects).
    # Using 'w' (write mode) ensures you get a fresh file each time.
    # indent=4 makes the JSON file human-readable.
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=4)
        
    print("\n" + "="*50)
    print(f"Processing complete. {len(all_results)} images processed.")
    print(f"Results saved to '{output_file}'")
    print("="*50)

# Standard Python practice to run the main function
if __name__ == "__main__":
    main()