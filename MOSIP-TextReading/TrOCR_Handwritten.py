from transformers import VisionEncoderDecoderModel, TrOCRProcessor
import torch
from PIL import Image
import os
import json
import re

# Load the processor and model from Hugging Face
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')

def extract_information(input_text):
    """
    Extracts personal information from a string using regular expressions.

    Args:
        input_text (str): A string containing the information to be parsed.

    Returns:
        dict: A dictionary containing the extracted information.
    """
    # Define patterns for each piece of information.
    # The '(...)' part is a "capturing group" that grabs the actual value.
    # re.IGNORECASE makes the search case-insensitive.
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
        # Search for the pattern in the input text
        match = re.search(pattern, input_text, re.IGNORECASE)
        
        # If a match is found, extract the captured group and clean it up
        if match:
            extracted_data[key] = match.group(1).strip()
        else:
            extracted_data[key] = None 

    return extracted_data

#iterating through every image in the handwriting folder
folder_path = "..\Images\Handwriting"
for filename in os.listdir(folder_path):
    image_path = os.path.join(folder_path, filename)
    try:
        image = Image.open(image_path).convert("RGB")
        print(f"Image loaded from: {image_path}")
    except FileNotFoundError:
        print(f"ERROR: The file '{image_path}' was not found.")
        exit()


    # 1. Process the image: This converts the PIL image into a numerical tensor.
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    print("Image processed into a tensor.")

    # 2. Generate text IDs: The model "looks" at the tensor and generates a sequence
    #    of numbers (IDs), where each number represents a character token.
    generated_ids = model.generate(pixel_values)
    print("Model generated output token IDs.")

    # 3. Decode the IDs: The processor converts the numerical IDs back into a
    #    human-readable text string.
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print("Decoded text from token IDs.")

    # find key value pairs for first name, middle name, last name, gender, date of birth, address line 1, address line 2, city, state, pin code, phone number and email id from generated text and place them in the dictionary
    generated_dict = extract_information(generated_text)

    # Creating a JSON file to store the generated text
    with open('output.json', 'a') as f:
        json.dump(generated_dict, f)

    # --- Final Output ---
    print("\n" + "="*50)
    print(f"TRANSCRIBED TEXT: {generated_text}")
    print("="*50)


