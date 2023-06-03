import os
import json
import PyPDF2
import pdfplumber
import openai
import spacy

# OpenAI API key
openai.api_key = os.getenv('OPENAI_KEY')

def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

nlp = spacy.load("en_core_web_sm")

def count_tokens(text):
    doc = nlp(text)
    token_count = len(doc)
    return token_count

def truncate_text(text, max_tokens):
    words = text.split()
    if len(words) > max_tokens:
        words = words[:max_tokens]
    return ' '.join(words)

def process_text_with_gpt3(text):
    instruction = """I want you to act as a data annotator. 
    Put the extracted text from the energy invoice into the desired format. 
    The desired format:
    {
        "vendor_name": "input}",    
        "invoice_date": "input",
        "invoice_number": "input",
        "total_amount": "input",
        "charge_period_start_date": "input",
        "charge_period_end_date": "input",
        "mpan": "input",
        "account_number": "input"
    }
    The extracted text:  
    """
    # Set a safe limit for the output
    max_tokens_output = 1000

    # Calculate the remaining tokens available for the input
    max_tokens_input = 4096 - max_tokens_output

    text_token_count = count_tokens(text)
    instruction_token_count = count_tokens(instruction)

    # If the total tokens in the input exceed the maximum allowed,
    # then truncate the text
    if text_token_count + instruction_token_count > max_tokens_input:
        excess = text_token_count + instruction_token_count - max_tokens_input
        text = truncate_text(text, text_token_count - excess)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": text},
        ],
        max_tokens=max_tokens_output  # Use the safe limit for the output
    )
    return response.choices[0].message['content']

def write_to_json(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f)

def parse_processed_text_to_json(text):
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        print(f"Could not parse text: {text}")
        return None

    required_keys = {
        "vendor_name", "invoice_date", "invoice_number", "total_amount",
        "charge_period_start_date", "charge_period_end_date", "mpan", "account_number"
    }
    if not required_keys.issubset(data.keys()):
        print(f"Missing keys in data: {data}")
        return None

    return data


def main():
    data_folder = './train_data'
    for filename in os.listdir(data_folder):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(data_folder, filename)
            extracted_text = extract_text_from_pdf(pdf_path)
            processed_text = process_text_with_gpt3(extracted_text)

            parsed_data = parse_processed_text_to_json(processed_text)

            json_path = os.path.splitext(pdf_path)[0] + '.json'
            write_to_json(json_path, parsed_data)

if __name__ == '__main__':
    main()

