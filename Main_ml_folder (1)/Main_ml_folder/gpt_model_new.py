import os
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from docx import Document
from PIL import Image
import io

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Function to preprocess text
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalnum()]
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

# Function to extract keywords from questions
def extract_keywords(question):
    patterns = [
        r'what\s+is\s+([\w\s]+)\b',
        r'what\s+are\s+([\w\s]+)\b',
        r'explain\s+([\w\s]+)\b',
        r'what\s+does\s+([\w\s]+)\s+mean\b',
        r'define\s+([\w\s]+)\b',
        r'can\s+you\s+tell\s+me\s+about\s+([\w\s]+)\b',
        r'how\s+does\s+([\w\s]+)\s+work\b',
        r'describe\s+([\w\s]+)\b',
        r'difference\s+between\s+([\w\s]+)\s+and\s+([\w\s]+)\b',
        r'([\w\s]+)\b\?'
    ]

    keywords = []
    for pattern in patterns:
        match = re.search(pattern, question, re.IGNORECASE)
        if match:
            if len(match.groups()) == 1:
                keywords.append(match.group(1).strip())
            elif len(match.groups()) > 1:
                keywords.extend([match.group(i).strip() for i in range(1, len(match.groups()) + 1)])
            break

    if not keywords:
        keywords = [token for token in word_tokenize(question.lower()) if token.isalnum() and token not in stopwords.words('english')]

    return keywords

# Read all text files in all subfolders and concatenate their contents
import fitz  # PyMuPDF

def read_all_text_files(main_folder_path):
    all_text = ""
    for root, dirs, files in os.walk(main_folder_path):
        for filename in files:
            if filename.endswith('.txt'):
                file_path = os.path.join(root, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    all_text += file.read() + "\n"
            elif filename.endswith('.pdf'):
                file_path = os.path.join(root, filename)
                pdf_document = fitz.open(file_path)
                for page_num in range(pdf_document.page_count):
                    page = pdf_document[page_num]
                    page_text = page.get_text()
                    all_text += page_text + "\n"
    return all_text


# Function to get relevant information based on keywords
def get_relevant_info(keywords, data):
    relevant_info = {}

    sections = re.split(r'\n\s*\n', data.strip())
    for section in sections:
        for keyword in keywords:
            if keyword.lower() not in relevant_info and section.lower().startswith(keyword.lower() + ":"):
                relevant_info[keyword.lower()] = section

    return relevant_info



# Function to extract images from a DOCX file
def extract_images_from_docx(docx_file_path):
    doc = Document(docx_file_path)
    images = {}
    current_heading = "uncategorized"

    for paragraph in doc.paragraphs:
        if paragraph.style.name.startswith('Heading') or paragraph.style.name == 'Normal':
            current_heading = paragraph.text.strip().lower()  # Convert heading to lowercase
            if current_heading not in images:
                images[current_heading] = []
            print(f"Detected heading or normal text: {current_heading}")

        for run in paragraph.runs:
            inline_shapes = run.element.findall('.//a:blip', namespaces={'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'})
            for inline_shape in inline_shapes:
                image_id = inline_shape.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed']
                image_part = doc.part.related_parts[image_id]
                image_bytes = image_part.blob
                images[current_heading].append(image_bytes)
                print(f"Associated image with heading or normal text: {current_heading}")

    return images

# Function to save images to disk and return their paths
def save_images_to_disk(images):
    image_paths = {}
    for heading, image_data_list in images.items():
        image_paths[heading] = []
        for idx, image_data in enumerate(image_data_list):
            safe_heading = re.sub(r'[\\/*?:"<>|]', "", heading)  # Remove invalid characters from heading
            image_path = f"{safe_heading.replace(' ', '_')}_{idx + 1}.png"
            with open(image_path, 'wb') as f:
                image = Image.open(io.BytesIO(image_data))  # Open image using PIL
                image.save(f, format='PNG')  # Specify PNG format explicitly
            image_paths[heading].append(image_path)
            print(f"Saved image {image_path}")
    return image_paths

# Load GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Function to generate a response using GPT-2
# Function to generate a response using GPT-2
def generate_response(prompt):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    max_length = min(len(inputs[0]) + 200, 1024)
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the keyword from the response
    response = re.sub(r'^.*?":', '', response)
    response = response.replace('{', '').replace('}', '').replace('"', '').replace('\\n', '<br>')
    return response.strip()


# Function to display images using PIL
def display_image_pil(image_path):
    img = Image.open(image_path)
    img.show()

# Main function
def main():
    main_folder_path = r"E:\Amity University\Main_ml_folder"
    docx_file_path = r"E:\Amity University\Main_ml_folder\diagrams\diag.docx"

    data = read_all_text_files(main_folder_path)
    images = extract_images_from_docx(docx_file_path)
    image_paths = save_images_to_disk(images)

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        # Extract keywords from the user input
        keywords = extract_keywords(user_input)
        print(f"Extracted keywords: {keywords}")

        # Normalize keywords to lowercase
        keywords = [keyword.lower() for keyword in keywords]

        # Attempt to find relevant information in the text data
        relevant_info = get_relevant_info(keywords, data)

        if relevant_info:
            for keyword, info in relevant_info.items():
                print("Bot:")
                print(f"Information about {keyword.capitalize()}:\n{info}\n")
                if keyword in image_paths and image_paths[keyword]:
                    image_path = image_paths[keyword][0]
                    print(f"Image related to {keyword.capitalize()}: {image_path}")
                    display_image_pil(image_path)
        else:
            found_image = False
            for keyword in keywords:
                if keyword in image_paths and image_paths[keyword]:
                    image_path = image_paths[keyword][0]
                    print(f"Image related to {keyword.capitalize()}: {image_path}")
                    display_image_pil(image_path)
                    found_image = True
                    break

            if not found_image:
                response = generate_response(
                    f"The user asked: {user_input}\nBot: Sorry, I couldn't find information related to your query.")
                print("Bot:", response)

        print(
            "-------------------------------------------------------------------------------------------------------------------------------------------")

if __name__ == "__main__":
    main()
