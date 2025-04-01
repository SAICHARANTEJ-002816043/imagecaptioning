import os
import pandas as pd
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load and clean captions
def load_captions(file_path):
    captions = {}
    with open(file_path, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            image_id, caption = tokens[0].split('#')[0], ' '.join(tokens[1:])
            if image_id not in captions:
                captions[image_id] = []
            captions[image_id].append(caption)
    return captions

# Preprocess images (not used here but kept for consistency)
def preprocess_image(image_path):
    img = Image.open(image_path).resize((299, 299))  # For InceptionV3
    img = np.array(img) / 255.0  # Normalize
    return img

# Tokenize captions
def tokenize_captions(captions_dict, max_words=5000):
    all_captions = [cap for caps in captions_dict.values() for cap in caps]
    tokenizer = Tokenizer(num_words=max_words, oov_token='<UNK>')
    tokenizer.fit_on_texts(['<START> ' + cap + ' <END>' for cap in all_captions])
    return tokenizer

# Main preprocessing
def main():
    # Use absolute path based on script location
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Path to src/
    data_dir = os.path.join(script_dir, '..', 'data')        # Path to data/
    captions_file = os.path.join(data_dir, 'caption.txt')    # Path to caption.txt
    
    # Verify the file exists
    if not os.path.exists(captions_file):
        raise FileNotFoundError(f"Caption file not found at: {captions_file}. Please ensure 'caption.txt' is in 'data/'.")
    
    # Load and process captions
    captions_dict = load_captions(captions_file)
    tokenizer = tokenize_captions(captions_dict)
    
    # Save processed data
    processed_dir = os.path.join(data_dir, 'processed_data')
    os.makedirs(processed_dir, exist_ok=True)
    pd.to_pickle(captions_dict, os.path.join(processed_dir, 'captions_dict.pkl'))
    pd.to_pickle(tokenizer, os.path.join(processed_dir, 'tokenizer.pkl'))
    print("Preprocessing complete! Processed data saved in 'data/processed_data/'.")

if __name__ == "__main__":
    main()