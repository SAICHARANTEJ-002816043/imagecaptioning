import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow as tf
import numpy as np
import pandas as pd
from src.preprocess import preprocess_image

def load_model_and_tokenizer(model_path, tokenizer_path):
    model = tf.keras.models.load_model(model_path)
    tokenizer = pd.read_pickle(tokenizer_path)
    return model, tokenizer

def generate_caption(model, tokenizer, image_path, max_len=40):
    img = preprocess_image(image_path)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    caption = ['<START>']
    for _ in range(max_len):
        seq = tokenizer.texts_to_sequences([caption])[0]
        seq = tf.keras.preprocessing.sequence.pad_sequences([seq], maxlen=max_len, padding='post')
        pred = model.predict([img, seq], verbose=0)
        next_word_idx = np.argmax(pred[0])
        next_word = tokenizer.index_word.get(next_word_idx, '<UNK>')
        if next_word == '<END>':
            break
        caption.append(next_word)
    
    return ' '.join(caption[1:])  # Skip <START>

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    model_path = os.path.join(script_dir, '..', 'models', 'saved_model', 'my_caption_model.h5')
    tokenizer_path = os.path.join(data_dir, 'processed_data', 'tokenizer.pkl')
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(model_path, tokenizer_path)
    
    # Example image (first one from dataset)
    image_dir = os.path.join(data_dir, 'image')
    sample_image = os.path.join(image_dir, '1000268201_693b08cb0e.jpg')  # From Flickr8k
    
    # Generate caption
    print(f"Generating caption for {sample_image}...")
    caption = generate_caption(model, tokenizer, sample_image)
    print(f"Generated caption: {caption}")

if __name__ == "__main__":
    main()