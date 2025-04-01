import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.preprocess import load_captions, preprocess_image, tokenize_captions
from src.model import build_model

def prepare_data(captions_dict, tokenizer, image_dir, max_len=40):
    X_img, X_cap, y = [], [], []
    total_images = len(captions_dict)
    processed_images = 0
    print(f"Starting data preparation for {total_images} images...")
    for img_id, captions in captions_dict.items():
        img_path = f"{image_dir}/{img_id}"
        try:
            img = preprocess_image(img_path)
            for cap in captions:
                seq = tokenizer.texts_to_sequences(['<START> ' + cap + ' <END>'])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_len)[0]
                    out_seq = tf.keras.utils.to_categorical([out_seq], num_classes=len(tokenizer.word_index) + 1)[0]
                    X_img.append(img)
                    X_cap.append(in_seq)
                    y.append(out_seq)
            processed_images += 1
            print(f"Processed {processed_images}/{total_images} images", end='\r')
        except FileNotFoundError:
            print(f"Image not found: {img_path}. Skipping.")
            continue
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    print("\nConverting lists to NumPy arrays...")
    try:
        X_img = np.array(X_img)
        X_cap = np.array(X_cap)
        y = np.array(y)
        print(f"Data shapes - X_img: {X_img.shape}, X_cap: {X_cap.shape}, y: {y.shape}")
    except Exception as e:
        print(f"Error converting to NumPy arrays: {e}")
        raise
    print("Data preparation complete!")
    return X_img, X_cap, y

def main():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, '..', 'data')
        print("Loading preprocessed data...")
        captions_dict = pd.read_pickle(os.path.join(data_dir, 'processed_data', 'captions_dict.pkl'))
        tokenizer = pd.read_pickle(os.path.join(data_dir, 'processed_data', 'tokenizer.pkl'))
        print(f"Loaded {len(captions_dict)} image-caption pairs")
        
        captions_dict = {k: v for k, v in list(captions_dict.items())[:100]}
        print(f"Reduced to {len(captions_dict)} image-caption pairs for training")

        image_dir = os.path.join(data_dir, 'image')
        X_img, X_cap, y = prepare_data(captions_dict, tokenizer, image_dir)
        
        print("Building model...")
        model = build_model(vocab_size=len(tokenizer.word_index) + 1, max_len=40)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        print("Starting training...")
        model.fit([X_img, X_cap], y, epochs=10, batch_size=32, validation_split=0.2)
        
        # Ensure the save directory exists
        save_dir = os.path.join(script_dir, '..', 'models', 'saved_model')
        if os.path.exists(save_dir) and not os.path.isdir(save_dir):
            print(f"Removing existing file at {save_dir}...")
            os.remove(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        model.save(os.path.join(save_dir, 'my_caption_model.h5'))
        print("Training complete! Model saved in 'models/saved_model/'.")
    except Exception as e:
        print(f"Main function failed: {e}")
        raise

if __name__ == "__main__":
    main()