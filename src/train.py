import tensorflow as tf
import numpy as np
import pandas as pd
from src.preprocess import load_captions, preprocess_image, tokenize_captions
from src.model import build_model

def prepare_data(captions_dict, tokenizer, image_dir, max_len=40):
    X_img, X_cap, y = [], [], []
    for img_id, captions in captions_dict.items():
        img_path = f"{image_dir}/{img_id}"
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
    return np.array(X_img), np.array(X_cap), np.array(y)

def main():
    captions_dict = pd.read_pickle('../data/processed_data/captions_dict.pkl')
    tokenizer = pd.read_pickle('../data/processed_data/tokenizer.pkl')
    X_img, X_cap, y = prepare_data(captions_dict, tokenizer, '../data/images/')
    
    model = build_model(vocab_size=len(tokenizer.word_index) + 1, max_len=40)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    model.fit([X_img, X_cap], y, epochs=10, batch_size=32, validation_split=0.2)
    model.save('../models/saved_model/my_caption_model.h5')
    print("Training complete!")

if __name__ == "__main__":
    main()