import tensorflow as tf
import numpy as np
from src.preprocess import preprocess_image, tokenize_captions

def beam_search_predict(model, image, tokenizer, max_len=40, beam_width=3):
    start_token = tokenizer.word_index['<START>']
    end_token = tokenizer.word_index['<END>']
    sequences = [[start_token], 1.0]
    
    img_features = preprocess_image(image)
    img_features = np.expand_dims(img_features, axis=0)
    
    for _ in range(max_len):
        all_candidates = []
        for seq, score in sequences:
            padded_seq = pad_sequences([seq], maxlen=max_len)[0]
            preds = model.predict([img_features, np.array([padded_seq])])[0]
            top_words = np.argsort(preds)[-beam_width:]
            for word in top_words:
                new_seq = seq + [word]
                new_score = score * preds[word]
                all_candidates.append([new_seq, new_score])
        sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
        if sequences[0][0][-1] == end_token:
            break
    
    best_seq = sequences[0][0]
    caption = ' '.join([tokenizer.index_word[i] for i in best_seq if i not in [start_token, end_token]])
    return caption

def main():
    model = tf.keras.models.load_model('../models/saved_model/my_caption_model.h5')
    tokenizer = pd.read_pickle('../data/processed_data/tokenizer.pkl')
    
    test_image = '../data/images/sample_image.jpg'  # Replace with your image
    caption = beam_search_predict(model, test_image, tokenizer)
    print(f"Generated Caption: {caption}")

if __name__ == "__main__":
    main()