import argparse
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import pickle

def generate_caption(image_path, model, tokenizer, max_length=40, verbose=False):
    """Generate a caption for the given image with confidence scores."""
    # Load and preprocess image
    image = load_img(image_path, target_size=(299, 299))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalize

    # Start with '<start>' token
    caption = '<start>'
    confidence_scores = []
    prev_word = None  # To detect repetition
    
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([caption])[0]
        sequence = np.pad(sequence, (0, max_length - len(sequence)), mode='constant', constant_values=0)
        sequence = np.expand_dims(sequence, axis=0)
        
        pred = model.predict([image, sequence], verbose=0)
        pred_idx = np.argmax(pred[0])
        confidence = np.max(pred[0])  # Confidence of the predicted word
        
        word = tokenizer.index_word.get(pred_idx, '')
        # Stop on '<end>' or 'end' explicitly
        if word in ['<end>', 'end']:
            break
        # Prevent infinite repetition
        if word == prev_word and len(confidence_scores) > 5:  # Arbitrary threshold
            break
        if word:
            caption += ' ' + word
            confidence_scores.append(confidence)
            prev_word = word
            if verbose:
                print(f"Predicted word: '{word}' (Confidence: {confidence:.2f})")
    
    caption = caption.replace('<start>', '').strip()
    avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
    return caption, avg_confidence

def main():
    """Command-line interface for image captioning with optional verbose output and custom paths."""
    parser = argparse.ArgumentParser(description="Generate captions for images using a trained model.")
    parser.add_argument('image_path', type=str, help="Path to the image file (e.g., data/image/example.jpg)")
    parser.add_argument('--model', type=str, default='models/saved_model/my_caption_model.h5', 
                        help="Path to the model file (default: models/saved_model/my_caption_model.h5)")
    parser.add_argument('--tokenizer', type=str, default='data/processed_data/tokenizer.pkl', 
                        help="Path to the tokenizer file (default: data/processed_data/tokenizer.pkl)")
    parser.add_argument('--verbose', action='store_true', help="Print detailed output with confidence scores")
    args = parser.parse_args()

    # Verbose output
    if args.verbose:
        print(f"Loading model from: {args.model}")
        print(f"Loading tokenizer from: {args.tokenizer}")
        print(f"Processing image: {args.image_path}")

    # Validate file paths
    if not os.path.exists(args.image_path):
        print(f"Error: Image file '{args.image_path}' not found.")
        return
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found.")
        return
    if not os.path.exists(args.tokenizer):
        print(f"Error: Tokenizer file '{args.tokenizer}' not found.")
        return

    # Load model and tokenizer
    try:
        model = load_model(args.model)
        with open(args.tokenizer, 'rb') as f:
            tokenizer = pickle.load(f)
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return

    # Generate caption with confidence
    caption, avg_confidence = generate_caption(args.image_path, model, tokenizer, verbose=args.verbose)
    print(f"Generated Caption: {caption}")
    if args.verbose:
        print(f"Average Confidence: {avg_confidence:.2f}")

if __name__ == "__main__":
    main()