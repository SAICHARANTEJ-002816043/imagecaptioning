import argparse
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import pickle

def generate_caption(image_path, model, tokenizer, max_length=40, beam_width=3, verbose=False):
    """Generate a caption for the given image using beam search with confidence scores."""
    # Load and preprocess image
    image = load_img(image_path, target_size=(299, 299))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalize

    # Beam search initialization
    start = ['<start>']
    sequences = [(start, 0.0)]  # List of (caption list, cumulative log probability)
    
    for _ in range(max_length):
        all_candidates = []
        for seq, score in sequences:
            if seq[-1] in ['<end>', 'end']:  # Stop if sequence ends
                all_candidates.append((seq, score))
                continue
            
            # Convert current sequence to tokenized input
            sequence = tokenizer.texts_to_sequences([' '.join(seq)])[0]
            sequence = np.pad(sequence, (0, max_length - len(sequence)), mode='constant', constant_values=0)
            sequence = np.expand_dims(sequence, axis=0)
            
            # Predict next word probabilities
            pred = model.predict([image, sequence], verbose=0)
            top_preds = np.argsort(pred[0])[-beam_width:]  # Get top beam_width indices
            
            # Generate candidates
            for idx in top_preds:
                word = tokenizer.index_word.get(idx, '')
                prob = pred[0][idx]  # Probability of the word
                new_score = score + np.log(prob + 1e-10)  # Add log prob (avoid log(0))
                new_seq = seq + [word]
                all_candidates.append((new_seq, new_score))
        
        # Select top beam_width sequences
        sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
    
    # Extract the best sequence
    best_seq, best_score = sequences[0]
    caption = ' '.join([w for w in best_seq if w not in ['<start>', '<end>', 'end']])
    
    # Confidence approximation (average log prob converted to prob-like value)
    avg_confidence = np.exp(best_score / max(1, len(best_seq) - 2))  # -2 for <start>, <end>
    
    if verbose:
        print("Beam Search Path (Best Sequence):")
        for word in best_seq:
            if word not in ['<start>', '<end>', 'end']:
                # Confidence per word is approximate since beam search mixes probs
                print(f"Predicted word: '{word}' (Approx. Confidence: {avg_confidence:.2f})")
    
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
    parser.add_argument('--beam_width', type=int, default=3, help="Beam width for search (default: 3)")
    args = parser.parse_args()

    # Verbose output
    if args.verbose:
        print(f"Loading model from: {args.model}")
        print(f"Loading tokenizer from: {args.tokenizer}")
        print(f"Processing image: {args.image_path}")
        print(f"Using beam width: {args.beam_width}")

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
    caption, avg_confidence = generate_caption(
        args.image_path, model, tokenizer, max_length=40, beam_width=args.beam_width, verbose=args.verbose
    )
    print(f"Generated Caption: {caption}")
    if args.verbose:
        print(f"Average Confidence: {avg_confidence:.2f}")

if __name__ == "__main__":
    main()