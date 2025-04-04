
# ImageSpeak: Automated Captioning with Confidence Scoring
A deep learning project to generate image captions using a CLI tool...

This project implements an image captioning system using deep learning. It leverages a pre-trained InceptionV3 model for image feature extraction and an LSTM-based sequence model to generate natural language captions. The project is built with TensorFlow and trained on a subset of the Flickr8k dataset.

---

## Project Overview

### Features
- **Preprocessing**: Converts raw image-caption pairs into tokenized, preprocessed data.
- **Training**: Trains a model to generate captions from images.
- **Prediction**: Generates captions for new images using the trained model.
- **Dataset**: Uses Flickr8k (8,091 images with 5 captions each), with an option to train on a subset (e.g., 100 images).

### Directory Structure
imagecaptioning/
├── data/
│   ├── caption.txt           # Flickr8k captions (Flickr8k.token.txt renamed)
│   ├── image/                # Flickr8k images (~8,000 .jpg files, optional in repo)
│   └── processed_data/       # Preprocessed files
│       ├── captions_dict.pkl # Image-caption dictionary
│       └── tokenizer.pkl     # Tokenizer for captions
├── models/
│   └── saved_model/          # Trained model
│       └── my_caption_model.h5
├── src/
│   ├── preprocess.py         # Preprocesses data
│   ├── model.py              # Defines the model architecture
│   ├── train.py              # Trains the model
│   └── predict.py            # Generates captions
├── README.md                 # This file
└── requirements.txt          # Python dependencies



---

## Prerequisites

- **Operating System**: macOS, Linux, or Windows
- **Python**: Version 3.11 (or compatible, e.g., 3.8+)
- **Git**: For cloning the repository
- **Git LFS**: If images are tracked via Git LFS
- **Hardware**: 
  - Minimum: 8GB RAM (for 100-image subset)
  - Recommended: 16GB+ RAM or GPU (for full dataset)

---

## Setup Instructions

### Step 1: Clone the Repository
Clone the project from GitHub:
```bash
git clone git@github.com:username/repo.git
cd imagecaptioning
Replace username/repo with your GitHub username and repository name (e.g., charan/imagecaptioning).

Alternative (HTTPS):
bash
git clone https://github.com/username/repo.git
Step 2: Set Up the Virtual Environment

Create a Virtual Environment:
bash
python3 -m venv venv
Activate It:

macOS/Linux:
bash
source venv/bin/activate
Windows:
bash
venv\Scripts\activate

Install Dependencies:
bash
pip install -r requirements.txt

If requirements.txt is missing, create it:
bash
echo "tensorflow>=2.12.0" > requirements.txt
echo "numpy>=1.24.0" >> requirements.txt
echo "pandas>=2.0.0" >> requirements.txt
echo "pillow>=9.0.0" >> requirements.txt
pip install -r requirements.txt
Step 3: Prepare the Data
If Images Are Included (via Git LFS)

Pull LFS Files:
bash
git lfs pull
Downloads ~8,000 .jpg files (1.1 GB) into data/image/.

Verify:
bash
ls data/image/  # macOS/Linux (should list ~8,000 .jpg files)
dir data\image\ # Windows
If Images Are Excluded (via .gitignore)
Download Flickr8k Dataset:
Source: Kaggle Flickr8k (requires login).
Download and extract:
Images/ (contains .jpg files)
Flickr8k_text/Flickr8k.token.txt (captions)

Organize Files:
bash
mkdir -p data/image
mv Images/* data/image/
mv Flickr8k_text/Flickr8k.token.txt data/caption.txt

Verify:
bash
ls data/image/  # Should list ~8,000 .jpg files
ls data/caption.txt  # Should exist
Preprocessed Files
data/processed_data/captions_dict.pkl and tokenizer.pkl should be in the repo.

If missing, regenerate:
bash
python src/preprocess.py
Takes ~1-2 minutes, requires data/caption.txt.
Step 4: Verify or Train the Model

Check Model:
bash
ls models/saved_model/my_caption_model.h5  # macOS/Linux
dir models\saved_model\my_caption_model.h5 # Windows

If Missing, Train:
bash
python src/train.py
Trains on 100 images by default (~40 minutes on CPU).

For full dataset (8,091 images), edit src/train.py:
python
# Remove or adjust this line in main():
captions_dict = {k: v for k, v in list(captions_dict.items())[:100]}
Requires more RAM/GPU, takes hours on CPU.
Step 5: Generate Captions

Run Prediction:
bash
python src/predict.py
Expected Output:
text
Loading model and tokenizer...
Generating caption for /path/to/imagecaptioning/data/image/1000268201_693b08cb0e.jpg...
Generated caption: little girl in a pink dress going into a wooden cabin
Caption may vary slightly due to limited training data.
Usage
Predict on Custom Images:
Place your image (e.g., my_image.jpg) in data/image/.
Edit src/predict.py:
python
sample_image = os.path.join(image_dir, 'my_image.jpg')

Run:
bash
python src/predict.py
Retrain with More Data:
Adjust train.py to use more images (see Step 4) and rerun.
Project Details
Model Architecture
Image Encoder: InceptionV3 (pre-trained, feature extraction).
Text Decoder: LSTM with embedding layer.
Training:
Optimizer: Adam
Loss: Categorical Crossentropy
Epochs: 10 (default)
Batch Size: 32
Performance
Trained on 100 images:
Final Training Loss: ~1.6
Final Training Accuracy: ~0.57
Validation Loss: ~4.7
Validation Accuracy: ~0.29
Full dataset training improves generalization (requires more compute).
Troubleshooting
ModuleNotFoundError: No module named 'src':
Run scripts from imagecaptioning/ directory, not src/.
FileNotFoundError:
Ensure data/image/, data/caption.txt, and models/saved_model/my_caption_model.h5 exist.
Regenerate missing files with preprocess.py or train.py.
Memory Issues:
Reduce dataset size in train.py:
python
captions_dict = {k: v for k, v in list(captions_dict.items())[:50]}  # Use 50 images
Close other applications or use a machine with more RAM/GPU.
Push Errors:

If cloning fails due to LFS, ensure git lfs is installed:
bash
git lfs install
git lfs pull
Contributing
Fork the repository.
Create a branch: git checkout -b feature-name.
Commit changes: git commit -m "Add feature".
Push: git push origin feature-name.
Open a pull request.
License
This project is licensed under the MIT License (add a LICENSE file if desired).

Acknowledgments
Flickr8k Dataset: For training data.
TensorFlow: For deep learning framework.
xAI: For inspiration (via Grok assistance!).
text

---

### How to Use
1. Open VS Code in your `imagecaptioning/` directory.
2. Create or open `README.md`.
3. Copy the entire block above (click and drag from the top `#` to the bottom `!`, or triple-click to select all).
4. Paste it into `README.md`.
5. Replace `username/repo` with your GitHub path (e.g., `charan/imagecaptioning`).

6. Save, then commit and push:
   ```bash
   git add README.md
   git commit -m "Add detailed README"
   git push


   markdown
## CLI Usage
Run the captioning tool from the command line:
```bash
python app.py data/image/1000268201_693b08cb0e.jpg
Options:

--model: Specify custom model path (default: models/saved_model/my_caption_model.h5)
--tokenizer: Specify custom tokenizer path (default: data/processed_data/tokenizer.pkl)
--verbose: Show detailed output with per-word confidence scores
Example:

bash
python app.py data/image/1000268201_693b08cb0e.jpg --verbose
Output example:

text
Loading model from: models/saved_model/my_caption_model.h5
Loading tokenizer from: data/processed_data/tokenizer.pkl
Processing image: data/image/1000268201_693b08cb0e.jpg
Predicted word: 'a' (Confidence: 0.95)
Predicted word: 'little' (Confidence: 0.89)
...
Generated Caption: a little girl in a pink dress going into a wooden cabin
Average Confidence: 0.90
text
- Commit:
  ```bash
  git add README.md
  git commit -m "Update README with CLI usage instructions"
  git push
Action
Decide on Flask:
Remove: git checkout -- requirements.txt
Keep: git add requirements.txt && git commit -m "Add Flask for future web app" && git push
I’d go with remove since you’re CLI-focused—your call!

Ignore static/:
bash
echo "static/" >> .gitignore
git add .gitignore
git commit -m "Ignore static folder"
git push

Test app.py:
bash
python app.py data/image/1000268201_693b08cb0e.jpg --verbose