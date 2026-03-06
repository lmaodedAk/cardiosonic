# CardioSonic – AI Heart Sound Classification System

CardioSonic is an AI-powered heart sound analysis system that detects **Normal, Murmur, and Abnormal cardiac patterns** from heart sound recordings using deep learning and spectrogram-based CNN models.

The system converts heart sound recordings into spectrogram representations and uses a convolutional neural network (CNN) to classify cardiac conditions.

---

# Features

- Heart sound classification using deep learning
- Spectrogram-based audio analysis
- Detection of Normal, Murmur, and Abnormal heart sounds
- Interactive web interface for uploading recordings
- Visual spectrogram analysis
- Model evaluation metrics and performance graphs
- Clean project architecture for ML pipelines

---

# Categories

The system classifies heart sounds into:

1. Normal  
2. Murmur  
3. Abnormal  

---

# Project Structure


cardiosonic/
│
├── app.py # Main application entry
├── requirements.txt # Project dependencies
├── README.md
│
├── src/
│ ├── preprocessing/ # Audio preprocessing
│ ├── training/ # Model training code
│ ├── evaluation/ # Evaluation metrics
│
├── models/ # Saved trained models
├── images/ # Visualization images
├── results/ # Evaluation results
├── notebooks/ # Experiment notebooks
├── frontend/ # UI components
├── data/ # Dataset storage


---

# Installation

### 1. Clone the repository

```bash
git clone https://github.com/lmaodedAk/cardiosonic.git
cd cardiosonic
2. Create virtual environment
python -m venv venv
source venv/bin/activate

Mac / Linux

venv\Scripts\activate

Windows

3. Install dependencies
pip install -r requirements.txt
Training the Model

To train the heart sound classification model:

python src/training/train.py

The training pipeline will:

preprocess heart sound recordings

convert audio into spectrograms

train CNN model

generate evaluation metrics

save trained model

Running the Application

Start the application:

python app.py

Then open:

http://localhost:5173

You can:

Upload heart sound recordings

View classification results

See spectrogram analysis

Evaluate model predictions

Model Architecture

The system uses a CNN-based architecture trained on spectrogram representations of heart sound recordings.

Pipeline:

Audio Recording
      ↓
Preprocessing
      ↓
Spectrogram Conversion
      ↓
CNN Model
      ↓
Classification (Normal / Murmur / Abnormal)
Model Performance

Evaluation metrics include:

Accuracy

Precision

Recall

F1 Score

AUC-ROC

Confusion Matrix

Performance results are stored in:

results/
images/
Future Improvements

Larger medical datasets

Domain adaptation for phone recordings

Model compression for mobile deployment

Real-time heart sound monitoring

Integration with digital stethoscopes

License

This project is licensed under the MIT License.

Author

Akshat Jain

AI & Machine Learning Developer
Project: CardioSonic – Intelligent Cardiac Audio Analysis System