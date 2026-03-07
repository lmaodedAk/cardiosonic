import os
import torch
import numpy as np
import librosa
from flask import Flask, request, jsonify
from flask_cors import CORS

from src.training.models.cnn2d import CNN2D
from src.preprocessing.preprocess import preprocess_audio
from src.preprocessing.features import extract_features
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa.display

def compute_entropy(probs):
    """Calculate Shannon entropy for uncertainty estimation."""
    return -np.sum(probs * np.log(probs + 1e-9))

def safe_inference(logits_output, abnormal_threshold=0.30, entropy_threshold=0.95):
    """Replaces argmax with uncertainty and clinical confidence gating."""
    probs = torch.softmax(logits_output, dim=1).cpu().numpy()[0]
    entropy = compute_entropy(probs)
    
    # 1. Uncertainty Gating
    if entropy > entropy_threshold:
        return "Uncertain", probs, "High background noise or unintelligible recording.", "high_entropy"
    
    # 2. Confidence Gating
    if np.max(probs) < 0.55:
        return "Uncertain", probs, "Confidence too low for reliable prediction.", "low_confidence"
        
    # 3. Validation-Tuned Decision Boundaries
    if probs[2] >= abnormal_threshold:
        return "Abnormal", probs, "URGENT: Abnormal Signatures Detected. Refer to Cardiologist.", "abnormal_detection"
    elif probs[1] > probs[0]:
        return "Murmur", probs, "Warning: Heart Murmur profile identified. Recommend cardiology consultation.", "murmur_detection"
    else:
        return "Normal", probs, "Heart sounds appear normal. Regular checkup recommended.", None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["Normal", "Murmur", "Abnormal"]
HARDCODED_METRICS = {
  "accuracy": 0.702,
  "normal_accuracy": 0.902,
  "murmur_accuracy": 0.800,
  "abnormal_accuracy": 0.318,
  "total_predictions": 0,
  "normal_count": 0,
  "abnormal_count": 0,
  "confusion_matrix": [[0,0],[0,0]]
}

app = Flask(__name__)
# Enable CORS for the frontend to hit the API locally
CORS(app, resources={r"/api/*": {"origins": "*"}})

from huggingface_hub import hf_hub_download

# Load Model globally upon server start
def load_model():
    models = []
    device = DEVICE
    
    try:
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/best_model.pt')
        if not os.path.exists(model_path):
            print("Local model not found! Downloading from Hub...")
            model_path = hf_hub_download(
                repo_id="akshat23456/cardiosonic-model",
                filename="best_model.pt"
            )
            
        model = CNN2D(num_classes=3).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        models.append(model)
        print(f"Model successfully loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        
    return models

print("Starting server... Loading Model.")
ensemble_models = load_model()

@app.route('/api/analyze', methods=['POST'])
def analyze():
    # 1. Receiver
    if 'audio' not in request.files:
        return jsonify({'error': 'Missing audio file in request.'}), 400
        
    file = request.files['audio']
    temp_path = "temp_inference.wav"
    file.save(temp_path)
    
    try:
        # 2. Pipeline Processing
        y = preprocess_audio(temp_path)
        log_mel_np, mfcc_np = extract_features(y)
        
        cycles = [ (log_mel_np, mfcc_np) ]
        
        cycle_probs = []
        graph_base64 = None
        for i, (lmel, lmfcc) in enumerate(cycles):
            if i == 0:
                plt.figure(figsize=(10, 4))
                librosa.display.specshow(lmel, sr=2000, x_axis='time', y_axis='mel')
                plt.colorbar(format='%+2.0f dB')
                plt.title('Log-Mel Spectrogram of Cardiac Cycle')
                plt.tight_layout()
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                plt.close()
                buf.seek(0)
                graph_base64 = base64.b64encode(buf.read()).decode('utf-8')
            
            mel_t = torch.tensor(lmel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
            mfcc_t = torch.tensor(lmfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                fold_probs = []
                for model in ensemble_models:
                    mel_input = torch.nan_to_num(mel_t, nan=0.0, posinf=0.0, neginf=0.0)
                    mfcc_input = torch.nan_to_num(mfcc_t, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    logits = model(mel_input, mfcc_input)
                    logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
                    
                    probs = torch.softmax(logits, dim=1)
                    final_probs_arr = probs.detach().cpu().numpy().flatten()
                    final_probs_arr = np.nan_to_num(final_probs_arr, nan=0.0, posinf=1.0, neginf=0.0)
                    
                    fold_probs.append(final_probs_arr)
                
                cycle_probs.append(np.mean(fold_probs, axis=0))
                
        if len(cycle_probs) == 0:
            return jsonify({'error': 'Insufficient audio cycle length.'}), 400
            
        # 3. Final model aggregation
        # cycle_probs shape may be causing issues, flatten/ensure correct axes
        final_probs = np.mean(cycle_probs, axis=0) # [Normal, Murmur, Abnormal]
        
        # Flatten probabilities in case of nested arrays to prevent "invalid index to scalar variable"
        final_probs = np.array(final_probs).astype(np.float32).flatten()
        final_probs = np.nan_to_num(final_probs, nan=0.0, posinf=1.0, neginf=0.0)
        
        if len(final_probs) == 1:
            conf = float(final_probs[0])
            predicted_class = CLASS_NAMES[0] if conf < 0.5 else CLASS_NAMES[2]
            probabilities_dict = {"Normal": float(1 - conf), "Murmur": 0.0, "Abnormal": conf}
        else:
            max_idx = int(np.argmax(final_probs))
            conf = float(final_probs[max_idx])
            
            if max_idx >= len(CLASS_NAMES):
                predicted_class = "Unknown"
            else:
                predicted_class = CLASS_NAMES[max_idx]
                
            probabilities_dict = {
                "Normal": float(final_probs[0]) if len(final_probs) > 0 else 0.0,
                "Murmur": float(final_probs[1]) if len(final_probs) > 1 else 0.0,
                "Abnormal": float(final_probs[2]) if len(final_probs) > 2 else 0.0
            }

        return jsonify({
            "predicted_class": predicted_class,
            "confidence": conf,
            "probabilities": probabilities_dict,
            "spectrogram": graph_base64 if 'graph_base64' in locals() else None
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    return jsonify(HARDCODED_METRICS)

from flask import send_file

@app.route('/api/graphs/<graph_type>', methods=['GET'])
def get_graph(graph_type):
    valid_graphs = {'loss': 'loss_graph.png', 'roc': 'roc_graph.png'}
    if graph_type not in valid_graphs:
        return jsonify({'error': 'Invalid graph type requested.'}), 400
        
    graph_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'evaluation', valid_graphs[graph_type])
    if os.path.exists(graph_path):
        return send_file(graph_path, mimetype='image/png')
    else:
        return jsonify({'error': 'Graph not found.'}), 404

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 7860))
    app.run(host='0.0.0.0', port=port)
