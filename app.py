import os
import torch
import numpy as np
import librosa
from flask import Flask, request, jsonify
from flask_cors import CORS
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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa.display

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)
# Enable CORS for the frontend to hit the API locally
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Load Model globally upon server start
def load_model():
    models = []
    device = DEVICE
    model_path = 'models/best_model.pt'
    if os.path.exists(model_path):
        model = CNN2D(num_classes=3).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        models.append(model)
    else:
        print(f"Warning: Model not found at {model_path}")
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
                    logits = model(mel_t, mfcc_t)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                    fold_probs.append(probs)
                
                cycle_probs.append(np.mean(fold_probs, axis=0))
                
        if len(cycle_probs) == 0:
            return jsonify({'error': 'Insufficient audio cycle length.'}), 400
            
        # 3. Final model aggregation
        final_probs = np.mean(cycle_probs, axis=0) # [Normal, Murmur, Abnormal]
        
        entropy = compute_entropy(final_probs)
        CONFIDENCE_THRESHOLD = 0.55
        ABNORMAL_THRESHOLD = 0.345
        ENTROPY_THRESHOLD = 0.95

        normal_prob   = float(final_probs[0])
        murmur_prob   = float(final_probs[1])
        abnormal_prob = float(final_probs[2])
        
        max_idx = np.argmax(final_probs)
        conf = float(final_probs[max_idx])

        if max_idx == 0:
            predicted_class = "Normal"
            recommendation  = "Heart sounds appear normal. Regular checkup recommended."
            flag            = None
        elif max_idx == 1:
            predicted_class = "Murmur"
            recommendation  = "Possible murmur detected. Recommend cardiology consultation."
            flag            = "murmur_detection"
        else:
            predicted_class = "Abnormal"
            recommendation  = "Abnormal sounds detected. Recommend immediate cardiology evaluation."
            flag            = "abnormal_detection"

        return jsonify({
            "status": "success",
            "predicted_class": predicted_class,
            "confidence":      round(conf, 4),
            "probabilities": {
                "Normal":   round(normal_prob, 4),
                "Murmur":   round(murmur_prob, 4),
                "Abnormal": round(abnormal_prob, 4)
            },
            "flag":           flag,
            "recommendation": recommendation,
            "model_note":     "Model trained on 582 PhysioNet recordings. "
                              "Best accuracy with clinical stethoscope recordings.",
            "known_limits": {
                "normal_recall":   "90.2%",
                "murmur_recall":   "80.0%",
                "abnormal_recall": "31.8%"
            },
            "graph": graph_base64
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
    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'evaluation', 'results.json')
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            import json
            data = json.load(f)
            return jsonify(data)
    else:
        return jsonify({'error': 'Metrics not found. Please run evaluation first.'}), 404

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
    app.run(debug=True, port=5001)
