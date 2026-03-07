import { useState, useRef, useEffect } from 'react';
import { Activity, UploadCloud, FileAudio, Play, Loader2, ArrowRight } from 'lucide-react';
import axios from 'axios';
import './index.css';

function PredictionResult({ result }) {
  if (!result) return null;

  const isUncertain = result.predicted_class === "Uncertain";

  const colors = {
    Normal: { bg: "#EAFAF1", border: "#27AE60", text: "#1E8449" },
    Murmur: { bg: "#FEF9E7", border: "#F39C12", text: "#D68910" },
    Abnormal: { bg: "#FDEDEC", border: "#E74C3C", text: "#C0392B" },
    Uncertain: { bg: "#F2F3F4", border: "#7F8C8D", text: "#626567" }
  };

  const color = colors[result.predicted_class] || colors.Uncertain;

  return (
    <div style={{
      border: `2px solid ${color.border}`,
      borderRadius: "12px",
      padding: "24px",
      backgroundColor: color.bg,
      marginTop: "20px"
    }}>

      {/* Main prediction */}
      <div style={{
        fontSize: "2rem", fontWeight: "bold",
        color: color.text, textAlign: "center"
      }}>
        {isUncertain ? "⚠️ Uncertain" : `${result.predicted_class}`}
      </div>

      {/* Confidence */}
      <div style={{
        textAlign: "center", marginTop: "8px",
        color: "#666", fontSize: "14px"
      }}>
        Confidence: {(result.confidence * 100).toFixed(1)}%
        {result.confidence < 0.60 && (
          <span style={{ color: "#E74C3C", marginLeft: "8px" }}>
            (Too low for reliable prediction)
          </span>
        )}
      </div>

      {/* Probability bars */}
      <div style={{ marginTop: "16px" }}>
        {Object.entries(result.probabilities).map(([cls, prob]) => (
          <div key={cls} style={{
            display: "flex", alignItems: "center",
            gap: "10px", margin: "6px 0"
          }}>
            <span style={{
              width: "80px", fontSize: "13px",
              fontWeight: "600"
            }}>{cls}</span>
            <div style={{
              flex: 1, background: "#ddd",
              borderRadius: "4px", height: "18px"
            }}>
              <div style={{
                width: `${(prob * 100).toFixed(1)}%`,
                height: "100%",
                borderRadius: "4px",
                backgroundColor: cls === "Normal" ? "#27AE60" :
                  cls === "Murmur" ? "#F39C12" :
                    "#E74C3C",
                transition: "width 0.5s ease"
              }} />
            </div>
            <span style={{
              width: "45px", fontSize: "13px",
              textAlign: "right"
            }}>
              {(prob * 100).toFixed(1)}%
            </span>
          </div>
        ))}
      </div>

      {/* Recommendation */}
      <div style={{
        marginTop: "16px",
        padding: "10px 14px",
        background: "rgba(255,255,255,0.6)",
        borderRadius: "8px",
        fontSize: "14px"
      }}>
        {result.recommendation}
      </div>

      {/* Uncertain extra warning */}
      {isUncertain && (
        <div style={{
          marginTop: "12px",
          padding: "10px 14px",
          background: "#FEF9E7",
          border: "1px solid #F39C12",
          borderRadius: "8px",
          fontSize: "13px",
          color: "#7D6608"
        }}>
          <strong>Why uncertain?</strong> The model's confidence was below
          the reliability threshold. This often happens with:
          recordings from phone microphones, excessive background noise,
          recordings shorter than 5 seconds, or non-stethoscope recordings.
        </div>
      )}

      {/* Model limitation footer */}
      <div style={{
        marginTop: "14px", fontSize: "11px",
        color: "#999", textAlign: "center"
      }}>
        Model accuracy: Normal 90.2% · Murmur 80.0% · Abnormal 31.8%
        &nbsp;|&nbsp; Always consult a cardiologist for diagnosis.
      </div>

    </div>
  );
}

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5001';

function App() {
  const [file, setFile] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [metrics, setMetrics] = useState(null);

  useEffect(() => {
    fetch(`${API_URL}/api/metrics`)
      .then(res => res.json())
      .then(data => {
        if (!data.error) {
          setMetrics(data || {});
        } else {
          console.warn("Metrics endpoint returned an error, falling back to empty metrics.", data.error);
          setMetrics({});
        }
      })
      .catch(err => console.error("Could not check metrics on backend", err));
  }, []);

  const fileInputRef = useRef(null);
  const resultsRef = useRef(null);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setAudioUrl(URL.createObjectURL(selectedFile));
      setResults(null);
      setError(null);
    }
  };

  const processAudio = async () => {
    if (!file) return;

    setIsAnalyzing(true);
    setError(null);

    const formData = new FormData();
    formData.append('audio', file);

    try {
      const response = await axios.post(`${API_URL}/api/analyze`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setResults(response.data);
      setTimeout(() => {
        resultsRef.current?.scrollIntoView({ behavior: 'smooth' });
      }, 100);
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.error || "Failed to analyze audio. Ensure the backend server is running.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="App">
      {/* Navigation */}
      <nav className="container">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontWeight: 'bold', fontSize: '1.25rem', color: 'var(--navy)' }}>
          <Activity color="var(--teal)" size={28} />
          CardioSonic
        </div>
        <div className="nav-links">
          <a href="#about">About</a>
          <a href="#how-it-works">How It Works</a>
          <a href="#architecture">Architecture</a>
          <a href="#evaluation">Evaluation</a>
        </div>
        <button className="btn btn-primary" onClick={() => document.getElementById('upload').scrollIntoView({ behavior: 'smooth' })}>
          Upload Recording
        </button>
      </nav>

      {/* Hero Section */}
      <header className="hero container">
        <div className="hero-bg"></div>
        <div className="badge">
          <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: 'var(--teal)', display: 'inline-block' }}></span>
          RESEARCH PREVIEW
        </div>
        <h1>AI-Powered Heart Sound Analysis for<br />Early Cardiac Detection</h1>
        <p style={{ maxWidth: '800px', margin: '0 auto 2.5rem auto' }}>
          CardioSonic leverages advanced deep learning models to analyze phonocardiogram audio signals and assist in early detection of cardiac abnormalities.
        </p>
        <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center' }}>
          <button className="btn btn-primary" onClick={() => document.getElementById('upload').scrollIntoView({ behavior: 'smooth' })}>
            Upload Recording
          </button>
          <button className="btn btn-outline" onClick={() => document.getElementById('architecture').scrollIntoView({ behavior: 'smooth' })}>
            View Research
          </button>
        </div>
      </header>

      {/* About Section */}
      <section id="about" className="container">
        <div className="section-header">
          <span className="eyebrow">ABOUT THE SYSTEM</span>
          <h2>Understanding Cardiac Sound Intelligence</h2>
          <p style={{ maxWidth: '700px' }}>CardioSonic transforms raw phonocardiogram signals into actionable cardiac health insights through a rigorous computational pipeline.</p>
        </div>

        <div className="grid-2">
          <div className="card">
            <Activity className="text-teal mb-2" size={32} />
            <h3 className="mb-1">Phonocardiogram Analysis</h3>
            <p style={{ fontSize: '0.95rem' }}>A phonocardiogram (PCG) records heart sounds using a digital stethoscope, capturing the acoustic signatures of cardiac valve movements and blood flow patterns.</p>
          </div>
          <div className="card">
            <Activity className="text-teal mb-2" size={32} />
            <h3 className="mb-1">Why Heart Sound Analysis Matters</h3>
            <p style={{ fontSize: '0.95rem' }}>Heart sound analysis provides a non-invasive, cost-effective screening method that can detect structural and functional cardiac abnormalities before symptoms manifest.</p>
          </div>
        </div>
      </section>

      {/* Upload Interface Dashboard */}
      <section id="upload" className="container" style={{ background: 'var(--white)', padding: '4rem', borderRadius: '1.5rem', boxShadow: '0 4px 6px -1px rgba(0,0,0,0.05)' }}>
        <div className="section-header text-center">
          <span className="eyebrow">CLINICAL DASHBOARD</span>
          <h2>Acoustic Inference Engine</h2>
          <p>Upload a .wav file from a digital stethoscope to run a CNN2D classification.</p>
        </div>

        <div style={{
          background: "#EAF2FB",
          border: "1px solid #2980B9",
          borderRadius: "8px",
          padding: "12px 16px",
          fontSize: "13px",
          marginBottom: "24px",
          maxWidth: "600px",
          margin: "0 auto 24px auto"
        }}>
          <strong>For best results:</strong> Upload recordings made with a
          digital stethoscope or clinical PCG device. Phone microphone
          recordings may produce uncertain results. Recordings should be
          5–30 seconds with minimal background noise.
        </div>

        {!results && !isAnalyzing && (
          <div
            className="upload-zone"
            onClick={() => fileInputRef.current.click()}
          >
            <UploadCloud className="upload-icon" size={48} />
            <h3 className="mb-1">Click or drag audio to upload</h3>
            <p style={{ fontSize: '0.9rem' }}>Supported format: WAV (16-bit, mono, 2000Hz)</p>
            <input
              type="file"
              accept=".wav"
              style={{ display: 'none' }}
              ref={fileInputRef}
              onChange={handleFileChange}
            />
            {file && (
              <div style={{ marginTop: '1.5rem', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem', color: 'var(--navy)', fontWeight: '500' }}>
                <FileAudio size={20} className="text-teal" />
                {file.name}
              </div>
            )}

            {audioUrl && (
              <div
                style={{ marginTop: '1rem', width: '100%', display: 'flex', justifyContent: 'center' }}
                onClick={(e) => e.stopPropagation()}
              >
                <audio controls src={audioUrl} style={{ width: '80%', maxWidth: '400px', height: '40px', outline: 'none' }} />
              </div>
            )}

            {file && (
              <button
                className="btn btn-primary mt-4"
                onClick={(e) => { e.stopPropagation(); processAudio(); }}
              >
                Run AI Analysis <ArrowRight size={18} />
              </button>
            )}
          </div>
        )}

        {isAnalyzing && (
          <div className="upload-zone active" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <Loader2 className="upload-icon mb-4" size={48} style={{ animation: 'spin 2s linear infinite' }} />
            <h3>Processing through CNN2D Model...</h3>
            <p style={{ fontSize: '0.9rem' }}>Segmenting cycles, extracting Log-Mel spectrograms, and analyzing.</p>
          </div>
        )}

        {error && (
          <div style={{ background: '#fef2f2', color: 'var(--danger)', padding: '1rem', borderRadius: '0.5rem', marginTop: '1rem', textAlign: 'center', fontWeight: '500' }}>
            {error}
            <div className="mt-2">
              <button className="btn btn-outline" onClick={() => { setError(null); setFile(null); }}>Try Again</button>
            </div>
          </div>
        )}

        {results && (
          <div className="results-dashboard mt-4" ref={resultsRef}>
            <h3 className="mb-4">Evaluation Complete</h3>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>
              <PredictionResult result={results} />

              <div className="card" style={{ marginTop: '20px' }}>
                <span className="eyebrow">CARDIAC CYCLE SPECTROGRAM</span>
                <p style={{ fontSize: '0.875rem', color: 'var(--slate)', marginBottom: '1rem' }}>
                  The model analyzes the audio by converting it into a visual representation of acoustic frequencies over time.
                </p>
                {results.graph && (
                  <img
                    src={`data:image/png;base64,${results.graph}`}
                    alt="Log-Mel Spectrogram"
                    style={{ width: '100%', borderRadius: '0.5rem', border: '1px solid var(--card-border)' }}
                  />
                )}
              </div>
            </div>

            <div className="text-center mt-8">
              <button className="btn btn-outline" onClick={() => { setResults(null); setFile(null); }}>
                Analyze Another Recording
              </button>
            </div>
          </div>
        )}
      </section>

      {/* Architecture Display */}
      <section id="architecture" className="container">
        <div className="section-header">
          <span className="eyebrow">MODEL DESIGN</span>
          <h2>Architecture & Performance</h2>
          <p>A CNN-based ensemble architecture optimized for temporal pattern recognition in cardiac audio signals.</p>
        </div>

        <h3 className="mb-4" style={{ fontSize: '0.875rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--slate)' }}>NETWORK ARCHITECTURE</h3>

        <div className="arch-list">
          <div className="arch-item">
            <div className="arch-number">1</div>
            <div className="arch-content">
              <h4>Input Layer</h4>
              <p>Log-Mel Spectrogram (128 bins, 47 frames)</p>
            </div>
          </div>
          <div className="arch-item">
            <div className="arch-number">2</div>
            <div className="arch-content">
              <h4>Conv2D Blocks (x2)</h4>
              <p>32 & 64 filters, kernel=3x3, ReLU, BatchNorm, MaxPool(2x2)</p>
            </div>
          </div>
          <div className="arch-item">
            <div className="arch-number">3</div>
            <div className="arch-content">
              <h4>Feature Extraction</h4>
              <p>128 filters, kernel=3x3, AdaptiveAvgPool2d</p>
            </div>
          </div>
          <div className="arch-item">
            <div className="arch-number">4</div>
            <div className="arch-content">
              <h4>Dense Classifier</h4>
              <p>256 -{'>'} 128 -{'>'} 3 (Normal / Murmur / Abnormal), Dropout(0.4)</p>
            </div>
          </div>
          <div className="arch-item">
            <div className="arch-number">5</div>
            <div className="arch-content">
              <h4>Model Output</h4>
              <p>Probabilistic Classification via CNN2D Architecture</p>
            </div>
          </div>
        </div>
      </section>

      {/* Evaluation Dashboard Section */}
      <section id="evaluation" className="container">
        <div className="section-header">
          <h2>Evaluation Dashboard</h2>
          <p>Comprehensive model evaluation on held-out test data with cross-validated performance metrics.</p>
        </div>

        <span className="eyebrow">EVALUATION METRICS</span>
        {metrics ? (
          <>
            <div className="grid-3 mb-8 mt-4">
              <div className="metric-card">
                <h3>{((metrics.accuracy || 0) * 100).toFixed(1)}%</h3>
                <div className="metric-name">Accuracy</div>
                <div className="metric-desc">Overall classification accuracy</div>
              </div>
              <div className="metric-card">
                <h3>{(((metrics.precision || metrics.precision_macro) || 0) * 100).toFixed(1)}%</h3>
                <div className="metric-name">Weighted Precision</div>
                <div className="metric-desc">Positive predictive value</div>
              </div>
              <div className="metric-card">
                <h3>{(((metrics.recall || metrics.recall_macro) || 0) * 100).toFixed(1)}%</h3>
                <div className="metric-name">Weighted Recall</div>
                <div className="metric-desc">Sensitivity / True positive rate</div>
              </div>
              <div className="metric-card">
                <h3>{(((metrics.f1_score || metrics.weighted_f1) || 0) * 100).toFixed(1)}%</h3>
                <div className="metric-name">Weighted F1</div>
                <div className="metric-desc">Harmonic mean of precision & recall</div>
              </div>
              <div className="metric-card">
                <h3>{(metrics.auc_roc || metrics.auc || 0).toFixed(3)}</h3>
                <div className="metric-name">AUC-ROC</div>
                <div className="metric-desc">Area under ROC curve</div>
              </div>
              <div className="metric-card">
                <h3>{(metrics.mcc || 0).toFixed(3)}</h3>
                <div className="metric-name">MCC</div>
                <div className="metric-desc">Matthews Correlation Coefficient</div>
              </div>
            </div>

            <span className="eyebrow">CONFIDENCE MATRIX</span>
            <div className="card mt-4 mb-8">
              <div className="confusion-matrix">
                <div></div>
                <div className="cm-header">Pred: <span style={{ opacity: 0.7 }}>Normal</span></div>
                <div className="cm-header">Pred: <span style={{ opacity: 0.7 }}>Murmur</span></div>
                <div className="cm-header">Pred: <span style={{ opacity: 0.7 }}>Abnormal</span></div>

                <div className="cm-label">Actual: <span style={{ opacity: 0.7, marginLeft: '0.25rem' }}>Normal</span></div>
                <div className="cm-cell">{metrics.confusion_matrix?.[0]?.[0] || 0}</div>
                <div className="cm-cell" style={{ background: '#fef3c7', color: 'var(--warning)' }}>{metrics.confusion_matrix?.[0]?.[1] || 0}</div>
                <div className="cm-cell" style={{ background: '#f8fafc', color: 'var(--slate)' }}>{metrics.confusion_matrix?.[0]?.[2] || 0}</div>

                <div className="cm-label">Actual: <span style={{ opacity: 0.7, marginLeft: '0.25rem' }}>Murmur</span></div>
                <div className="cm-cell" style={{ background: '#f8fafc', color: 'var(--slate)' }}>{metrics.confusion_matrix?.[1]?.[0] || 0}</div>
                <div className="cm-cell">{metrics.confusion_matrix?.[1]?.[1] || 0}</div>
                <div className="cm-cell" style={{ background: '#f8fafc', color: 'var(--slate)' }}>{metrics.confusion_matrix?.[1]?.[2] || 0}</div>

                <div className="cm-label">Actual: <span style={{ opacity: 0.7, marginLeft: '0.25rem' }}>Abnormal</span></div>
                <div className="cm-cell" style={{ background: '#f8fafc', color: 'var(--slate)' }}>{metrics.confusion_matrix?.[2]?.[0] || 0}</div>
                <div className="cm-cell" style={{ background: '#fef3c7', color: 'var(--warning)' }}>{metrics.confusion_matrix?.[2]?.[1] || 0}</div>
                <div className="cm-cell" style={{ background: '#fef2f2', color: 'var(--danger)' }}>{metrics.confusion_matrix?.[2]?.[2] || 0}</div>
              </div>
            </div>

            <span className="eyebrow">TRAINING & VALIDATION GRAPHS</span>
            <div className="grid-2 mt-4 mb-8">
              <div className="card">
                <h4 style={{ marginBottom: '1rem', color: 'var(--navy)' }}>Training Loss History</h4>
                <img
                  src={`${API_URL}/api/graphs/loss`}
                  alt="Training Loss Graph"
                  style={{ width: '100%', borderRadius: '0.5rem', border: '1px solid var(--card-border)' }}
                  onError={(e) => e.target.style.display = 'none'}
                />
              </div>
              <div className="card">
                <h4 style={{ marginBottom: '1rem', color: 'var(--navy)' }}>Multi-Class ROC Curve</h4>
                <img
                  src={`${API_URL}/api/graphs/roc`}
                  alt="ROC Curve Graph"
                  style={{ width: '100%', borderRadius: '0.5rem', border: '1px solid var(--card-border)' }}
                  onError={(e) => e.target.style.display = 'none'}
                />
              </div>
            </div>
          </>
        ) : (
          <div className="card mt-4 mb-8" style={{ textAlign: 'center', padding: '2rem' }}>
            <p>Evaluation metrics not yet available. Please run the model evaluation script.</p>
          </div>
        )}
      </section>

      {/* Dataset & Methodology Section */}
      <section id="dataset-research" className="container" style={{ background: 'var(--white)', padding: '4rem', borderRadius: '1.5rem', boxShadow: '0 4px 6px -1px rgba(0,0,0,0.05)' }}>
        <div className="section-header">
          <span className="eyebrow text-teal">RESEARCH</span>
          <h2>Dataset & Methodology</h2>
          <p>Details on the data sources, preprocessing pipeline, and training methodology used to develop the CardioSonic model.</p>
        </div>

        <div className="grid-2">
          <div className="card">
            <span className="eyebrow">DATASET OVERVIEW</span>
            <div className="mt-4">
              <div className="dataset-stat">
                <span className="label">Total Samples</span>
                <span className="value">3,240</span>
              </div>
              <div className="dataset-stat">
                <span className="label">Training Set</span>
                <span className="value">2,592 (80%)</span>
              </div>
              <div className="dataset-stat">
                <span className="label">Validation Set</span>
                <span className="value">324 (10%)</span>
              </div>
              <div className="dataset-stat">
                <span className="label">Test Set</span>
                <span className="value">324 (10%)</span>
              </div>
              <div className="dataset-stat">
                <span className="label">Normal Samples</span>
                <span className="value">1,585 (48.9%)</span>
              </div>
              <div className="dataset-stat" style={{ borderBottom: 'none' }}>
                <span className="label">Abnormal Samples</span>
                <span className="value">1,655 (51.1%)</span>
              </div>
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            <div className="card" style={{ padding: '1.5rem' }}>
              <h4 style={{ color: 'var(--navy)', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Activity size={18} className="text-teal" /> Dataset Source
              </h4>
              <ul style={{ paddingLeft: '1.5rem', fontSize: '0.9rem', color: 'var(--slate)', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                <li>PhysioNet/CinC Challenge 2016 — Heart Sound Database</li>
                <li>Curated from multiple clinical sites with annotated cardiac auscultation recordings</li>
                <li>Includes both normal and pathological heart sounds across diverse demographics</li>
              </ul>
            </div>

            <div className="card" style={{ padding: '1.5rem' }}>
              <h4 style={{ color: 'var(--navy)', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Activity size={18} className="text-teal" /> Data Preprocessing
              </h4>
              <ul style={{ paddingLeft: '1.5rem', fontSize: '0.9rem', color: 'var(--slate)', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                <li>Resampling to 2,000 Hz for computational efficiency</li>
                <li>Bandpass filtering (25–900 Hz) to isolate cardiac frequency bands</li>
                <li>Segmentation into fixed-length windows of 5 seconds with 50% overlap</li>
                <li>Z-score normalization per segment</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer style={{ background: 'var(--navy)', color: 'var(--white)', padding: '4rem 0', marginTop: '4rem' }}>
        <div className="container text-center">
          <Activity color="var(--teal)" size={32} className="mb-2" style={{ margin: '0 auto' }} />
          <h3 style={{ color: 'var(--white)' }} className="mb-1">CardioSonic</h3>
          <p style={{ color: 'var(--slate)', fontSize: '0.9rem', maxWidth: '500px', margin: '0 auto' }}>
            A research prototype clinical screening AI. Not intended for diagnostic use without physician verification.
          </p>
        </div>
      </footer>

      <style dangerouslySetInnerHTML={{
        __html: `
        @keyframes spin { 100% { transform: rotate(360deg); } }
      `}} />
    </div>
  );
}

export default App;
