import React, { useState } from 'react';
import axios from 'axios';
import { Loader2, Search } from 'lucide-react';
import ExplainabilityViewer from './components/ExplainabilityViewer';

const api = axios.create({
  baseURL: 'http://localhost:8000',
});

const SAMPLES = [
  'A framework using neural networks to optimize data routing in distributed operating systems, specifically designed to mitigate unauthorized access and severe encryption attacks.',
  'We propose a novel deep reinforcement learning agent for autonomous navigation using convolutional neural networks trained on simulated environments.',
  'This paper presents a scalable intrusion detection system based on anomaly detection algorithms that monitor network traffic for zero-day exploits and malware signatures.',
  'We introduce a distributed consensus protocol for fault-tolerant cloud computing clusters that achieves sub-millisecond latency under high contention workloads.',
  'Our approach combines transformer-based language models with graph neural networks to perform multi-hop reasoning over large-scale knowledge bases.',
  'A lightweight TCP congestion control algorithm is proposed for 5G mobile edge computing environments with highly variable bandwidth and packet loss.',
  'We present a differential privacy framework that protects user data during federated learning while maintaining model accuracy across heterogeneous devices.',
  'This work explores containerized microservice orchestration using Kubernetes with adaptive autoscaling policies for latency-sensitive serverless applications.',
];

function App() {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [explanations, setExplanations] = useState(null);

  const loadSample = () => {
    const random = SAMPLES[Math.floor(Math.random() * SAMPLES.length)];
    setText(random);
  };

  const handleAnalyze = async () => {
    if (!text.trim()) return;

    setLoading(true);
    setExplanations(null);

    try {
      const res = await api.post('/explain', { text });
      setExplanations(res.data.explanations);
    } catch (error) {
      console.error('Backend error:', error);
      alert('Failed to connect to backend. Is it running on port 8000?');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header>
        <h1>ArXiv X-Ray</h1>
        <p>Explainable Multi-Label Classifier</p>
      </header>

      <section className="info-section">
        <h2>Why did the model choose that label?</h2>
        <p className="info-desc">
          This tool doesn't just classify — it shows <em>which words</em> in
          your text drove each prediction, using three independent explainability
          methods:
        </p>
        <div className="method-cards">
          <div className="method-card">
            <span className="method-dot lime"></span>
            <div>
              <strong>LIME</strong>
              <p>
                Locally Interpretable Model-agnostic Explanations. Perturbs words
                in the input and observes how predictions change to find which
                words matter most <em>for this specific input</em>.
              </p>
            </div>
          </div>
          <div className="method-card">
            <span className="method-dot shap"></span>
            <div>
              <strong>SHAP</strong>
              <p>
                SHapley Additive exPlanations. Uses game theory to assign each
                word a fair contribution score — measuring how much each word
                pushes the prediction up or down.
              </p>
            </div>
          </div>
          <div className="method-card">
            <span className="method-dot attention"></span>
            <div>
              <strong>Attention</strong>
              <p>
                Extracts the model's internal attention weights from the
                transformer layers, revealing which words the model
                "focused on" when making its decision.
              </p>
            </div>
          </div>
        </div>

        <h3 className="legend-title">Color Key</h3>
        <div className="legend">
          <span className="legend-item"><span className="legend-swatch" style={{ background: 'var(--lime-bg)' }}></span>LIME</span>
          <span className="legend-item"><span className="legend-swatch" style={{ background: 'var(--shap-bg)' }}></span>SHAP</span>
          <span className="legend-item"><span className="legend-swatch" style={{ background: 'var(--attention-bg)' }}></span>Attention</span>
          <span className="legend-item"><span className="legend-swatch" style={{ background: 'var(--lime-shap-bg)' }}></span>LIME + SHAP</span>
          <span className="legend-item"><span className="legend-swatch" style={{ background: 'var(--lime-att-bg)' }}></span>LIME + Attention</span>
          <span className="legend-item"><span className="legend-swatch" style={{ background: 'var(--shap-att-bg)' }}></span>SHAP + Attention</span>
          <span className="legend-item"><span className="legend-swatch" style={{ background: 'var(--all-bg)' }}></span>All Three</span>
        </div>
      </section>

      <div className="input-section">
        <textarea
          placeholder="Paste a paper abstract here..."
          value={text}
          onChange={(e) => setText(e.target.value)}
          disabled={loading}
        />
        <div className="actions">
          <button
            className="secondary"
            onClick={loadSample}
            disabled={loading}
          >
            Load Sample
          </button>
          <button onClick={handleAnalyze} disabled={loading || !text.trim()}>
            {loading ? <Loader2 className="loading-spinner" size={16} /> : <Search size={16} />}
            Analyze
          </button>
        </div>
      </div>

      {explanations && (
        <div className="results-section">
          {explanations.length === 0 && (
            <p className="no-results">No labels exceeded the confidence threshold.</p>
          )}
          {explanations.length > 0 && (
            <ExplainabilityViewer text={text} explanations={explanations} />
          )}
        </div>
      )}
    </div>
  );
}

export default App;
