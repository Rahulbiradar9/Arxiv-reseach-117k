import React, { useState } from 'react';
import axios from 'axios';
import { Loader2, Search, Zap } from 'lucide-react';
import ExplainabilityViewer from './components/ExplainabilityViewer';

// Setup axios to talk to the local FastAPI backend
const api = axios.create({
  baseURL: 'http://localhost:8000',
});

function App() {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [predictions, setPredictions] = useState(null);
  const [explanations, setExplanations] = useState(null);
  const [activeExplainLabel, setActiveExplainLabel] = useState(null);

  const handleAnalyze = async () => {
    if (!text.trim()) return;
    
    setLoading(true);
    setPredictions(null);
    setExplanations(null);
    setActiveExplainLabel(null);
    
    try {
      // Fetch pure predictions first
      const predRes = await api.post('/predict', { text });
      const preds = predRes.data.predictions;
      setPredictions(preds);
      
      if (preds && preds.length > 0) {
        // Automatically fetch explanations since UI expects it
        const expRes = await api.post('/explain', { text });
        setExplanations(expRes.data.explanations);
        if (expRes.data.explanations.length > 0) {
            setActiveExplainLabel(expRes.data.explanations[0].label);
        }
      }
    } catch (error) {
      console.error("Error communicating with backend:", error);
      alert("Failed to connect to the backend server. Is it running on port 8000?");
    } finally {
      setLoading(false);
    }
  };

  const getActiveExplanation = () => {
    if (!explanations || !activeExplainLabel) return null;
    return explanations.find(e => e.label === activeExplainLabel) || null;
  };

  return (
    <div className="app-container">
      <header>
        <h1>ArXiv X-Ray</h1>
        <p className="subtitle">Explainable Multi-Label Classification for Research Papers</p>
      </header>

      <div className="glass-panel input-section">
        <textarea
          placeholder="Paste paper abstract or introduction here..."
          value={text}
          onChange={(e) => setText(e.target.value)}
          disabled={loading}
        />
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '1rem' }}>
          <button 
            className="secondary" 
            onClick={() => setText('A framework using neural networks to optimize data routing in distributed operating systems, specifically designed to mitigate unauthorized access and severe encryption attacks.')}
            disabled={loading}
          >
            Load Sample
          </button>
          <button onClick={handleAnalyze} disabled={loading || !text.trim()}>
            {loading ? <Loader2 className="loading-spinner" size={20} /> : <Search size={20} />}
            Analyze Paper
          </button>
        </div>
      </div>

      {predictions && (
        <div className="results-grid">
          <div className="glass-panel predictions-panel">
            <h2>Predicted Tags <Zap size={18} style={{ display: 'inline', color: '#fbbf24', marginLeft: '4px' }} /></h2>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem', marginTop: '1rem' }}>
              {predictions.map((p, idx) => (
                <div 
                  key={idx} 
                  className={`label-card ${activeExplainLabel === p.label ? 'active' : ''}`}
                  onClick={() => setActiveExplainLabel(p.label)}
                  style={{ cursor: 'pointer' }}
                >
                  <span className="label-name">{p.label}</span>
                  <span className="label-conf">{(p.confidence * 100).toFixed(1)}%</span>
                </div>
              ))}
              {predictions.length === 0 && (
                <p style={{ color: '#94a3b8', fontStyle: 'italic' }}>No strong labels detected across thresholds.</p>
              )}
            </div>
          </div>

          <div className="explanation-section">
            {getActiveExplanation() ? (
              <ExplainabilityViewer 
                text={text} 
                explanation={getActiveExplanation()} 
              />
            ) : (
                predictions.length > 0 && <div className="glass-panel" style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#94a3b8' }}>
                    <p>Select a label to view its explanation map.</p>
                </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
