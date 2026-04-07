import React, { useState } from 'react';
import axios from 'axios';
import { Loader2, Search } from 'lucide-react';
import ExplainabilityViewer from './components/ExplainabilityViewer';

const api = axios.create({
  baseURL: 'http://localhost:8000',
});

function App() {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [explanations, setExplanations] = useState(null);

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
            onClick={() =>
              setText(
                'A framework using neural networks to optimize data routing in distributed operating systems, specifically designed to mitigate unauthorized access and severe encryption attacks.'
              )
            }
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
          {explanations.map((exp, idx) => (
            <ExplainabilityViewer key={idx} text={text} explanation={exp} />
          ))}
        </div>
      )}
    </div>
  );
}

export default App;
