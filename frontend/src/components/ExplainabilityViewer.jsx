import React, { useState, useMemo } from 'react';

const HighlightedText = ({ text, limeWords, shapWords, attentionWords, showLime, showShap, showAttention }) => {
  // Split the text into parts (words and punctuation)
  const tokens = useMemo(() => {
    // Regex splits by word boundary but preserves the delimiter so we don't lose punctuation/spaces
    return text.split(/(\b[a-zA-Z0-9_]+\b)/g);
  }, [text]);

  const limeSet = new Set(limeWords.map(w => w.toLowerCase()));
  const shapSet = new Set(shapWords.map(w => w.toLowerCase()));
  const attSet = new Set(attentionWords.map(w => w.toLowerCase()));

  return (
    <div className="text-content">
      {tokens.map((token, i) => {
        // Only process actual words
        if (!/^[a-zA-Z0-9_]+$/.test(token)) {
          return <span key={i}>{token}</span>;
        }

        const lowerToken = token.toLowerCase();
        const hasLime = showLime && limeSet.has(lowerToken);
        const hasShap = showShap && shapSet.has(lowerToken);
        const hasAtt = showAttention && attSet.has(lowerToken);

        // Calculate background based on overlaps
        let background = 'transparent';
        if (hasLime && hasShap && hasAtt) {
          background = 'linear-gradient(45deg, var(--lime-highlight), var(--shap-highlight), var(--attention-highlight))';
        } else if (hasLime && hasShap) {
          background = 'linear-gradient(45deg, var(--lime-highlight), var(--shap-highlight))';
        } else if (hasLime && hasAtt) {
          background = 'linear-gradient(45deg, var(--lime-highlight), var(--attention-highlight))';
        } else if (hasShap && hasAtt) {
          background = 'linear-gradient(45deg, var(--shap-highlight), var(--attention-highlight))';
        } else if (hasLime) {
          background = 'var(--lime-highlight)';
        } else if (hasShap) {
          background = 'var(--shap-highlight)';
        } else if (hasAtt) {
          background = 'var(--attention-highlight)';
        }

        const spanStyle = {
          background: background,
          borderRadius: '2px',
          padding: background !== 'transparent' ? '0 2px' : '0',
          transition: 'background 0.3s ease'
        };

        return (
          <span key={i} style={spanStyle} className="highlight">
            {token}
          </span>
        );
      })}
    </div>
  );
};

const ExplainabilityViewer = ({ text, explanation }) => {
  const [showLime, setShowLime] = useState(true);
  const [showShap, setShowShap] = useState(true);
  const [showAttention, setShowAttention] = useState(true);

  if (!explanation) return null;

  const { lime_words, shap_words, attention_words } = explanation.why;

  return (
    <div className="glass-panel explain-viewer">
      <div className="explain-header">
        <h2>Why <span style={{ color: '#3b82f6' }}>{explanation.label}</span>?</h2>
        <div className="toggles">
          <button 
            className={`toggle-btn lime ${showLime ? 'active' : ''}`}
            onClick={() => setShowLime(!showLime)}
          >
            LIME
          </button>
          <button 
            className={`toggle-btn shap ${showShap ? 'active' : ''}`}
            onClick={() => setShowShap(!showShap)}
          >
            SHAP
          </button>
          <button 
            className={`toggle-btn attention ${showAttention ? 'active' : ''}`}
            onClick={() => setShowAttention(!showAttention)}
          >
            ATTENTION
          </button>
        </div>
      </div>
      
      <HighlightedText 
        text={text}
        limeWords={lime_words}
        shapWords={shap_words}
        attentionWords={attention_words}
        showLime={showLime}
        showShap={showShap}
        showAttention={showAttention}
      />
    </div>
  );
};

export default ExplainabilityViewer;
