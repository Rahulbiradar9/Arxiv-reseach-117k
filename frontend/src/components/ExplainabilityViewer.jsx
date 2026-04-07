import React, { useState, useMemo } from 'react';

const HighlightedText = ({ text, limeWords, shapWords, attentionWords, showLime, showShap, showAttention }) => {
  const tokens = useMemo(() => {
    return text.split(/(\b[a-zA-Z0-9_]+\b)/g);
  }, [text]);

  const limeSet = new Set(limeWords.map(w => w.toLowerCase()));
  const shapSet = new Set(shapWords.map(w => w.toLowerCase()));
  const attSet = new Set(attentionWords.map(w => w.toLowerCase()));

  return (
    <div className="text-content">
      {tokens.map((token, i) => {
        if (!/^[a-zA-Z0-9_]+$/.test(token)) {
          return <span key={i}>{token}</span>;
        }

        const lower = token.toLowerCase();
        const hasLime = showLime && limeSet.has(lower);
        const hasShap = showShap && shapSet.has(lower);
        const hasAtt = showAttention && attSet.has(lower);

        let bg = 'transparent';
        if (hasLime && hasShap && hasAtt) {
          bg = 'linear-gradient(45deg, var(--lime-highlight), var(--shap-highlight), var(--attention-highlight))';
        } else if (hasLime && hasShap) {
          bg = 'linear-gradient(45deg, var(--lime-highlight), var(--shap-highlight))';
        } else if (hasLime && hasAtt) {
          bg = 'linear-gradient(45deg, var(--lime-highlight), var(--attention-highlight))';
        } else if (hasShap && hasAtt) {
          bg = 'linear-gradient(45deg, var(--shap-highlight), var(--attention-highlight))';
        } else if (hasLime) {
          bg = 'var(--lime-highlight)';
        } else if (hasShap) {
          bg = 'var(--shap-highlight)';
        } else if (hasAtt) {
          bg = 'var(--attention-highlight)';
        }

        return (
          <span
            key={i}
            className="highlight"
            style={{
              background: bg,
              padding: bg !== 'transparent' ? '0 2px' : '0',
            }}
          >
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
    <div className="explain-card">
      <div className="explain-card-header">
        <div className="label-info">
          <span className="label-name">{explanation.label}</span>
          <span className="label-conf">{(explanation.confidence * 100).toFixed(1)}%</span>
        </div>
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
            ATT
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
