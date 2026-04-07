import React, { useState, useMemo } from 'react';

const getHighlightColor = (hasLime, hasShap, hasAtt) => {
  if (hasLime && hasShap && hasAtt) return 'var(--all-bg)';
  if (hasLime && hasShap) return 'var(--lime-shap-bg)';
  if (hasLime && hasAtt) return 'var(--lime-att-bg)';
  if (hasShap && hasAtt) return 'var(--shap-att-bg)';
  if (hasLime) return 'var(--lime-bg)';
  if (hasShap) return 'var(--shap-bg)';
  if (hasAtt) return 'var(--attention-bg)';
  return 'transparent';
};

const HighlightedText = ({ text, limeWords, shapWords, attentionWords, showLime, showShap, showAttention }) => {
  const tokens = useMemo(() => text.split(/(\b[a-zA-Z0-9_]+\b)/g), [text]);

  const limeSet = new Set(limeWords.map(w => w.toLowerCase()));
  const shapSet = new Set(shapWords.map(w => w.toLowerCase()));
  const attSet  = new Set(attentionWords.map(w => w.toLowerCase()));

  return (
    <div className="text-content">
      {tokens.map((token, i) => {
        if (!/^[a-zA-Z0-9_]+$/.test(token)) {
          return <span key={i}>{token}</span>;
        }

        const lower = token.toLowerCase();
        const hasLime = showLime && limeSet.has(lower);
        const hasShap = showShap && shapSet.has(lower);
        const hasAtt  = showAttention && attSet.has(lower);
        const bg = getHighlightColor(hasLime, hasShap, hasAtt);

        return (
          <span key={i} className="highlight" style={{ background: bg }}>
            {token}
          </span>
        );
      })}
    </div>
  );
};

const ExplainabilityViewer = ({ text, explanations }) => {
  const [showLime, setShowLime] = useState(true);
  const [showShap, setShowShap] = useState(true);
  const [showAttention, setShowAttention] = useState(true);

  if (!explanations || explanations.length === 0) return null;

  // Merge all LIME/SHAP/Attention words across all predicted labels
  const allLime = [];
  const allShap = [];
  const allAttention = [];

  explanations.forEach(exp => {
    allLime.push(...(exp.why.lime_words || []));
    allShap.push(...(exp.why.shap_words || []));
    allAttention.push(...(exp.why.attention_words || []));
  });

  // Deduplicate
  const limeWords = [...new Set(allLime)];
  const shapWords = [...new Set(allShap)];
  const attentionWords = [...new Set(allAttention)];

  return (
    <div className="explain-card">
      <div className="explain-card-header">
        <div className="predicted-labels">
          {explanations.map((exp, i) => (
            <span key={i} className="label-badge">
              {exp.label} <span className="label-conf">{(exp.confidence * 100).toFixed(1)}%</span>
            </span>
          ))}
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
        limeWords={limeWords}
        shapWords={shapWords}
        attentionWords={attentionWords}
        showLime={showLime}
        showShap={showShap}
        showAttention={showAttention}
      />
    </div>
  );
};

export default ExplainabilityViewer;
