import React from 'react'

function ResultsPanel({ result }) {
  if (!result) {
    return null
  }

  const isToxic = result.label === 'TOXIC'
  const prob = result.prob || 0
  const probPercentage = (prob * 100).toFixed(1)

  return (
    <div className={`results-panel ${isToxic ? 'toxic' : 'non-toxic'}`}>
      <h2 className="results-title">Moderation Results</h2>

      <div className="result-grid">
        <div className="result-item">
          <span className="result-label">Probability:</span>
          <span className="result-value probability">
            {prob.toFixed(3)} ({probPercentage}%)
          </span>
        </div>

        <div className="result-item">
          <span className="result-label">Threshold:</span>
          <span className="result-value threshold">
            {result.threshold?.toFixed(3) || 'N/A'}
          </span>
        </div>

        <div className="result-item full-width">
          <span className="result-label">Prediction:</span>
          <span className={`result-value label ${isToxic ? 'toxic' : 'non-toxic'}`}>
            {result.label || 'N/A'}
          </span>
        </div>
      </div>

      <div className="result-explanation">
        {isToxic ? (
          <p className="explanation-text toxic">
            ⚠️ This comment has been flagged as toxic based on the model's prediction.
            The probability ({prob.toFixed(3)}) exceeds the threshold ({result.threshold?.toFixed(3) || 'N/A'}).
          </p>
        ) : (
          <p className="explanation-text non-toxic">
            ✓ This comment appears to be non-toxic. The probability ({prob.toFixed(3)})
            is below the threshold ({result.threshold?.toFixed(3) || 'N/A'}).
          </p>
        )}
      </div>
    </div>
  )
}

export default ResultsPanel

