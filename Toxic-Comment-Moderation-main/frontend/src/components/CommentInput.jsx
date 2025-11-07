import React from 'react'

function CommentInput({ value, onChange, onModerate, onClear, loading, disabled }) {
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && e.ctrlKey) {
      onModerate()
    }
  }

  return (
    <div className="comment-input-container">
      <label htmlFor="comment-textarea" className="input-label">
        Enter comment to moderate:
      </label>
      <textarea
        id="comment-textarea"
        className="comment-textarea"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Type or paste a comment here..."
        rows={6}
        disabled={loading || disabled}
      />
      <div className="button-group">
        <button
          className="btn btn-primary"
          onClick={onModerate}
          disabled={loading || !value.trim() || disabled}
        >
          {loading ? 'Moderating...' : 'Moderate'}
        </button>
        <button
          className="btn btn-secondary"
          onClick={onClear}
          disabled={loading || !value.trim() || disabled}
        >
          Clear
        </button>
      </div>
      <small className="hint">Press Ctrl+Enter to submit</small>
    </div>
  )
}

export default CommentInput

