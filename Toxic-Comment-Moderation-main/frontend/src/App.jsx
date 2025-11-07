import { useState, useEffect } from 'react'
import CommentInput from './components/CommentInput'
import ResultsPanel from './components/ResultsPanel'
import ExampleButtons from './components/ExampleButtons'
import './styles/App.css'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const REQUEST_TIMEOUT = 30000 // 30 seconds

function App() {
  const [inputText, setInputText] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [backendConnected, setBackendConnected] = useState(false)
  const [checkingBackend, setCheckingBackend] = useState(true)

  // Check backend health on component mount
  useEffect(() => {
    checkBackendHealth()
  }, [])

  const checkBackendHealth = async () => {
    try {
      setCheckingBackend(true)
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), 5000) // 5 second timeout for health check

      const response = await fetch(`${API_BASE_URL}/health`, {
        signal: controller.signal,
      })

      clearTimeout(timeoutId)

      if (response.ok) {
        setBackendConnected(true)
        setError(null)
      } else {
        setBackendConnected(false)
        setError('Backend is not responding correctly. Please ensure the server is running.')
      }
    } catch (err) {
      setBackendConnected(false)
      if (err.name === 'AbortError') {
        setError('Backend connection timeout. Please check if the server is running on port 8000.')
      } else {
        setError('Cannot connect to backend. Please ensure the server is running on port 8000.')
      }
    } finally {
      setCheckingBackend(false)
    }
  }

  const handleModerate = async () => {
    if (!inputText.trim()) {
      setError('Please enter some text to moderate')
      return
    }

    if (!backendConnected) {
      setError('Backend is not connected. Please check if the server is running.')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT)

      const response = await fetch(`${API_BASE_URL}/moderate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: inputText }),
        signal: controller.signal,
      })

      clearTimeout(timeoutId)

      if (!response.ok) {
        let errorMessage = 'Failed to moderate text'
        try {
          const errorData = await response.json()
          errorMessage = errorData.detail || errorMessage
        } catch {
          // If response is not JSON, use status text
          errorMessage = `Server error: ${response.status} ${response.statusText}`
        }
        throw new Error(errorMessage)
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      if (err.name === 'AbortError') {
        setError('Request timed out. The server is taking too long to respond. Please try again.')
      } else if (err.message.includes('Failed to fetch') || err.message.includes('NetworkError')) {
        setError('Network error: Cannot connect to backend. Please ensure the server is running on port 8000.')
        setBackendConnected(false)
        // Retry health check
        checkBackendHealth()
      } else {
        setError(err.message || 'An error occurred. Please try again.')
      }
      console.error('Moderation error:', err)
    } finally {
      setLoading(false)
    }
  }

  const handleExampleClick = (exampleText) => {
    setInputText(exampleText)
    setResult(null)
    setError(null)
  }

  const handleClear = () => {
    setInputText('')
    setResult(null)
    setError(null)
  }

  return (
    <div className="app">
      <div className="container">
        <header className="header">
          <h1>Toxic Comment Moderation</h1>
          <p className="subtitle">
            AI-powered content moderation using DistilBERT
          </p>
          {!checkingBackend && (
            <div className={`connection-status ${backendConnected ? 'connected' : 'disconnected'}`}>
              {backendConnected ? '✓ Backend Connected' : '✗ Backend Disconnected'}
              {!backendConnected && (
                <button onClick={checkBackendHealth} className="retry-btn">
                  Retry Connection
                </button>
              )}
            </div>
          )}
        </header>

        <div className="main-content">
          <div className="input-section">
            <CommentInput
              value={inputText}
              onChange={setInputText}
              onModerate={handleModerate}
              onClear={handleClear}
              loading={loading}
              disabled={!backendConnected}
            />

            <ExampleButtons onExampleClick={handleExampleClick} />

            {error && (
              <div className="error-message" role="alert">
                {error}
              </div>
            )}
          </div>

          {result && (
            <ResultsPanel result={result} />
          )}

          {loading && (
            <div className="loading-container">
              <div className="spinner"></div>
              <p>Analyzing...</p>
            </div>
          )}
        </div>

        <footer className="footer">
          <p className="threshold-note">
            Note: This threshold prioritizes precision; lowering it increases recall.
          </p>
        </footer>
      </div>
    </div>
  )
}

export default App

