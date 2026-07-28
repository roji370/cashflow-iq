import { useState } from 'react'
import './App.css'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

function App() {
  const [customerId, setCustomerId] = useState('cust_001')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const fetchScore = async () => {
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const res = await fetch(
        `${API_BASE_URL}/score/${customerId}?product=home_loan`
      )
      if (!res.ok) {
        const data = await res.json()
        throw new Error(data.detail || `HTTP ${res.status}`)
      }
      const data = await res.json()
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div id="app-root" style={{ fontFamily: 'system-ui, sans-serif', maxWidth: 600, margin: '40px auto', padding: '0 20px' }}>
      <h1>Cashflow IQ — Scoring Dashboard</h1>
      <p style={{ color: '#888' }}>Phase A Walking Skeleton</p>

      <div id="search-section" style={{ marginBottom: 24 }}>
        <label htmlFor="customer-id-input" style={{ fontWeight: 'bold' }}>Customer ID: </label>
        <input
          id="customer-id-input"
          type="text"
          value={customerId}
          onChange={(e) => setCustomerId(e.target.value)}
          style={{ padding: '8px 12px', fontSize: 16, marginRight: 8, border: '1px solid #ccc', borderRadius: 4 }}
        />
        <button
          id="fetch-score-btn"
          onClick={fetchScore}
          disabled={loading || !customerId.trim()}
          style={{ padding: '8px 16px', fontSize: 16, cursor: 'pointer' }}
        >
          {loading ? 'Loading...' : 'Get Score'}
        </button>
      </div>

      {error && (
        <div id="error-message" style={{ padding: 16, background: '#fee', border: '1px solid #fcc', borderRadius: 4, color: '#c00', marginBottom: 16 }}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {result && (
        <div id="score-results">
          <h2>Results for {result.customer_id}</h2>

          <div id="capacity-section" style={{ background: '#f0f8ff', padding: 16, borderRadius: 8, marginBottom: 16 }}>
            <h3>Capacity Score</h3>
            <table style={{ width: '100%' }}>
              <tbody>
                <tr>
                  <td style={{ fontWeight: 'bold' }}>Estimated Income</td>
                  <td id="estimated-income">₹{result.capacity.estimated_income.toLocaleString()}</td>
                </tr>
                <tr>
                  <td style={{ fontWeight: 'bold' }}>Confidence</td>
                  <td id="confidence">{(result.capacity.confidence * 100).toFixed(0)}%</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div id="intent-section" style={{ background: '#f0fff0', padding: 16, borderRadius: 8 }}>
            <h3>Intent Score</h3>
            <table style={{ width: '100%' }}>
              <tbody>
                <tr>
                  <td style={{ fontWeight: 'bold' }}>Product</td>
                  <td id="product">{result.intent.product}</td>
                </tr>
                <tr>
                  <td style={{ fontWeight: 'bold' }}>Intent Score</td>
                  <td id="intent-score">{result.intent.intent_score}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
