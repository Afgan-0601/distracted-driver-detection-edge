import React, { useState, useEffect } from 'react'
import './App.css'

const API_BASE = import.meta.env.VITE_API_BASE_URL || ''
const VIDEO_MIN_SECONDS = 5
const VIDEO_MAX_SECONDS = 10

const CLASS_LABELS = {
  c0: 'Safe - Both Hands on Wheel',
  c1: 'Texting - Right Hand',
  c2: 'Talking on Phone - Right',
  c3: 'Texting - Left Hand',
  c4: 'Talking on Phone - Left',
  c5: 'Operating Radio',
  c6: 'Drinking',
  c7: 'Reaching Behind',
  c8: 'Hair and Makeup',
  c9: 'Talking to Passenger'
}

function getRiskColor(level) {
  const colors = {
    'SAFE': '#10b981',
    'LOW': '#f59e0b',
    'MEDIUM': '#f97316',
    'HIGH': '#ef4444',
    'CRITICAL': '#7c2d12'
  }
  return colors[level] || '#6b7280'
}

function calculateSafetyGrade(distractionRate) {
  if (distractionRate <= 10) return 'A+'
  if (distractionRate <= 20) return 'A'
  if (distractionRate <= 30) return 'B+'
  if (distractionRate <= 40) return 'B'
  if (distractionRate <= 50) return 'C'
  return 'D'
}

function getRecommendation(result) {
  if (result.distracted_pct <= 10) {
    return 'Excellent focus! Keep up the safe driving.'
  } else if (result.distracted_pct <= 25) {
    return 'Good focus overall, but avoid brief distractions.'
  } else if (result.distracted_pct <= 40) {
    return 'Watch your attention. Minimize phone use while driving.'
  } else {
    return '⚠️ High distraction detected. Focus on the road.'
  }
}

function getAverageConfidence(predictions) {
  if (!Array.isArray(predictions) || predictions.length === 0) return 0
  const total = predictions.reduce((sum, item) => sum + Number(item?.[1] ?? 0), 0)
  return total / predictions.length
}

const BATCH_PRESETS = [
  {
    id: 'safe_trip',
    label: 'Safe Trip',
    description: '20 frames, fully attentive driver',
    icon: '✅',
    data: [[0,0.97],[0,0.95],[0,0.93],[0,0.96],[0,0.98],[0,0.94],[0,0.92],[0,0.97],[0,0.95],[0,0.96],[0,0.93],[0,0.91],[0,0.97],[0,0.98],[0,0.95],[0,0.94],[0,0.96],[0,0.93],[0,0.97],[0,0.95]],
  },
  {
    id: 'texting',
    label: 'Texting',
    description: '20 frames, sustained right-hand texting',
    icon: '📱',
    data: [[0,0.95],[1,0.88],[1,0.92],[1,0.94],[1,0.91],[1,0.89],[1,0.93],[1,0.90],[1,0.87],[1,0.92],[1,0.95],[1,0.91],[1,0.88],[1,0.93],[1,0.90],[1,0.89],[0,0.72],[0,0.81],[1,0.85],[0,0.91]],
  },
  {
    id: 'phone_call',
    label: 'Phone Call',
    description: '20 frames, phone held to right ear',
    icon: '📞',
    data: [[0,0.96],[2,0.87],[2,0.91],[2,0.93],[2,0.90],[2,0.88],[2,0.92],[2,0.89],[2,0.86],[2,0.91],[2,0.94],[2,0.90],[2,0.87],[2,0.92],[2,0.89],[2,0.88],[0,0.75],[0,0.83],[2,0.84],[0,0.92]],
  },
  {
    id: 'drinking',
    label: 'Drinking',
    description: '15 frames, drinking behavior',
    icon: '🥤',
    data: [[0,0.95],[6,0.86],[6,0.89],[6,0.92],[6,0.88],[6,0.85],[6,0.90],[6,0.87],[6,0.84],[0,0.78],[0,0.85],[6,0.82],[0,0.89],[0,0.93],[0,0.96]],
  },
  {
    id: 'mixed',
    label: 'Mixed',
    description: '30 frames, various distractions',
    icon: '⚠️',
    data: [[0,0.97],[0,0.95],[1,0.88],[1,0.91],[3,0.85],[3,0.89],[0,0.72],[6,0.83],[6,0.87],[0,0.91],[2,0.86],[2,0.90],[0,0.93],[0,0.95],[5,0.78],[5,0.82],[0,0.87],[7,0.75],[7,0.79],[0,0.88],[0,0.92],[9,0.81],[9,0.84],[0,0.90],[1,0.86],[1,0.89],[0,0.93],[0,0.96],[0,0.94],[0,0.97]],
  },
  {
    id: 'critical',
    label: 'High Risk',
    description: '25 frames, escalating sustained distraction',
    icon: '🚨',
    data: [[0,0.94],[1,0.87],[1,0.90],[1,0.92],[1,0.95],[1,0.93],[1,0.91],[1,0.88],[1,0.94],[1,0.96],[3,0.85],[3,0.88],[3,0.91],[3,0.86],[3,0.89],[3,0.92],[3,0.94],[3,0.87],[3,0.90],[3,0.93],[3,0.95],[3,0.88],[3,0.91],[3,0.85],[0,0.71]],
  },
]

function App() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [exampleImages, setExampleImages] = useState([])
  const [selectedExample, setSelectedExample] = useState(null)
  const [activeTab, setActiveTab] = useState('single') // 'single' | 'batch' | 'video'
  const [batchData, setBatchData] = useState('')
  const [selectedPreset, setSelectedPreset] = useState(null)
  const [batchResult, setBatchResult] = useState(null)
  const [batchLoading, setBatchLoading] = useState(false)
  const [batchError, setBatchError] = useState(null)
  const [videoFile, setVideoFile] = useState(null)
  const [videoPreview, setVideoPreview] = useState(null)
  const [videoLoading, setVideoLoading] = useState(false)
  const [videoError, setVideoError] = useState(null)
  const [videoResult, setVideoResult] = useState(null)
  const [videoProgress, setVideoProgress] = useState('')

  // Load example images from public/samples folder
  useEffect(() => {
    const loadExamples = async () => {
      try {
        const response = await fetch('/samples/manifest.json')
        if (response.ok) {
          const data = await response.json()
          setExampleImages(data)
        }
      } catch (err) {
        // Silently fail if no examples available
        console.log('No examples available')
      }
    }
    loadExamples()
  }, [])

  const handleSelectExample = async (imagePath) => {
    try {
      const response = await fetch(imagePath)
      const blob = await response.blob()
      const fakeFile = new File([blob], imagePath.split('/').pop(), { type: blob.type })
      setFile(fakeFile)
      setSelectedExample(imagePath)
      setPreview(imagePath)
      setError(null)
      setResult(null)
    } catch (err) {
      setError('Failed to load example image')
    }
  }

  const handleFileChange = (e) => {
    const selected = e.target.files?.[0]
    if (selected) {
      setFile(selected)
      const reader = new FileReader()
      reader.onload = (evt) => setPreview(evt.target.result)
      reader.readAsDataURL(selected)
      setError(null)
      setResult(null)
      setSelectedExample(null)
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!file) {
      setError('Please select an image')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await fetch(`${API_BASE}/api/v1/predict`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`API Error: ${response.statusText}`)
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleBatchSubmit = async (e) => {
    e.preventDefault()
    if (!batchData.trim()) {
      setBatchError('Please enter batch data')
      return
    }

    setBatchLoading(true)
    setBatchError(null)
    setBatchResult(null)

    try {
      let payload = {}
      try {
        const parsed = JSON.parse(batchData)
        
        // Handle different input formats
        if (Array.isArray(parsed) && parsed.length > 0) {
          if (parsed[0].class_id !== undefined) {
            // Format: [{"class_id": 0, "confidence": 0.95}, ...]
            payload.predictions = parsed.map(p => [p.class_id, p.confidence])
          } else if (Array.isArray(parsed[0])) {
            // Format: [[0, 0.95], [1, 0.87], ...]
            payload.predictions = parsed
          } else {
            setBatchError('Invalid format. Use either [{class_id, confidence}...] or [[0, 0.95], ...]')
            setBatchLoading(false)
            return
          }
        } else {
          setBatchError('Data must be a non-empty array')
          setBatchLoading(false)
          return
        }
      } catch (parseErr) {
        setBatchError(`Invalid JSON: ${parseErr.message}`)
        setBatchLoading(false)
        return
      }

      const response = await fetch(`${API_BASE}/api/v1/predict/batch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}))
        throw new Error(errData.detail || `API Error: ${response.statusText}`)
      }

      const data = await response.json()
      setBatchResult({
        ...data,
        avg_confidence: getAverageConfidence(payload.predictions),
      })
    } catch (err) {
      setBatchError(err.message)
    } finally {
      setBatchLoading(false)
    }
  }

  const handleReset = () => {
    setFile(null)
    setPreview(null)
    setResult(null)
    setError(null)
  }

  const handleSelectPreset = (preset) => {
    setSelectedPreset(preset.id)
    setBatchData(JSON.stringify(preset.data, null, 2))
    setBatchResult(null)
    setBatchError(null)
  }

  const handleClearBatch = () => {
    setBatchData('')
    setSelectedPreset(null)
    setBatchResult(null)
    setBatchError(null)
  }

  const handleVideoChange = (e) => {
    const selected = e.target.files?.[0]
    if (!selected) return

    if (videoPreview) {
      URL.revokeObjectURL(videoPreview)
    }

    const objectUrl = URL.createObjectURL(selected)
    setVideoFile(selected)
    setVideoPreview(objectUrl)
    setVideoError(null)
    setVideoResult(null)
    setVideoProgress('')
  }

  const handleVideoReset = () => {
    if (videoPreview) {
      URL.revokeObjectURL(videoPreview)
    }
    setVideoFile(null)
    setVideoPreview(null)
    setVideoError(null)
    setVideoResult(null)
    setVideoProgress('')
  }

  const extractFramesFromVideo = async (file, frameCount) => {
    const videoUrl = URL.createObjectURL(file)
    const video = document.createElement('video')
    const canvas = document.createElement('canvas')
    const ctx = canvas.getContext('2d')

    video.src = videoUrl
    video.crossOrigin = 'anonymous'
    video.muted = true
    video.playsInline = true

    await new Promise((resolve, reject) => {
      video.onloadedmetadata = resolve
      video.onerror = () => reject(new Error('Unable to load video metadata'))
    })

    const duration = video.duration
    if (duration < VIDEO_MIN_SECONDS || duration > VIDEO_MAX_SECONDS) {
      URL.revokeObjectURL(videoUrl)
      throw new Error(`Video duration must be ${VIDEO_MIN_SECONDS}-${VIDEO_MAX_SECONDS} seconds. Uploaded: ${duration.toFixed(1)}s`)
    }

    canvas.width = video.videoWidth
    canvas.height = video.videoHeight

    const frames = []
    const safeFrameCount = Math.max(10, Math.min(30, frameCount))

    for (let i = 0; i < safeFrameCount; i += 1) {
      const time = (i / safeFrameCount) * Math.max(0.1, duration - 0.1)
      setVideoProgress(`Extracting frames ${i + 1}/${safeFrameCount}`)

      await new Promise((resolve, reject) => {
        const onSeeked = () => {
          video.removeEventListener('seeked', onSeeked)
          resolve()
        }
        video.addEventListener('seeked', onSeeked)
        video.currentTime = time
        video.onerror = () => reject(new Error('Failed to seek video frame'))
      })

      ctx.drawImage(video, 0, 0, canvas.width, canvas.height)

      const blob = await new Promise((resolve) => {
        canvas.toBlob((b) => resolve(b), 'image/jpeg', 0.9)
      })

      if (blob) {
        frames.push(blob)
      }
    }

    URL.revokeObjectURL(videoUrl)
    return { frames, duration }
  }

  const handleVideoSubmit = async (e) => {
    e.preventDefault()
    if (!videoFile) {
      setVideoError('Please select a video file')
      return
    }

    setVideoLoading(true)
    setVideoError(null)
    setVideoResult(null)

    try {
      const { frames, duration } = await extractFramesFromVideo(videoFile, Math.round(videoFile.size > 10_000_000 ? 16 : 20))

      if (!frames.length) {
        throw new Error('No frames could be extracted from this video')
      }

      // Reset sustained-state timer for a clean session before frame-by-frame inference.
      await fetch(`${API_BASE}/api/v1/session/reset`, { method: 'POST' }).catch(() => null)

      const predictions = []

      for (let i = 0; i < frames.length; i += 1) {
        setVideoProgress(`Analyzing frames ${i + 1}/${frames.length}`)

        const frameBlob = frames[i]
        const formData = new FormData()
        formData.append('file', frameBlob, `frame_${i + 1}.jpg`)

        const response = await fetch(`${API_BASE}/api/v1/predict`, {
          method: 'POST',
          body: formData,
        })

        if (!response.ok) {
          throw new Error(`Frame ${i + 1} failed: ${response.statusText}`)
        }

        const data = await response.json()
        predictions.push([data.class_id, data.confidence])
      }

      setVideoProgress('Building trip summary...')
      const batchResponse = await fetch(`${API_BASE}/api/v1/predict/batch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ predictions }),
      })

      if (!batchResponse.ok) {
        const errData = await batchResponse.json().catch(() => ({}))
        throw new Error(errData.detail || `API Error: ${batchResponse.statusText}`)
      }

      const summary = await batchResponse.json()
      setVideoResult({
        ...summary,
        source_duration_seconds: duration,
        avg_confidence: getAverageConfidence(predictions),
      })
      setVideoProgress('Done')
    } catch (err) {
      setVideoError(err.message)
      setVideoProgress('')
    } finally {
      setVideoLoading(false)
    }
  }

  return (
    <div className="container">
      <div className="card">
        <h1>🚗 Distracted Driver Detection</h1>
        <p className="subtitle">Upload a driver image to analyze attention level</p>

        {/* Tabs */}
        <div className="tabs">
          <button 
            className={`tab ${activeTab === 'single' ? 'active' : ''}`}
            onClick={() => setActiveTab('single')}
          >
            📸 Single Image
          </button>
          <button 
            className={`tab ${activeTab === 'batch' ? 'active' : ''}`}
            onClick={() => setActiveTab('batch')}
          >
            📊 Batch Analysis
          </button>
          <button
            className={`tab ${activeTab === 'video' ? 'active' : ''}`}
            onClick={() => setActiveTab('video')}
          >
            🎬 Video (5-10s)
          </button>
        </div>

        {/* Single Image Mode */}
        {activeTab === 'single' && (
          <>
            {exampleImages.length > 0 && (
          <div className="examples-section">
            <h3>📸 Example Images</h3>
            <div className="examples-gallery">
              {exampleImages.map((img) => (
                <div 
                  key={img.id}
                  className={`example-card ${selectedExample === img.path ? 'selected' : ''}`}
                  onClick={() => handleSelectExample(img.path)}
                >
                  <img src={img.thumbnail} alt={img.label} className="example-thumb" />
                  <div className="example-label">{img.label}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        <form onSubmit={handleSubmit} className="form">
          <div className="upload-area">
            <input
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              id="file-input"
              className="file-input"
              disabled={loading}
            />
            <label htmlFor="file-input" className="upload-label">
              {preview ? '✓ Image Selected' : '📁 Click to Upload Image'}
            </label>
          </div>

          {preview && (
            <div className="preview-section">
              <img src={preview} alt="Preview" className="preview-image" />
            </div>
          )}

          <div className="button-group">
            <button 
              type="submit" 
              className="btn btn-primary"
              disabled={!file || loading}
            >
              {loading ? 'Analyzing...' : 'Analyze Image'}
            </button>
            {(file || result) && (
              <button 
                type="button" 
                className="btn btn-secondary"
                onClick={handleReset}
                disabled={loading}
              >
                Reset
              </button>
            )}
          </div>
        </form>

        {error && (
          <div className="error-box">
            <strong>❌ Error:</strong> {error}
          </div>
        )}

        {result && (
          <div className="result-section">
            <h2>📊 Analysis Result</h2>

            {/* Risk Legend */}
            <div className="risk-legend">
              <h4>Risk Levels</h4>
              <div className="legend-items">
                <div className="legend-item"><span className="legend-badge safe">SAFE</span> 0-20</div>
                <div className="legend-item"><span className="legend-badge low">LOW</span> 20-45</div>
                <div className="legend-item"><span className="legend-badge medium">MEDIUM</span> 45-70</div>
                <div className="legend-item"><span className="legend-badge high">HIGH</span> 70-85 ⚠️</div>
                <div className="legend-item"><span className="legend-badge critical">CRITICAL</span> 85-100 🚨</div>
              </div>
            </div>

            {/* Classification */}
            <div className="result-card">
              <h3>Classification</h3>
              <div className="classification">
                <div className="class-label">
                  <span className="label-key">{result.class_key}</span>
                  <span className="label-name">{CLASS_LABELS[result.class_key]}</span>
                </div>
                <div className="confidence-bar">
                  <div
                    className="confidence-fill"
                    style={{ width: `${result.confidence * 100}%` }}
                  />
                </div>
                <span className="confidence-text">
                  Confidence: {(result.confidence * 100).toFixed(1)}%
                </span>
              </div>
            </div>

            {/* Risk Assessment */}
            <div className="result-card">
              <h3>⚠️ Risk Assessment</h3>
              <div className="risk-box">
                <div className="risk-level-badge" style={{ 
                  backgroundColor: getRiskColor(result.risk.risk_level),
                  color: '#fff'
                }}>
                  {result.risk.risk_level}
                </div>
                <div className="risk-details">
                  <div className="risk-item">
                    <span className="risk-label">Composite Risk Score:</span>
                    <span className="risk-value">{result.risk.composite_risk.toFixed(2)}/100</span>
                  </div>
                  <div className="risk-item">
                    <span className="risk-label">Base Risk (Behavior):</span>
                    <span className="risk-value">{result.risk.base_risk}</span>
                  </div>
                  <div className="risk-item">
                    <span className="risk-label">Duration of Distraction:</span>
                    <span className="risk-value">{result.risk.sustained_seconds.toFixed(1)}s</span>
                  </div>
                  <div className="risk-item">
                    <span className="risk-label">Time Penalty Multiplier:</span>
                    <span className="risk-value">{result.risk.sustained_multiplier.toFixed(2)}x</span>
                  </div>
                  <div className="risk-item">
                    <span className="risk-label">Alert (≥ HIGH level):</span>
                    <span className={result.risk.alert ? 'alert-yes' : 'alert-no'}>
                      {result.risk.alert ? '🔴 YES' : '✅ NO'}
                    </span>
                  </div>
                </div>
              </div>
              <div className="risk-explanation">
                <p><strong>How it works:</strong> Composite Risk = Base Risk × Confidence × Time Multiplier</p>
                <p>The longer a distraction persists, the multiplier increases: 1.0x (0s) → 1.2x (2s) → 1.5x (5s) → 2.0x (10s) → 2.5x (20s+)</p>
              </div>
            </div>

            {/* Confidence Scores */}
            <div className="result-card">
              <h3>📈 All Class Scores</h3>
              <div className="scores-grid">
                {result.all_scores.map((score, idx) => {
                  const classKey = `c${idx}`
                  const isTop = idx === result.class_id
                  return (
                    <div 
                      key={idx} 
                      className={`score-bar ${isTop ? 'top-prediction' : ''}`}
                    >
                      <div className="score-name">{classKey}</div>
                      <div className="score-visualization">
                        <div 
                          className="score-fill"
                          style={{ width: `${score * 100}%` }}
                        />
                      </div>
                      <div className="score-percent">
                        {(score * 100).toFixed(1)}%
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>

            {/* JSON Response */}
            <details className="json-response">
              <summary>📝 Raw JSON Response</summary>
              <pre>{JSON.stringify(result, null, 2)}</pre>
            </details>
          </div>
        )}
          </>
        )}

        {/* Batch Mode */}
        {activeTab === 'batch' && (
          <>
            <div className="batch-presets-section">
              <h3>⚡ Quick Presets</h3>
              <div className="batch-presets-grid">
                {BATCH_PRESETS.map((preset) => (
                  <div
                    key={preset.id}
                    className={`batch-preset-card ${selectedPreset === preset.id ? 'selected' : ''}`}
                    onClick={() => handleSelectPreset(preset)}
                  >
                    <div className="preset-icon">{preset.icon}</div>
                    <div className="preset-label">{preset.label}</div>
                    <div className="preset-desc">{preset.description}</div>
                  </div>
                ))}
              </div>
            </div>

            <form onSubmit={handleBatchSubmit} className="form">
              <div className="batch-section">
                <h3>📋 Batch Predictions</h3>
                <p className="batch-info">
                  Select a preset above, or paste JSON — Format 1: <code>[{"{"}class_id: 0, confidence: 0.95{"}"}]</code> / Format 2: <code>[[0, 0.95]]</code>
                </p>
                <textarea
                  value={batchData}
                  onChange={(e) => { setBatchData(e.target.value); setSelectedPreset(null) }}
                  placeholder='Format 1: [{"class_id": 0, "confidence": 0.95}, {"class_id": 1, "confidence": 0.87}]
Format 2: [[0, 0.95], [1, 0.87], [3, 0.78]]'
                  className="batch-textarea"
                  disabled={batchLoading}
                  rows="8"
                />
                <div className="button-group">
                  <button
                    type="submit"
                    className="btn btn-primary"
                    disabled={!batchData.trim() || batchLoading}
                  >
                    {batchLoading ? 'Analyzing...' : 'Analyze Batch'}
                  </button>
                  {batchData && (
                    <button
                      type="button"
                      className="btn btn-secondary"
                      onClick={handleClearBatch}
                      disabled={batchLoading}
                    >
                      Clear
                    </button>
                  )}
                </div>
              </div>
            </form>

            {batchError && (
              <div className="error-box">
                <strong>❌ Error:</strong> {batchError}
              </div>
            )}

            {batchResult && (
              <div className="result-section">
                <h2>� Trip Report</h2>

                {/* Trip Scorecard */}
                <div className="trip-scorecard">
                  <div className="score-display">
                    <div className="safety-grade">{calculateSafetyGrade(batchResult.distracted_pct)}</div>
                    <div className="grade-label">Safety Score</div>
                  </div>
                  <div className="trip-summary-text">
                    <p><strong>Trip Duration:</strong> {batchResult.total_frames} frames (~{(batchResult.total_frames * 0.1).toFixed(0)} seconds)</p>
                    <p><strong>Driver Status:</strong> {batchResult.distracted_pct > 40 ? '⚠️ High Risk' : batchResult.distracted_pct > 20 ? '🟡 Medium Safety' : '✅ Safe'}</p>
                    <p><strong>Recommendation:</strong> {getRecommendation(batchResult)}</p>
                  </div>
                </div>

                <div className="result-card">
                  <h3>📈 Trip Statistics</h3>
                  <div className="batch-stats">
                    <div className="stat-item">
                      <span className="stat-label">Total Frames:</span>
                      <span className="stat-value">{batchResult.total_frames}</span>
                    </div>
                    <div className="stat-item">
                      <span className="stat-label">Distracted Frames:</span>
                      <span className="stat-value">{batchResult.distracted_frames}</span>
                    </div>
                    <div className="stat-item">
                      <span className="stat-label">Distraction Rate:</span>
                      <span className="stat-value">{batchResult.distracted_pct.toFixed(1)}%</span>
                    </div>
                    <div className="stat-item">
                      <span className="stat-label">Alert Frames:</span>
                      <span className="stat-value">{batchResult.alert_frames}</span>
                    </div>
                    <div className="stat-item">
                      <span className="stat-label">Max Risk:</span>
                      <span className="stat-value">{batchResult.max_composite_risk.toFixed(2)}</span>
                    </div>
                    <div className="stat-item">
                      <span className="stat-label">Avg Risk:</span>
                      <span className="stat-value">{batchResult.mean_composite_risk.toFixed(2)}</span>
                    </div>
                    <div className="stat-item">
                      <span className="stat-label">Avg Confidence:</span>
                      <span className="stat-value">{((batchResult.avg_confidence || 0) * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                </div>

                {/* Safety Timeline */}
                <div className="result-card">
                  <h3>📊 Safety Timeline</h3>
                  <div className="safety-timeline">
                    {batchResult.time_per_level && Object.entries(batchResult.time_per_level)
                      .sort(([levelA], [levelB]) => {
                        const order = { 'SAFE': 0, 'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'CRITICAL': 4 }
                        return order[levelA] - order[levelB]
                      })
                      .map(([level, pct]) => (
                        <div key={level} className="timeline-item">
                          <div className="timeline-bar-container">
                            <div className="timeline-bar-label">{level}</div>
                            <div className="timeline-bar-bg">
                              <div 
                                className="timeline-bar-fill"
                                style={{ 
                                  width: `${pct}%`,
                                  backgroundColor: getRiskColor(level)
                                }}
                              />
                            </div>
                            <div className="timeline-bar-value">{pct.toFixed(1)}%</div>
                          </div>
                        </div>
                      ))
                    }
                  </div>
                </div>

                {batchResult.time_per_level && Object.keys(batchResult.time_per_level).length > 0 && (
                  <div className="result-card">
                    <h3>⏱️ Frames per Risk Level</h3>
                    <div className="time-per-level">
                      {Object.entries(batchResult.time_per_level).map(([level, pct]) => (
                        <div key={level} className="level-item">
                          <div className="level-badge" style={{ backgroundColor: getRiskColor(level) }}>
                            {level}
                          </div>
                          <div className="level-time">{pct.toFixed(1)}%</div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                <details className="json-response">
                  <summary>📝 Raw Batch Response</summary>
                  <pre>{JSON.stringify(batchResult, null, 2)}</pre>
                </details>
              </div>
            )}
          </>
        )}

        {/* Video Mode */}
        {activeTab === 'video' && (
          <>
            <form onSubmit={handleVideoSubmit} className="form">
              <div className="batch-section">
                <h3>🎬 Upload Video Clip</h3>
                <p className="batch-info">
                  Upload a short clip ({VIDEO_MIN_SECONDS}-{VIDEO_MAX_SECONDS} seconds). Frames are extracted and analyzed by the backend model, then summarized as a trip report.
                </p>

                <div className="upload-area">
                  <input
                    type="file"
                    accept="video/*"
                    onChange={handleVideoChange}
                    id="video-input"
                    className="file-input"
                    disabled={videoLoading}
                  />
                  <label htmlFor="video-input" className="upload-label">
                    {videoPreview ? '✓ Video Selected' : '📁 Click to Upload Video'}
                  </label>
                </div>

                {videoPreview && (
                  <div className="preview-section">
                    <video src={videoPreview} controls className="preview-video" />
                  </div>
                )}

                <div className="button-group">
                  <button
                    type="submit"
                    className="btn btn-primary"
                    disabled={!videoFile || videoLoading}
                  >
                    {videoLoading ? 'Analyzing Video...' : 'Analyze Video'}
                  </button>
                  {(videoFile || videoResult) && (
                    <button
                      type="button"
                      className="btn btn-secondary"
                      onClick={handleVideoReset}
                      disabled={videoLoading}
                    >
                      Reset
                    </button>
                  )}
                </div>

                {videoProgress && (
                  <div className="progress-box">
                    <strong>Progress:</strong> {videoProgress}
                  </div>
                )}
              </div>
            </form>

            {videoError && (
              <div className="error-box">
                <strong>❌ Error:</strong> {videoError}
              </div>
            )}

            {videoResult && (
              <div className="result-section">
                <h2>🎬 Video Trip Report</h2>

                <div className="trip-scorecard">
                  <div className="score-display">
                    <div className="safety-grade">{calculateSafetyGrade(videoResult.distracted_pct)}</div>
                    <div className="grade-label">Safety Score</div>
                  </div>
                  <div className="trip-summary-text">
                    <p><strong>Clip Length:</strong> {videoResult.source_duration_seconds.toFixed(1)} seconds</p>
                    <p><strong>Frames Analyzed:</strong> {videoResult.total_frames}</p>
                    <p><strong>Recommendation:</strong> {getRecommendation(videoResult)}</p>
                  </div>
                </div>

                <div className="result-card">
                  <h3>📈 Trip Statistics</h3>
                  <div className="batch-stats">
                    <div className="stat-item">
                      <span className="stat-label">Total Frames:</span>
                      <span className="stat-value">{videoResult.total_frames}</span>
                    </div>
                    <div className="stat-item">
                      <span className="stat-label">Distracted Frames:</span>
                      <span className="stat-value">{videoResult.distracted_frames}</span>
                    </div>
                    <div className="stat-item">
                      <span className="stat-label">Distraction Rate:</span>
                      <span className="stat-value">{videoResult.distracted_pct.toFixed(1)}%</span>
                    </div>
                    <div className="stat-item">
                      <span className="stat-label">Alert Frames:</span>
                      <span className="stat-value">{videoResult.alert_frames}</span>
                    </div>
                    <div className="stat-item">
                      <span className="stat-label">Max Risk:</span>
                      <span className="stat-value">{videoResult.max_composite_risk.toFixed(2)}</span>
                    </div>
                    <div className="stat-item">
                      <span className="stat-label">Avg Risk:</span>
                      <span className="stat-value">{videoResult.mean_composite_risk.toFixed(2)}</span>
                    </div>
                    <div className="stat-item">
                      <span className="stat-label">Avg Confidence:</span>
                      <span className="stat-value">{((videoResult.avg_confidence || 0) * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                </div>

                <div className="result-card">
                  <h3>📊 Safety Timeline</h3>
                  <div className="safety-timeline">
                    {videoResult.time_per_level && Object.entries(videoResult.time_per_level)
                      .sort(([levelA], [levelB]) => {
                        const order = { 'SAFE': 0, 'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'CRITICAL': 4 }
                        return order[levelA] - order[levelB]
                      })
                      .map(([level, pct]) => (
                        <div key={level} className="timeline-item">
                          <div className="timeline-bar-container">
                            <div className="timeline-bar-label">{level}</div>
                            <div className="timeline-bar-bg">
                              <div
                                className="timeline-bar-fill"
                                style={{
                                  width: `${pct}%`,
                                  backgroundColor: getRiskColor(level)
                                }}
                              />
                            </div>
                            <div className="timeline-bar-value">{pct.toFixed(1)}%</div>
                          </div>
                        </div>
                      ))
                    }
                  </div>
                </div>

                {videoResult.time_per_level && Object.keys(videoResult.time_per_level).length > 0 && (
                  <div className="result-card">
                    <h3>⏱️ Frames per Risk Level</h3>
                    <div className="time-per-level">
                      {Object.entries(videoResult.time_per_level).map(([level, pct]) => (
                        <div key={level} className="level-item">
                          <div className="level-badge" style={{ backgroundColor: getRiskColor(level) }}>
                            {level}
                          </div>
                          <div className="level-time">{pct.toFixed(1)}%</div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}

export default App
