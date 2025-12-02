import { useState, useEffect, useCallback, useMemo } from 'react';
import { Play, RotateCcw, Settings, Pause } from 'lucide-react';
import './App.css';

// Constants
const GRID_SIZE = 10;
const PIXEL_SIZE = 30;
const DEFAULT_NOISE_LEVEL = 0.25;
const DEFAULT_MAX_ITERATIONS = 10;
const STEP_DELAY_MS = 500;
const UPDATE_RATIO = 0.5;

// Reference patterns (images to memorize)
const REFERENCE_PATTERNS = {
  square: [
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,1,1,1,1,1,1,0,0],
    [0,0,1,0,0,0,0,1,0,0],
    [0,0,1,0,0,0,0,1,0,0],
    [0,0,1,0,0,0,0,1,0,0],
    [0,0,1,0,0,0,0,1,0,0],
    [0,0,1,1,1,1,1,1,0,0],
    [1,0,0,0,0,0,0,0,0,0],
    [1,1,0,0,0,0,0,0,0,0]
  ],
  diamond: [
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,1,1,0,0,0,0],
    [0,0,0,1,0,0,1,0,0,0],
    [0,0,1,0,0,0,0,1,0,0],
    [0,0,1,0,0,0,0,1,0,0],
    [0,0,0,1,0,0,1,0,0,0],
    [0,0,0,0,1,1,0,0,0,0],
    [1,0,0,0,0,0,0,0,0,0],
    [1,1,0,0,0,0,0,0,0,0]
  ]
};

// Utility functions
const flatten = (matrix) => matrix.flat();

const unflatten = (vector, size) => {
  const matrix = [];
  for (let i = 0; i < size; i++) {
    matrix.push(vector.slice(i * size, (i + 1) * size));
  }
  return matrix;
};

const toBipolar = (vector) => vector.map(v => v === 0 ? -1 : 1);
const fromBipolar = (vector) => vector.map(v => v === -1 ? 0 : 1);

const createNoisyPattern = (pattern, noiseLevel = DEFAULT_NOISE_LEVEL) => {
  return pattern.map(row => 
    row.map(pixel => 
      Math.random() < noiseLevel ? 1 - pixel : pixel
    )
  );
};

const hasConverged = (pattern1, pattern2) => {
  return JSON.stringify(pattern1) === JSON.stringify(pattern2);
};

// Hopfield Network Training Methods
const trainHebb = (patterns, size) => {
  const n = size * size;
  const weights = Array(n).fill(0).map(() => Array(n).fill(0));
  
  patterns.forEach(pattern => {
    const p = toBipolar(flatten(pattern));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (i !== j) {
          weights[i][j] += p[i] * p[j];
        }
      }
    }
  });

  // Normalize
  const numPatterns = patterns.length;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      weights[i][j] /= numPatterns;
    }
  }

  return weights;
};

const trainPseudoinverse = (patterns, size) => {
  const n = size * size;
  const weights = Array(n).fill(0).map(() => Array(n).fill(0));
  
  const bipolarPatterns = patterns.map(p => toBipolar(flatten(p)));
  
  bipolarPatterns.forEach(p => {
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (i !== j) {
          weights[i][j] += p[i] * p[j] / bipolarPatterns.length;
        }
      }
    }
  });

  return weights;
};

// Network update function
const updateNetwork = (pattern, weights, size) => {
  const n = size * size;
  const current = toBipolar(flatten(pattern));
  const next = [...current];

  // Asynchronous update (random neurons per step)
  const indices = Array.from({ length: n }, (_, i) => i);
  const numToUpdate = Math.floor(n * UPDATE_RATIO);

  for (let k = 0; k < numToUpdate; k++) {
    const i = indices.splice(Math.floor(Math.random() * indices.length), 1)[0];
    let sum = 0;
    for (let j = 0; j < n; j++) {
      sum += weights[i][j] * next[j];
    }
    next[i] = sum >= 0 ? 1 : -1;
  }

  return unflatten(fromBipolar(next), size);
};

// Calculate energy function
const calculateEnergy = (pattern, weights) => {
  const p = toBipolar(flatten(pattern));
  const n = p.length;
  let energy = 0;
  
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      energy -= weights[i][j] * p[i] * p[j];
    }
  }
  
  return energy / 2;
};

function App() {
  const [method, setMethod] = useState('hebb');
  const [maxIterations] = useState(DEFAULT_MAX_ITERATIONS);

  // Calculate weights based on method (memoized)
  const weights = useMemo(() => {
    const patterns = [REFERENCE_PATTERNS.square, REFERENCE_PATTERNS.diamond];
    return method === 'hebb' 
      ? trainHebb(patterns, GRID_SIZE) 
      : trainPseudoinverse(patterns, GRID_SIZE);
  }, [method]);

  // Generate initial noisy pattern
  const generateInitialPattern = useCallback(() => {
    return createNoisyPattern(REFERENCE_PATTERNS.diamond, DEFAULT_NOISE_LEVEL);
  }, []);

  const [currentPattern, setCurrentPattern] = useState(generateInitialPattern);
  const [history, setHistory] = useState(() => {
    const initial = createNoisyPattern(REFERENCE_PATTERNS.diamond, DEFAULT_NOISE_LEVEL);
    return [initial];
  });
  const [isRunning, setIsRunning] = useState(false);
  const [iterations, setIterations] = useState(0);

  // Initialize network (for reset button and method change)
  const initializeNetwork = useCallback(() => {
    const noisy = createNoisyPattern(REFERENCE_PATTERNS.diamond, DEFAULT_NOISE_LEVEL);
    setCurrentPattern(noisy);
    setHistory([noisy]);
    setIterations(0);
    setIsRunning(false);
  }, []);

  // Run one step
  const runStep = useCallback(() => {
    if (!weights || iterations >= maxIterations) {
      setIsRunning(false);
      return;
    }

    const newPattern = updateNetwork(currentPattern, weights, GRID_SIZE);
    
    if (hasConverged(currentPattern, newPattern)) {
      setIsRunning(false);
    } else {
      setCurrentPattern(newPattern);
      setHistory(prev => [...prev, newPattern]);
      setIterations(prev => prev + 1);
    }
  }, [weights, currentPattern, iterations, maxIterations]);

  // Auto-run effect
  useEffect(() => {
    if (isRunning) {
      const timer = setTimeout(runStep, STEP_DELAY_MS);
      return () => clearTimeout(timer);
    }
  }, [isRunning, runStep]);

  // Reinitialize when method changes
  useEffect(() => {
    initializeNetwork();
  }, [method, initializeNetwork]);

  // Memoized energy calculation
  const currentEnergy = useMemo(() => {
    return weights ? calculateEnergy(currentPattern, weights).toFixed(2) : 'N/A';
  }, [weights, currentPattern]);

  return (
    <div style={{ 
      width: '100%', 
      minHeight: '100vh', 
      background: 'linear-gradient(135deg, #0f172a 0%, #1e293b 100%)', 
      padding: '24px' 
    }}>
      <div style={{ 
        maxWidth: '1200px', 
        margin: '0 auto', 
        background: '#1e293b', 
        borderRadius: '16px', 
        boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.5), 0 10px 10px -5px rgba(0, 0, 0, 0.04)', 
        padding: '32px',
        border: '1px solid #334155'
      }}>
        <h2 style={{ 
          fontSize: '28px', 
          fontWeight: 'bold', 
          color: '#f1f5f9', 
          marginBottom: '8px',
          textShadow: '0 2px 4px rgba(0,0,0,0.3)'
        }}>
          Hopfield Network - Image Pattern Recognition
        </h2>
        <p style={{ color: '#94a3b8', marginBottom: '28px', fontSize: '15px' }}>
          Pattern recovery simulation with noise reduction (10x10 pixels)
        </p>

        {/* Controls */}
        <div style={{ display: 'flex', gap: '16px', marginBottom: '28px', flexWrap: 'wrap', alignItems: 'center' }}>
          <button
            onClick={() => setIsRunning(!isRunning)}
            disabled={iterations >= maxIterations}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              padding: '10px 20px',
              background: iterations >= maxIterations ? '#475569' : 'linear-gradient(135deg, #3b82f6 0%, #2563eb 100%)',
              color: 'white',
              border: 'none',
              borderRadius: '10px',
              cursor: iterations >= maxIterations ? 'not-allowed' : 'pointer',
              fontSize: '14px',
              fontWeight: '600',
              boxShadow: iterations >= maxIterations ? 'none' : '0 4px 6px rgba(59, 130, 246, 0.3)',
              transition: 'all 0.2s'
            }}
            onMouseEnter={(e) => {
              if (iterations < maxIterations) {
                e.target.style.transform = 'translateY(-2px)';
                e.target.style.boxShadow = '0 6px 12px rgba(59, 130, 246, 0.4)';
              }
            }}
            onMouseLeave={(e) => {
              e.target.style.transform = 'translateY(0)';
              e.target.style.boxShadow = iterations >= maxIterations ? 'none' : '0 4px 6px rgba(59, 130, 246, 0.3)';
            }}
          >
            {isRunning ? <Pause size={20} /> : <Play size={20} />}
            {isRunning ? 'Pause' : 'Run'}
          </button>
          
          <button
            onClick={runStep}
            disabled={isRunning || iterations >= maxIterations}
            style={{
              padding: '10px 20px',
              background: (isRunning || iterations >= maxIterations) ? '#475569' : 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
              color: 'white',
              border: 'none',
              borderRadius: '10px',
              cursor: (isRunning || iterations >= maxIterations) ? 'not-allowed' : 'pointer',
              fontSize: '14px',
              fontWeight: '600',
              boxShadow: (isRunning || iterations >= maxIterations) ? 'none' : '0 4px 6px rgba(16, 185, 129, 0.3)',
              transition: 'all 0.2s'
            }}
            onMouseEnter={(e) => {
              if (!isRunning && iterations < maxIterations) {
                e.target.style.transform = 'translateY(-2px)';
                e.target.style.boxShadow = '0 6px 12px rgba(16, 185, 129, 0.4)';
              }
            }}
            onMouseLeave={(e) => {
              e.target.style.transform = 'translateY(0)';
              e.target.style.boxShadow = (isRunning || iterations >= maxIterations) ? 'none' : '0 4px 6px rgba(16, 185, 129, 0.3)';
            }}
          >
            Step
          </button>
          
          <button
            onClick={initializeNetwork}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              padding: '10px 20px',
              background: 'linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%)',
              color: 'white',
              border: 'none',
              borderRadius: '10px',
              cursor: 'pointer',
              fontSize: '14px',
              fontWeight: '600',
              boxShadow: '0 4px 6px rgba(139, 92, 246, 0.3)',
              transition: 'all 0.2s'
            }}
            onMouseEnter={(e) => {
              e.target.style.transform = 'translateY(-2px)';
              e.target.style.boxShadow = '0 6px 12px rgba(139, 92, 246, 0.4)';
            }}
            onMouseLeave={(e) => {
              e.target.style.transform = 'translateY(0)';
              e.target.style.boxShadow = '0 4px 6px rgba(139, 92, 246, 0.3)';
            }}
          >
            <RotateCcw size={20} />
            Reset
          </button>

          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <Settings size={20} style={{ color: '#94a3b8' }} />
            <select
              value={method}
              onChange={(e) => setMethod(e.target.value)}
              style={{
                padding: '10px 16px',
                background: '#0f172a',
                border: '1px solid #475569',
                borderRadius: '10px',
                fontSize: '14px',
                cursor: 'pointer',
                color: '#f1f5f9',
                fontWeight: '500'
              }}
            >
              <option value="hebb">Hebb Rule</option>
              <option value="pseudoinverse">Pseudoinverse</option>
            </select>
          </div>

          <div style={{ 
            marginLeft: 'auto', 
            fontSize: '18px', 
            fontWeight: '700', 
            color: '#f1f5f9',
            background: '#0f172a',
            padding: '10px 20px',
            borderRadius: '10px',
            border: '1px solid #475569'
          }}>
            Iteration: {iterations} / {maxIterations}
          </div>
        </div>

        {/* Visualization */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '24px', marginBottom: '24px' }}>
          {/* Noisy Pattern */}
          <div style={{ 
            background: '#0f172a', 
            padding: '20px', 
            borderRadius: '12px',
            border: '1px solid #334155',
            boxShadow: '0 4px 6px rgba(0, 0, 0, 0.3)',
            textAlign: 'center'
          }}>
            <h3 style={{ 
              fontWeight: '600', 
              color: '#cbd5e1', 
              marginBottom: '16px', 
              textAlign: 'center',
              fontSize: '16px'
            }}>
              Initial Pattern (with noise)
            </h3>
            <div style={{ 
              display: 'inline-block', 
              border: '2px solid #475569',
              borderRadius: '4px',
              overflow: 'hidden'
            }}>
              {history[0]?.map((row, i) => (
                <div key={i} style={{ display: 'flex' }}>
                  {row.map((pixel, j) => (
                    <div
                      key={j}
                      style={{
                        width: PIXEL_SIZE,
                        height: PIXEL_SIZE,
                        background: pixel ? '#64748b' : '#1e293b'
                      }}
                    />
                  ))}
                </div>
              ))}
            </div>
          </div>

          {/* Current Pattern */}
          <div style={{ 
            background: 'rgba(59, 130, 246, 0.1)', 
            padding: '20px', 
            borderRadius: '12px',
            border: '1px solid rgba(59, 130, 246, 0.3)',
            boxShadow: '0 4px 6px rgba(59, 130, 246, 0.2)',
            textAlign: 'center'
          }}>
            <h3 style={{ 
              fontWeight: '600', 
              color: '#93c5fd', 
              marginBottom: '16px', 
              textAlign: 'center',
              fontSize: '16px'
            }}>
              Current Pattern (Iteration {iterations})
            </h3>
            <div style={{ 
              display: 'inline-block', 
              border: '2px solid #3b82f6',
              borderRadius: '4px',
              overflow: 'hidden',
              boxShadow: '0 0 20px rgba(59, 130, 246, 0.3)'
            }}>
              {currentPattern.map((row, i) => (
                <div key={i} style={{ display: 'flex' }}>
                  {row.map((pixel, j) => (
                    <div
                      key={j}
                      style={{
                        width: PIXEL_SIZE,
                        height: PIXEL_SIZE,
                        background: pixel ? '#3b82f6' : '#0f172a'
                      }}
                    />
                  ))}
                </div>
              ))}
            </div>
          </div>

          {/* Reference Pattern */}
          <div style={{ 
            background: 'rgba(16, 185, 129, 0.1)', 
            padding: '20px', 
            borderRadius: '12px',
            border: '1px solid rgba(16, 185, 129, 0.3)',
            boxShadow: '0 4px 6px rgba(16, 185, 129, 0.2)',
            textAlign: 'center'
          }}>
            <h3 style={{ 
              fontWeight: '600', 
              color: '#6ee7b7', 
              marginBottom: '16px', 
              textAlign: 'center',
              fontSize: '16px'
            }}>
              Reference Pattern (target)
            </h3>
            <div style={{ 
              display: 'inline-block', 
              border: '2px solid #10b981',
              borderRadius: '4px',
              overflow: 'hidden',
              boxShadow: '0 0 20px rgba(16, 185, 129, 0.3)'
            }}>
              {REFERENCE_PATTERNS.diamond.map((row, i) => (
                <div key={i} style={{ display: 'flex' }}>
                  {row.map((pixel, j) => (
                    <div
                      key={j}
                      style={{
                        width: PIXEL_SIZE,
                        height: PIXEL_SIZE,
                        background: pixel ? '#10b981' : '#0f172a'
                      }}
                    />
                  ))}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Additional Information */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '20px' }}>
          <div style={{ 
            background: '#0f172a', 
            padding: '20px', 
            borderRadius: '12px',
            border: '1px solid #334155'
          }}>
            <h4 style={{ fontWeight: '600', color: '#cbd5e1', marginBottom: '12px', fontSize: '16px' }}>
              Model Characteristics
            </h4>
            <ul style={{ fontSize: '14px', color: '#94a3b8', listStyle: 'none', padding: 0, lineHeight: '1.8' }}>
              <li style={{ marginBottom: '6px' }}>• <strong style={{ color: '#cbd5e1' }}>Size:</strong> {GRID_SIZE}x{GRID_SIZE} = {GRID_SIZE*GRID_SIZE} neurons</li>
              <li style={{ marginBottom: '6px' }}>• <strong style={{ color: '#cbd5e1' }}>Method:</strong> {method === 'hebb' ? 'Hebb Rule' : 'Pseudoinverse'}</li>
              <li style={{ marginBottom: '6px' }}>• <strong style={{ color: '#cbd5e1' }}>Memorized patterns:</strong> 2 (square and diamond)</li>
              <li style={{ marginBottom: '6px' }}>• <strong style={{ color: '#cbd5e1' }}>Update:</strong> Asynchronous random</li>
              <li style={{ marginBottom: '6px' }}>• <strong style={{ color: '#cbd5e1' }}>Reference marker:</strong> Corner pixels</li>
            </ul>
          </div>

          <div style={{ 
            background: '#0f172a', 
            padding: '20px', 
            borderRadius: '12px',
            border: '1px solid #334155'
          }}>
            <h4 style={{ fontWeight: '600', color: '#cbd5e1', marginBottom: '12px', fontSize: '16px' }}>
              Process Status
            </h4>
            <ul style={{ fontSize: '14px', color: '#94a3b8', listStyle: 'none', padding: 0, lineHeight: '1.8' }}>
              <li style={{ marginBottom: '6px' }}>
                • <strong style={{ color: '#cbd5e1' }}>Convergence:</strong> {
                  iterations >= maxIterations 
                    ? <span style={{ color: '#f59e0b' }}>Max iterations</span>
                    : isRunning 
                      ? <span style={{ color: '#3b82f6' }}>Running...</span>
                      : <span style={{ color: '#10b981' }}>Stopped</span>
                }
              </li>
              <li style={{ marginBottom: '6px' }}>
                • <strong style={{ color: '#cbd5e1' }}>Energy:</strong> {currentEnergy}
              </li>
              <li style={{ marginBottom: '6px' }}>
                • <strong style={{ color: '#cbd5e1' }}>Initial noise:</strong> ~25% pixels altered
              </li>
            </ul>
          </div>
        </div>

        {/* Legend */}
        <div style={{ 
          marginTop: '24px', 
          padding: '20px', 
          background: 'rgba(251, 191, 36, 0.1)', 
          borderLeft: '4px solid #f59e0b', 
          borderRadius: '8px',
          border: '1px solid rgba(251, 191, 36, 0.2)'
        }}>
          <p style={{ fontSize: '14px', color: '#fbbf24', lineHeight: '1.6' }}>
            <strong style={{ color: '#fcd34d' }}>Note:</strong> The Hopfield network acts as an associative memory, removing noise and 
            recovering the closest memorized pattern. The corner pixels (bottom-left) serve as a 
            fixed reference marker to measure relative displacements of the target pattern.
          </p>
        </div>
      </div>
    </div>
  );
}

export default App;