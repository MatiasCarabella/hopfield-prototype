import React, { useState, useEffect } from 'react';
import { Play, RotateCcw, Settings, Pause } from 'lucide-react';
import './App.css';

function App() {
  const SIZE = 10;
  const PIXEL_SIZE = 30;
  
  // Patrones de referencia (imágenes a memorizar)
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
    reference: [
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

  const createNoisyPattern = (pattern, noiseLevel = 0.2) => {
    return pattern.map(row => 
      row.map(pixel => 
        Math.random() < noiseLevel ? 1 - pixel : pixel
      )
    );
  };

  const [weights, setWeights] = useState(null);
  const [currentPattern, setCurrentPattern] = useState(createNoisyPattern(REFERENCE_PATTERNS.reference));
  const [history, setHistory] = useState([]);
  const [isRunning, setIsRunning] = useState(false);
  const [iterations, setIterations] = useState(0);
  const [method, setMethod] = useState('hebb');
  const [maxIterations, setMaxIterations] = useState(10);

  // Convertir matriz 2D a vector
  const flatten = (matrix) => matrix.flat();
  
  // Convertir vector a matriz 2D
  const unflatten = (vector) => {
    const matrix = [];
    for (let i = 0; i < SIZE; i++) {
      matrix.push(vector.slice(i * SIZE, (i + 1) * SIZE));
    }
    return matrix;
  };

  // Convertir de {0,1} a {-1,1}
  const toBipolar = (vector) => vector.map(v => v === 0 ? -1 : 1);
  const fromBipolar = (vector) => vector.map(v => v === -1 ? 0 : 1);

  // Entrenar con Regla de Hebb
  const trainHebb = (patterns) => {
    const n = SIZE * SIZE;
    const W = Array(n).fill(0).map(() => Array(n).fill(0));
    
    patterns.forEach(pattern => {
      const p = toBipolar(flatten(pattern));
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          if (i !== j) {
            W[i][j] += p[i] * p[j];
          }
        }
      }
    });

    // Normalizar
    const numPatterns = patterns.length;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        W[i][j] /= numPatterns;
      }
    }

    return W;
  };

  // Calcular pseudoinversa (versión mejorada de Hebb para el ejemplo)
  const trainPseudoinverse = (patterns) => {
    const n = SIZE * SIZE;
    const W = Array(n).fill(0).map(() => Array(n).fill(0));
    
    const P = patterns.map(p => toBipolar(flatten(p)));
    
    P.forEach((p, idx) => {
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          if (i !== j) {
            W[i][j] += p[i] * p[j] / P.length;
          }
        }
      }
    });

    return W;
  };

 // Actualizar red (un paso)
 const updateNetwork = (pattern, W) => {
  const n = SIZE * SIZE;
  const current = toBipolar(flatten(pattern));
  const next = [...current];

  // Actualización asincrónica (1 neurona por paso)
  const indices = Array.from({ length: n }, (_, i) => i);
  const numToUpdate = Math.floor(n * 0.5); // ~50% por iteración (ajustable)

  for (let k = 0; k < numToUpdate; k++) {
    // Elegimos una neurona aleatoria
    const i = indices.splice(Math.floor(Math.random() * indices.length), 1)[0];
    let sum = 0;
    for (let j = 0; j < n; j++) {
      sum += W[i][j] * next[j];
    }

    // Regla determinista (sin ruido térmico)
    next[i] = sum >= 0 ? 1 : -1;
  }

  return unflatten(fromBipolar(next));
};


  // Calcular energía
  const calculateEnergy = (pattern, W) => {
    const p = toBipolar(flatten(pattern));
    const n = p.length;
    let energy = 0;
    
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        energy -= W[i][j] * p[i] * p[j];
      }
    }
    
    return energy / 2;
  };

  // Verificar convergencia
  const hasConverged = (p1, p2) => {
    return JSON.stringify(p1) === JSON.stringify(p2);
  };

  // Inicializar red
  const initializeNetwork = () => {
    const patterns = [REFERENCE_PATTERNS.square, REFERENCE_PATTERNS.reference];
    const W = method === 'hebb' ? trainHebb(patterns) : trainPseudoinverse(patterns);
    setWeights(W);
    
    const noisy = createNoisyPattern(REFERENCE_PATTERNS.reference, 0.25);
    setCurrentPattern(noisy);
    setHistory([noisy]);
    setIterations(0);
    setIsRunning(false);
  };

  // Ejecutar un paso
  const runStep = () => {
    if (!weights || iterations >= maxIterations) {
      setIsRunning(false);
      return;
    }

    const newPattern = updateNetwork(currentPattern, weights);
    
    if (hasConverged(currentPattern, newPattern)) {
      setIsRunning(false);
    } else {
      setCurrentPattern(newPattern);
      setHistory(prev => [...prev, newPattern]);
      setIterations(prev => prev + 1);
    }
  };

  // Ejecutar automáticamente
  useEffect(() => {
    if (isRunning) {
      const timer = setTimeout(runStep, 500);
      return () => clearTimeout(timer);
    }
  }, [isRunning, currentPattern, iterations]);

  useEffect(() => {
    initializeNetwork();
  }, [method]);

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
          Prototipo Red de Hopfield - Identificación de Imágenes
        </h2>
        <p style={{ color: '#94a3b8', marginBottom: '28px', fontSize: '15px' }}>
          Simulación de recuperación de patrones con eliminación de ruido (10x10 píxeles)
        </p>

        {/* Controles */}
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
            {isRunning ? 'Pausar' : 'Ejecutar'}
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
            Un Paso
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
            Reiniciar
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
              <option value="hebb">Regla de Hebb</option>
              <option value="pseudoinverse">Pseudoinversa</option>
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
            Iteración: {iterations} / {maxIterations}
          </div>
        </div>

        {/* Visualización */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '24px', marginBottom: '24px' }}>
          {/* Patrón con Ruido */}
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
              Patrón Inicial (con ruido)
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

          {/* Patrón Actual */}
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
              Patrón Actual (Iteración {iterations})
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

          {/* Patrón de Referencia */}
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
              Patrón de Referencia (objetivo)
            </h3>
            <div style={{ 
              display: 'inline-block', 
              border: '2px solid #10b981',
              borderRadius: '4px',
              overflow: 'hidden',
              boxShadow: '0 0 20px rgba(16, 185, 129, 0.3)'
            }}>
              {REFERENCE_PATTERNS.reference.map((row, i) => (
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

        {/* Información adicional */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '20px' }}>
          <div style={{ 
            background: '#0f172a', 
            padding: '20px', 
            borderRadius: '12px',
            border: '1px solid #334155'
          }}>
            <h4 style={{ fontWeight: '600', color: '#cbd5e1', marginBottom: '12px', fontSize: '16px' }}>
              Características del Modelo
            </h4>
            <ul style={{ fontSize: '14px', color: '#94a3b8', listStyle: 'none', padding: 0, lineHeight: '1.8' }}>
              <li style={{ marginBottom: '6px' }}>• <strong style={{ color: '#cbd5e1' }}>Tamaño:</strong> {SIZE}x{SIZE} = {SIZE*SIZE} neuronas</li>
              <li style={{ marginBottom: '6px' }}>• <strong style={{ color: '#cbd5e1' }}>Método:</strong> {method === 'hebb' ? 'Regla de Hebb' : 'Pseudoinversa'}</li>
              <li style={{ marginBottom: '6px' }}>• <strong style={{ color: '#cbd5e1' }}>Patrones memorizados:</strong> 2 (cuadrado y rombo)</li>
              <li style={{ marginBottom: '6px' }}>• <strong style={{ color: '#cbd5e1' }}>Actualización:</strong> Asíncrona aleatoria</li>
              <li style={{ marginBottom: '6px' }}>• <strong style={{ color: '#cbd5e1' }}>Elemento de referencia:</strong> Escuadra</li>
            </ul>
          </div>

          <div style={{ 
            background: '#0f172a', 
            padding: '20px', 
            borderRadius: '12px',
            border: '1px solid #334155'
          }}>
            <h4 style={{ fontWeight: '600', color: '#cbd5e1', marginBottom: '12px', fontSize: '16px' }}>
              Estado del Proceso
            </h4>
            <ul style={{ fontSize: '14px', color: '#94a3b8', listStyle: 'none', padding: 0, lineHeight: '1.8' }}>
              <li style={{ marginBottom: '6px' }}>
                • <strong style={{ color: '#cbd5e1' }}>Convergencia:</strong> {
                  iterations >= maxIterations 
                    ? <span style={{ color: '#f59e0b' }}>Máx. iteraciones</span>
                    : isRunning 
                      ? <span style={{ color: '#3b82f6' }}>En proceso...</span>
                      : <span style={{ color: '#10b981' }}>Detenido</span>
                }
              </li>
              <li style={{ marginBottom: '6px' }}>
                • <strong style={{ color: '#cbd5e1' }}>Energía:</strong> {weights ? calculateEnergy(currentPattern, weights).toFixed(2) : 'N/A'}
              </li>
              <li style={{ marginBottom: '6px' }}>
                • <strong style={{ color: '#cbd5e1' }}>Ruido inicial:</strong> ~25% píxeles alterados
              </li>
            </ul>
          </div>
        </div>

        {/* Leyenda */}
        <div style={{ 
          marginTop: '24px', 
          padding: '20px', 
          background: 'rgba(251, 191, 36, 0.1)', 
          borderLeft: '4px solid #f59e0b', 
          borderRadius: '8px',
          border: '1px solid rgba(251, 191, 36, 0.2)'
        }}>
          <p style={{ fontSize: '14px', color: '#fbbf24', lineHeight: '1.6' }}>
            <strong style={{ color: '#fcd34d' }}>Nota:</strong> El modelo de Hopfield actúa como memoria asociativa, eliminando ruido y 
            recuperando el patrón más cercano memorizado. La escuadra (esquina inferior izquierda) sirve como 
            referencia fija para medir desplazamientos relativos del patrón objetivo.
          </p>
        </div>
      </div>
    </div>
  );
}

export default App;