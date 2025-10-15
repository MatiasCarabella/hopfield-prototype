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
    
    // Actualización asíncrona (orden aleatorio)
    const indices = Array.from({length: n}, (_, i) => i);
    for (let idx = 0; idx < n; idx++) {
      const i = indices[Math.floor(Math.random() * indices.length)];
      indices.splice(indices.indexOf(i), 1);
      
      let sum = 0;
      for (let j = 0; j < n; j++) {
        sum += W[i][j] * next[j];
      }
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
    <div style={{ width: '100%', minHeight: '100vh', background: 'linear-gradient(to bottom right, #f8fafc, #e2e8f0)', padding: '24px' }}>
      <div style={{ maxWidth: '1200px', margin: '0 auto', background: 'white', borderRadius: '12px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)', padding: '24px' }}>
        <h2 style={{ fontSize: '24px', fontWeight: 'bold', color: '#1e293b', marginBottom: '8px' }}>
          Prototipo Red de Hopfield - Identificación de Imágenes
        </h2>
        <p style={{ color: '#64748b', marginBottom: '24px' }}>
          Simulación de recuperación de patrones con eliminación de ruido (10x10 píxeles)
        </p>

        {/* Controles */}
        <div style={{ display: 'flex', gap: '16px', marginBottom: '24px', flexWrap: 'wrap', alignItems: 'center' }}>
          <button
            onClick={() => setIsRunning(!isRunning)}
            disabled={iterations >= maxIterations}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              padding: '8px 16px',
              background: iterations >= maxIterations ? '#9ca3af' : '#2563eb',
              color: 'white',
              border: 'none',
              borderRadius: '8px',
              cursor: iterations >= maxIterations ? 'not-allowed' : 'pointer',
              fontSize: '14px',
              fontWeight: '500'
            }}
          >
            {isRunning ? <Pause size={20} /> : <Play size={20} />}
            {isRunning ? 'Pausar' : 'Ejecutar'}
          </button>
          
          <button
            onClick={runStep}
            disabled={isRunning || iterations >= maxIterations}
            style={{
              padding: '8px 16px',
              background: (isRunning || iterations >= maxIterations) ? '#9ca3af' : '#16a34a',
              color: 'white',
              border: 'none',
              borderRadius: '8px',
              cursor: (isRunning || iterations >= maxIterations) ? 'not-allowed' : 'pointer',
              fontSize: '14px',
              fontWeight: '500'
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
              padding: '8px 16px',
              background: '#9333ea',
              color: 'white',
              border: 'none',
              borderRadius: '8px',
              cursor: 'pointer',
              fontSize: '14px',
              fontWeight: '500'
            }}
          >
            <RotateCcw size={20} />
            Reiniciar
          </button>

          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Settings size={20} style={{ color: '#64748b' }} />
            <select
              value={method}
              onChange={(e) => setMethod(e.target.value)}
              style={{
                padding: '8px 12px',
                border: '1px solid #cbd5e1',
                borderRadius: '8px',
                fontSize: '14px',
                cursor: 'pointer'
              }}
            >
              <option value="hebb">Regla de Hebb</option>
              <option value="pseudoinverse">Pseudoinversa</option>
            </select>
          </div>

          <div style={{ marginLeft: 'auto', fontSize: '18px', fontWeight: '600', color: '#334155' }}>
            Iteración: {iterations} / {maxIterations}
          </div>
        </div>

        {/* Visualización */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '24px', marginBottom: '24px' }}>
          {/* Patrón con Ruido */}
          <div style={{ background: '#f8fafc', padding: '16px', borderRadius: '8px' }}>
            <h3 style={{ fontWeight: '600', color: '#334155', marginBottom: '12px', textAlign: 'center' }}>
              Patrón Inicial (con ruido)
            </h3>
            <div style={{ display: 'inline-block', border: '2px solid #cbd5e1' }}>
              {history[0]?.map((row, i) => (
                <div key={i} style={{ display: 'flex' }}>
                  {row.map((pixel, j) => (
                    <div
                      key={j}
                      style={{
                        width: PIXEL_SIZE,
                        height: PIXEL_SIZE,
                        background: pixel ? '#1e293b' : 'white'
                      }}
                    />
                  ))}
                </div>
              ))}
            </div>
          </div>

          {/* Patrón Actual */}
          <div style={{ background: '#eff6ff', padding: '16px', borderRadius: '8px' }}>
            <h3 style={{ fontWeight: '600', color: '#1e40af', marginBottom: '12px', textAlign: 'center' }}>
              Patrón Actual (Iteración {iterations})
            </h3>
            <div style={{ display: 'inline-block', border: '2px solid #60a5fa' }}>
              {currentPattern.map((row, i) => (
                <div key={i} style={{ display: 'flex' }}>
                  {row.map((pixel, j) => (
                    <div
                      key={j}
                      style={{
                        width: PIXEL_SIZE,
                        height: PIXEL_SIZE,
                        background: pixel ? '#1e40af' : 'white'
                      }}
                    />
                  ))}
                </div>
              ))}
            </div>
          </div>

          {/* Patrón de Referencia */}
          <div style={{ background: '#f0fdf4', padding: '16px', borderRadius: '8px' }}>
            <h3 style={{ fontWeight: '600', color: '#15803d', marginBottom: '12px', textAlign: 'center' }}>
              Patrón de Referencia (objetivo)
            </h3>
            <div style={{ display: 'inline-block', border: '2px solid #4ade80' }}>
              {REFERENCE_PATTERNS.reference.map((row, i) => (
                <div key={i} style={{ display: 'flex' }}>
                  {row.map((pixel, j) => (
                    <div
                      key={j}
                      style={{
                        width: PIXEL_SIZE,
                        height: PIXEL_SIZE,
                        background: pixel ? '#15803d' : 'white'
                      }}
                    />
                  ))}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Información adicional */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '16px' }}>
          <div style={{ background: '#f8fafc', padding: '16px', borderRadius: '8px' }}>
            <h4 style={{ fontWeight: '600', color: '#334155', marginBottom: '8px' }}>Características del Modelo</h4>
            <ul style={{ fontSize: '14px', color: '#64748b', listStyle: 'none', padding: 0 }}>
              <li style={{ marginBottom: '4px' }}>• <strong>Tamaño:</strong> {SIZE}x{SIZE} = {SIZE*SIZE} neuronas</li>
              <li style={{ marginBottom: '4px' }}>• <strong>Método:</strong> {method === 'hebb' ? 'Regla de Hebb' : 'Pseudoinversa'}</li>
              <li style={{ marginBottom: '4px' }}>• <strong>Patrones memorizados:</strong> 2 (cuadrado y rombo)</li>
              <li style={{ marginBottom: '4px' }}>• <strong>Actualización:</strong> Asíncrona aleatoria</li>
              <li style={{ marginBottom: '4px' }}>• <strong>Elemento de referencia:</strong> Escuadra (esquina inferior izquierda)</li>
            </ul>
          </div>

          <div style={{ background: '#f8fafc', padding: '16px', borderRadius: '8px' }}>
            <h4 style={{ fontWeight: '600', color: '#334155', marginBottom: '8px' }}>Estado del Proceso</h4>
            <ul style={{ fontSize: '14px', color: '#64748b', listStyle: 'none', padding: 0 }}>
              <li style={{ marginBottom: '4px' }}>
                • <strong>Convergencia:</strong> {iterations >= maxIterations ? 'Máx. iteraciones' : isRunning ? 'En proceso...' : 'Detenido'}
              </li>
              <li style={{ marginBottom: '4px' }}>
                • <strong>Energía aproximada:</strong> {weights ? calculateEnergy(currentPattern, weights).toFixed(2) : 'N/A'}
              </li>
              <li style={{ marginBottom: '4px' }}>• <strong>Ruido inicial:</strong> ~25% de píxeles alterados</li>
            </ul>
          </div>
        </div>

        {/* Leyenda */}
        <div style={{ marginTop: '24px', padding: '16px', background: '#fffbeb', borderLeft: '4px solid #f59e0b', borderRadius: '8px' }}>
          <p style={{ fontSize: '14px', color: '#78350f' }}>
            <strong>Nota:</strong> El modelo de Hopfield actúa como memoria asociativa, eliminando ruido y 
            recuperando el patrón más cercano memorizado. La escuadra (esquina inferior izquierda) sirve como 
            referencia fija para medir desplazamientos relativos del patrón objetivo.
          </p>
        </div>
      </div>
    </div>
  );
}

export default App;