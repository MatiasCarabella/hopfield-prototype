<div align="center">

# 🧠 Hopfield Network - Pattern Recognition

[![React](https://img.shields.io/badge/React-19.2.0-61DAFB?logo=react&logoColor=white)](https://reactjs.org/)
[![Vite](https://img.shields.io/badge/Vite-7.2.6-646CFF?logo=vite&logoColor=white)](https://vitejs.dev/)
[![JavaScript](https://img.shields.io/badge/JavaScript-ES2024-F7DF1E?logo=javascript&logoColor=black)](https://developer.mozilla.org/en-US/docs/Web/JavaScript)
[![Lucide](https://img.shields.io/badge/Lucide-0.555.0-F56565?logo=lucide&logoColor=white)](https://lucide.dev/)
[![License](https://img.shields.io/badge/License-MIT-blue?)](LICENSE)

An interactive visualization of a **Hopfield Neural Network** that demonstrates associative memory, pattern recognition, and noise reduction capabilities. Watch as the network recovers clean patterns from noisy inputs in real-time.

<img width="1452" height="739" alt="image" src="https://github.com/user-attachments/assets/e2f67f89-3dfa-4213-82d4-98643f6cbf4a" />

*The network recovering a diamond pattern from noisy input through iterative convergence*

</div>

## ✨ Features

- 🎯 **Real-time Pattern Recovery** - Visualize how the network converges from noisy inputs to memorized patterns
- 🔄 **Two Training Methods** - Compare Hebb's Rule and Pseudoinverse learning algorithms
- 🎨 **Interactive Controls** - Run, pause, step through iterations, and reset the simulation
- 📊 **Live Metrics** - Monitor energy levels, convergence status, and iteration count
- 🎭 **Multiple Patterns** - Network memorizes and distinguishes between square and diamond shapes
- 🌐 **Responsive Design** - Modern, dark-themed UI that works on all screen sizes


## 🚀 Demo

The network starts with a noisy version of a memorized pattern (25% pixel noise) and iteratively updates neurons until it converges to the closest stored pattern. The corner pixels serve as reference markers to track pattern alignment.

## 🛠️ Technologies

- **React 19** - Modern UI with hooks (useState, useEffect, useCallback, useMemo)
- **Vite** - Lightning-fast build tool and dev server
- **Lucide React** - Beautiful icon library
- **ESLint** - Code quality and consistency

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/hopfield-prototype.git

# Navigate to project directory
cd hopfield-prototype

# Install dependencies
npm install

# Start development server
npm run dev
```

## 🎮 Usage

1. **Run** - Start automatic iteration (500ms per step)
2. **Step** - Manually advance one iteration
3. **Reset** - Generate new noisy pattern and restart
4. **Method** - Switch between Hebb Rule and Pseudoinverse training

## 🧮 How It Works

### Hopfield Network Basics

A Hopfield network is a form of recurrent artificial neural network that serves as content-addressable memory. Key characteristics:

- **Fully Connected** - Every neuron connects to every other neuron
- **Symmetric Weights** - Connection weights are bidirectional (Wij = Wji)
- **Energy Function** - Network converges to local minima in energy landscape
- **Associative Memory** - Retrieves complete patterns from partial or noisy inputs

### Training Methods

**Hebb's Rule**
```
Wij = (1/P) Σ(p=1 to P) xi^p * xj^p
```
Simple and biologically plausible learning rule.

**Pseudoinverse**
```
W = (1/P) Σ(p=1 to P) xp * xp^T
```
Improved capacity and pattern separation.

### Update Rule

Asynchronous random update (50% of neurons per iteration):
```
xi(t+1) = sign(Σj Wij * xj(t))
```

## 📁 Project Structure

```
hopfield-prototype/
├── src/
│   ├── App.jsx          # Main component with network logic
│   ├── App.css          # Component styles
│   ├── main.jsx         # React entry point
│   └── index.css        # Global styles
├── public/              # Static assets
├── index.html           # HTML template
├── vite.config.js       # Vite configuration
├── eslint.config.js     # ESLint rules
└── package.json         # Dependencies
```

## 🔧 Development

```bash
# Run development server
npm run dev

# Build for production
npm run build

# Lint code
npm run lint
```

## 📊 Network Parameters

- **Grid Size**: 10×10 (100 neurons)
- **Patterns**: 2 memorized shapes
- **Noise Level**: 25% pixel corruption
- **Max Iterations**: 10
- **Update Strategy**: Asynchronous random (50% neurons/step)

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
