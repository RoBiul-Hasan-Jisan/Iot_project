'use client'

import { motion } from 'framer-motion'
import { Header } from '@/components/header'
import { CanvasWrapper } from '@/components/canvas-wrapper'
import { LinearRegressionScene } from '@/components/linear-regression-scene'
import { ChevronLeft, Play, Pause, RotateCcw } from 'lucide-react'
import Link from 'next/link'
import { useState, useEffect } from 'react'

export default function LinearRegressionPage() {
  const [isRunning, setIsRunning] = useState(false)
  const [iteration, setIteration] = useState(0)
  const [learningRate, setLearningRate] = useState(0.01)

  useEffect(() => {
    if (!isRunning) return

    const interval = setInterval(() => {
      setIteration((prev) => (prev + 1) % 100)
    }, 500)

    return () => clearInterval(interval)
  }, [isRunning])

  return (
    <main className="min-h-screen bg-background">
      <Header />

      <div className="pt-20">
        {/* Back button */}
        <div className="px-6 py-4">
          <Link
            href="/algorithms"
            className="inline-flex items-center gap-2 text-primary hover:text-primary/80 transition-colors"
          >
            <ChevronLeft className="w-5 h-5" />
            Back to Algorithms
          </Link>
        </div>

        <div className="grid lg:grid-cols-2 gap-8 px-6 pb-20">
          {/* 3D Visualization */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="h-96 lg:h-screen lg:sticky lg:top-0 rounded-xl overflow-hidden border border-primary/20"
          >
            <CanvasWrapper dpr={1}>
              <LinearRegressionScene learningRate={learningRate} iteration={iteration} />
            </CanvasWrapper>
          </motion.div>

          {/* Content */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="lg:py-20"
          >
            <div>
              <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
                Linear Regression
              </h1>

              <p className="text-lg text-foreground/70 mb-8">
                Linear Regression is the foundation of supervised learning. It models the linear relationship
                between features (X) and a continuous target (y). The algorithm finds the best-fit line that
                minimizes the prediction error, measured as the sum of squared residuals (shown as red lines above).
              </p>

              {/* Controls */}
              <div className="glass-effect p-6 rounded-xl mb-8 border border-primary/20">
                <div className="mb-4">
                  <p className="text-sm text-foreground/60 mb-4">Iteration: {iteration}</p>
                  <div className="flex gap-3">
                    <button
                      onClick={() => setIsRunning(!isRunning)}
                      className="flex items-center gap-2 px-4 py-2 rounded-lg bg-primary/30 border border-primary/50 text-primary hover:bg-primary/40 transition-all font-semibold"
                    >
                      {isRunning ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                      {isRunning ? 'Pause' : 'Play'}
                    </button>
                    <button
                      onClick={() => {
                        setIsRunning(false)
                        setIteration(0)
                      }}
                      className="flex items-center gap-2 px-4 py-2 rounded-lg bg-accent/20 border border-accent/50 text-accent hover:bg-accent/30 transition-all font-semibold"
                    >
                      <RotateCcw className="w-4 h-4" />
                      Reset
                    </button>
                  </div>
                </div>

                <div>
                  <label className="text-sm text-foreground/70 block mb-2">
                    Learning Rate: {learningRate.toFixed(4)}
                  </label>
                  <input
                    type="range"
                    min="0.001"
                    max="0.1"
                    step="0.001"
                    value={learningRate}
                    onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                    className="w-full accent-primary"
                  />
                </div>
              </div>

              {/* The Math */}
              <div className="space-y-6 mb-12">
                <h2 className="text-2xl font-bold text-foreground">The Mathematics</h2>

                <div className="glass-effect p-6 rounded-xl border border-primary/20">
                  <h3 className="font-semibold text-accent mb-3">Hypothesis Function</h3>
                  <p className="text-foreground/70 mb-2">Predict y using a linear combination of features:</p>
                  <div className="bg-background/50 p-3 rounded text-accent text-sm font-mono">
                    ŷ = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ
                  </div>
                </div>

                <div className="glass-effect p-6 rounded-xl border border-primary/20">
                  <h3 className="font-semibold text-accent mb-3">Cost Function (MSE)</h3>
                  <p className="text-foreground/70 mb-2">Measure average squared error between predictions and reality:</p>
                  <div className="bg-background/50 p-3 rounded text-accent text-sm font-mono">
                    J(w) = (1/2m) Σ(ŷⁱ - yⁱ)²
                  </div>
                </div>

                <div className="glass-effect p-6 rounded-xl border border-primary/20">
                  <h3 className="font-semibold text-accent mb-3">Gradient Descent Update</h3>
                  <p className="text-foreground/70 mb-2">Update weights to reduce cost:</p>
                  <div className="bg-background/50 p-3 rounded text-accent text-sm font-mono">
                    w := w - α(∂J/∂w)
                  </div>
                  <p className="text-foreground/70 text-xs mt-2">α = learning rate (step size)</p>
                </div>
              </div>

              {/* When to Use */}
              <div className="glass-effect p-6 rounded-xl border border-primary/20 mb-8">
                <h3 className="text-lg font-semibold text-foreground mb-4">Use Cases</h3>
                <ul className="space-y-2 text-foreground/70">
                  <li>• Stock price prediction</li>
                  <li>• House price estimation</li>
                  <li>• Temperature forecasting</li>
                  <li>• Any problem with continuous target values</li>
                  <li>• Baseline model before trying complex algorithms</li>
                </ul>
              </div>

              {/* Code Example */}
              <div className="glass-effect p-6 rounded-xl border border-primary/20">
                <h3 className="text-lg font-semibold text-foreground mb-4">Python Implementation</h3>
                <pre className="bg-background/50 p-4 rounded-lg overflow-x-auto text-sm text-accent">
{`from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

# Load data
X = np.random.randn(100, 3)
y = 3*X[:, 0] + 2*X[:, 1] - 1*X[:, 2] + np.random.randn(100)*0.1

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Get coefficients
print(f"Intercept: {model.intercept_}")
print(f"Coefficients: {model.coef_}")`}
                </pre>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </main>
  )
}
