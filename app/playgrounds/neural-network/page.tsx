'use client'

import { motion } from 'framer-motion'
import { Header } from '@/components/header'
import { CanvasWrapper } from '@/components/canvas-wrapper'
import { NeuralNetworkScene } from '@/components/neural-network-scene'
import { ChevronLeft, Volume2, VolumeX } from 'lucide-react'
import Link from 'next/link'
import { useState, useRef } from 'react'

export default function NeuralNetworkPlayground() {
  const [inputValues, setInputValues] = useState([0.5, 0.5, 0.5])
  const [hiddenUnits, setHiddenUnits] = useState(4)
  const [learningRate, setLearningRate] = useState(0.01)

  const handleInputChange = (index: number, value: number) => {
    const newValues = [...inputValues]
    newValues[index] = value
    setInputValues(newValues)
  }

  return (
    <main className="min-h-screen bg-background">
      <Header />

      <div className="pt-20">
        {/* Back button */}
        <div className="px-6 py-4">
          <Link
            href="/playgrounds"
            className="inline-flex items-center gap-2 text-primary hover:text-primary/80 transition-colors"
          >
            <ChevronLeft className="w-5 h-5" />
            Back to Playgrounds
          </Link>
        </div>

        <div className="grid lg:grid-cols-3 gap-8 px-6 pb-20">
          {/* 3D Visualization */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="lg:col-span-2 h-96 lg:h-screen lg:sticky lg:top-0 rounded-xl overflow-hidden border border-primary/20"
          >
            <CanvasWrapper dpr={1}>
              <NeuralNetworkScene input={inputValues} />
            </CanvasWrapper>
          </motion.div>

          {/* Controls */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="lg:py-20 space-y-6"
          >
            <div>
              <h1 className="text-4xl font-bold mb-2 bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
                Neural Network
              </h1>
              <p className="text-foreground/70">Playground</p>
            </div>

            {/* Input Controls */}
            <div className="glass-effect p-6 rounded-xl border border-primary/20">
              <h3 className="text-lg font-semibold text-foreground mb-4">Input Layer</h3>
              <div className="space-y-4">
                {inputValues.map((value, idx) => (
                  <div key={idx}>
                    <label className="text-sm text-foreground/70 block mb-2">
                      Input {idx + 1}: {value.toFixed(2)}
                    </label>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.01"
                      value={value}
                      onChange={(e) => handleInputChange(idx, parseFloat(e.target.value))}
                      className="w-full accent-primary"
                    />
                  </div>
                ))}
              </div>
            </div>

            {/* Network Architecture */}
            <div className="glass-effect p-6 rounded-xl border border-primary/20">
              <h3 className="text-lg font-semibold text-foreground mb-4">Architecture</h3>
              <div className="space-y-4">
                <div>
                  <label className="text-sm text-foreground/70 block mb-2">
                    Hidden Units: {hiddenUnits}
                  </label>
                  <input
                    type="range"
                    min="1"
                    max="8"
                    step="1"
                    value={hiddenUnits}
                    onChange={(e) => setHiddenUnits(parseInt(e.target.value))}
                    className="w-full accent-accent"
                  />
                </div>
              </div>
            </div>

            {/* Hyperparameters */}
            <div className="glass-effect p-6 rounded-xl border border-primary/20">
              <h3 className="text-lg font-semibold text-foreground mb-4">Hyperparameters</h3>
              <div className="space-y-4">
                <div>
                  <label className="text-sm text-foreground/70 block mb-2">
                    Learning Rate: {learningRate.toFixed(4)}
                  </label>
                  <input
                    type="range"
                    min="0.0001"
                    max="0.1"
                    step="0.0001"
                    value={learningRate}
                    onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                    className="w-full accent-secondary"
                  />
                </div>
              </div>
            </div>

            {/* Tips */}
            <div className="glass-effect p-6 rounded-xl border border-primary/20">
              <h3 className="text-lg font-semibold text-foreground mb-3">Tips</h3>
              <ul className="space-y-2 text-sm text-foreground/70">
                <li>• Move input sliders to see how neurons respond</li>
                <li>• Increase hidden units to increase model capacity</li>
                <li>• Higher learning rate = faster but less stable training</li>
                <li>• Watch neuron activation changes in the 3D view</li>
              </ul>
            </div>
          </motion.div>
        </div>
      </div>
    </main>
  )
}
