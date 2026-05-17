'use client'

import { motion } from 'framer-motion'
import { Header } from '@/components/header'
import { AlgorithmCard } from '@/components/algorithm-card'
import { Network, Zap, BarChart3, FileCode } from 'lucide-react'

const playgrounds = [
  {
    id: 'neural-network',
    name: 'Neural Network Playground',
    description: 'Build and train your own neural network. Experiment with layers, activation functions, and hyperparameters in real-time.',
    category: 'dl' as const,
    complexity: 'intermediate' as const,
    icon: <Network className="w-6 h-6" />,
  },
  {
    id: 'linear-regression',
    name: 'Linear Regression Tuner',
    description: 'Visualize how different learning rates and iterations affect model convergence on 2D data.',
    category: 'ml' as const,
    complexity: 'beginner' as const,
    icon: <BarChart3 className="w-6 h-6" />,
  },
  {
    id: 'gradient-descent',
    name: 'Gradient Descent Explorer',
    description: 'Navigate loss landscapes in 3D and see how optimization algorithms find the minimum.',
    category: 'ml' as const,
    complexity: 'intermediate' as const,
    icon: <Zap className="w-6 h-6" />,
  },
  {
    id: 'decision-tree',
    name: 'Decision Tree Builder',
    description: 'Interactively build and visualize decision trees. See how split selection affects model performance.',
    category: 'ml' as const,
    complexity: 'beginner' as const,
    icon: <FileCode className="w-6 h-6" />,
  },
]

export default function PlaygroundsPage() {
  return (
    <main className="min-h-screen bg-background">
      <Header />

      <div className="pt-32 pb-20 px-6">
        <div className="max-w-6xl mx-auto">
          {/* Header */}
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="mb-16">
            <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-primary via-accent to-secondary bg-clip-text text-transparent">
              Interactive Playgrounds
            </h1>
            <p className="text-xl text-foreground/70">
              Hands-on experiments where you can tune hyperparameters and see results instantly
            </p>
          </motion.div>

          {/* Grid */}
          <div className="grid md:grid-cols-2 gap-6">
            {playgrounds.map((playground, idx) => (
              <motion.div
                key={playground.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: idx * 0.1 }}
              >
                <AlgorithmCard {...playground} />
              </motion.div>
            ))}
          </div>

          {/* Feature Info */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="mt-20 glass-effect p-8 rounded-xl border border-primary/20"
          >
            <h2 className="text-2xl font-bold text-foreground mb-4">Coming Soon</h2>
            <p className="text-foreground/70 mb-4">
              Additional playgrounds will be rolled out in phases:
            </p>
            <ul className="space-y-2 text-foreground/70">
              <li>• Advanced hyperparameter tuning for different datasets</li>
              <li>• Real-time model comparison across algorithms</li>
              <li>• Dataset upload and custom training</li>
              <li>• Performance profiling and optimization challenges</li>
            </ul>
          </motion.div>
        </div>
      </div>
    </main>
  )
}
