'use client'

import { motion } from 'framer-motion'
import { Header } from '@/components/header'
import { AlgorithmCard } from '@/components/algorithm-card'
import { Trees, Network, Zap, Brain, BarChart3, Lightbulb } from 'lucide-react'
import { useState } from 'react'

interface Algorithm {
  id: string
  name: string
  description: string
  category: 'ml' | 'dl' | 'genai'
  complexity: 'beginner' | 'intermediate' | 'advanced'
  icon: JSX.Element
}

const algorithms: Algorithm[] = [
  {
    id: 'decision-tree',
    name: 'Decision Tree',
    description: 'Learn how decision trees make predictions by recursively splitting data based on feature values.',
    category: 'ml',
    complexity: 'beginner',
    icon: <Trees className="w-6 h-6" />,
  },
  {
    id: 'linear-regression',
    name: 'Linear Regression',
    description: 'Understand the fundamentals of fitting a linear relationship between features and targets.',
    category: 'ml',
    complexity: 'beginner',
    icon: <BarChart3 className="w-6 h-6" />,
  },
  {
    id: 'k-means',
    name: 'K-Means Clustering',
    description: 'Explore unsupervised learning by discovering natural groupings in high-dimensional data.',
    category: 'ml',
    complexity: 'intermediate',
    icon: <Network className="w-6 h-6" />,
  },
  {
    id: 'neural-network',
    name: 'Neural Networks',
    description: 'Dive into deep learning with interactive neurons, layers, and backpropagation visualization.',
    category: 'dl',
    complexity: 'intermediate',
    icon: <Brain className="w-6 h-6" />,
  },
  {
    id: 'gradient-descent',
    name: 'Gradient Descent',
    description: 'Watch optimization in action as algorithms navigate loss landscapes to find minima.',
    category: 'ml',
    complexity: 'intermediate',
    icon: <Zap className="w-6 h-6" />,
  },
  {
    id: 'transformer',
    name: 'Transformer Architecture',
    description: 'Visualize attention mechanisms and self-attention in modern language and vision models.',
    category: 'genai',
    complexity: 'advanced',
    icon: <Lightbulb className="w-6 h-6" />,
  },
]

export default function AlgorithmsPage() {
  const [selectedCategory, setSelectedCategory] = useState<'all' | 'ml' | 'dl' | 'genai'>('all')

  const filtered =
    selectedCategory === 'all'
      ? algorithms
      : algorithms.filter((algo) => algo.category === selectedCategory)

  return (
    <main className="min-h-screen bg-background">
      <Header />

      <div className="pt-32 pb-20 px-6">
        <div className="max-w-6xl mx-auto">
          {/* Header */}
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="mb-16">
            <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-primary via-accent to-secondary bg-clip-text text-transparent">
              Algorithm Explorer
            </h1>
            <p className="text-xl text-foreground/70">
              Interactive visualizations of fundamental machine learning algorithms
            </p>
          </motion.div>

          {/* Category Filter */}
          <div className="flex gap-4 mb-12 flex-wrap">
            {(['all', 'ml', 'dl', 'genai'] as const).map((cat) => (
              <button
                key={cat}
                onClick={() => setSelectedCategory(cat)}
                className={`px-4 py-2 rounded-lg transition-all font-semibold ${
                  selectedCategory === cat
                    ? 'bg-primary/30 border border-primary/50 text-primary'
                    : 'glass-effect border border-primary/10 text-foreground/70 hover:border-primary/30'
                }`}
              >
                {cat === 'all' ? 'All' : cat === 'ml' ? 'Machine Learning' : cat === 'dl' ? 'Deep Learning' : 'Gen AI'}
              </button>
            ))}
          </div>

          {/* Algorithm Grid */}
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filtered.map((algo, idx) => (
              <motion.div
                key={algo.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: idx * 0.1 }}
              >
                <AlgorithmCard {...algo} />
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </main>
  )
}
