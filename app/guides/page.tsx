'use client'

import { motion } from 'framer-motion'
import { Header } from '@/components/header'
import { BookOpen, Lightbulb, Zap, Brain } from 'lucide-react'
import Link from 'next/link'

const guides = [
  {
    id: 'ml-basics',
    title: 'Machine Learning Basics',
    description: 'Start here if you&apos;re new to ML. Learn fundamental concepts like supervised vs. unsupervised learning, training/testing splits, and evaluation metrics.',
    icon: <BookOpen className="w-6 h-6" />,
    topics: ['Supervised Learning', 'Unsupervised Learning', 'Evaluation Metrics', 'Cross-Validation'],
  },
  {
    id: 'neural-networks',
    title: 'Deep Dive: Neural Networks',
    description: 'Understand how neural networks work, from the perceptron to deep learning. Explore activation functions, backpropagation, and optimization techniques.',
    icon: <Brain className="w-6 h-6" />,
    topics: ['Perceptrons', 'Activation Functions', 'Backpropagation', 'Optimization'],
  },
  {
    id: 'feature-engineering',
    title: 'Feature Engineering Guide',
    description: 'Learn how to prepare and transform data for machine learning. Discover feature selection, scaling, normalization, and encoding techniques.',
    icon: <Zap className="w-6 h-6" />,
    topics: ['Scaling & Normalization', 'Feature Selection', 'Encoding', 'Handling Missing Data'],
  },
  {
    id: 'model-selection',
    title: 'Choosing the Right Model',
    description: 'A practical guide to selecting appropriate algorithms for different problem types. Understand the tradeoffs between accuracy, interpretability, and training time.',
    icon: <Lightbulb className="w-6 h-6" />,
    topics: ['Problem Type Matching', 'Complexity Tradeoffs', 'Ensemble Methods', 'When to Use What'],
  },
]

export default function GuidesPage() {
  return (
    <main className="min-h-screen bg-background">
      <Header />

      <div className="pt-32 pb-20 px-6">
        <div className="max-w-6xl mx-auto">
          {/* Header */}
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="mb-16">
            <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-primary via-accent to-secondary bg-clip-text text-transparent">
              Learning Guides
            </h1>
            <p className="text-xl text-foreground/70">
              Comprehensive tutorials to deepen your understanding of machine learning
            </p>
          </motion.div>

          {/* Guides Grid */}
          <div className="space-y-6">
            {guides.map((guide, idx) => (
              <motion.div
                key={guide.id}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: idx * 0.1 }}
              >
                <div className="glass-effect p-8 rounded-xl border border-primary/20 hover:border-primary/40 transition-all group cursor-pointer">
                  <div className="flex gap-6">
                    <div className="flex-shrink-0 text-accent group-hover:scale-110 transition-transform">
                      {guide.icon}
                    </div>
                    <div className="flex-grow">
                      <h3 className="text-2xl font-semibold text-foreground mb-2">{guide.title}</h3>
                      <p className="text-foreground/70 mb-6">{guide.description}</p>

                      <div className="flex flex-wrap gap-2 mb-6">
                        {guide.topics.map((topic) => (
                          <span
                            key={topic}
                            className="text-xs px-3 py-1 rounded-full bg-primary/10 text-primary border border-primary/20"
                          >
                            {topic}
                          </span>
                        ))}
                      </div>

                      <button className="text-primary font-semibold text-sm hover:gap-3 transition-all inline-flex items-center gap-2">
                        Read Guide
                        <span className="text-lg group-hover:translate-x-1 transition-transform">→</span>
                      </button>
                    </div>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>

          {/* Additional Resources */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="mt-20 glass-effect p-8 rounded-xl border border-primary/20"
          >
            <h2 className="text-2xl font-bold text-foreground mb-6">Additional Resources</h2>

            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h3 className="font-semibold text-accent mb-3">Recommended Reading</h3>
                <ul className="space-y-2 text-foreground/70">
                  <li>• &quot;Introduction to Statistical Learning&quot; (ISLR)</li>
                  <li>• &quot;Deep Learning&quot; by Goodfellow, Bengio & Courville</li>
                  <li>• &quot;Hands-on ML with Scikit-Learn and TensorFlow&quot;</li>
                  <li>• &quot;Pattern Recognition and Machine Learning&quot; (PRML)</li>
                </ul>
              </div>

              <div>
                <h3 className="font-semibold text-secondary mb-3">Online Communities</h3>
                <ul className="space-y-2 text-foreground/70">
                  <li>• Kaggle (kaggle.com) - Competitions & Datasets</li>
                  <li>• Papers with Code - Latest Research</li>
                  <li>• arXiv - Preprints & Research Papers</li>
                  <li>• Reddit r/MachineLearning - Community Discussions</li>
                </ul>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </main>
  )
}
