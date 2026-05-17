'use client'

import { motion } from 'framer-motion'
import { Header } from '@/components/header'
import { CanvasWrapper } from '@/components/canvas-wrapper'
import { KMeansScene } from '@/components/kmeans-scene'
import { ChevronLeft, Play, Pause, RotateCcw } from 'lucide-react'
import Link from 'next/link'
import { useState, useEffect } from 'react'

export default function KMeansPage() {
  const [isRunning, setIsRunning] = useState(false)
  const [iteration, setIteration] = useState(0)

  useEffect(() => {
    if (!isRunning) return

    const interval = setInterval(() => {
      setIteration((prev) => (prev + 1) % 50)
    }, 1000)

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
              <KMeansScene iteration={iteration} />
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
                K-Means Clustering
              </h1>

              <p className="text-lg text-foreground/70 mb-8">
                K-Means is an unsupervised learning algorithm that partitions data into K clusters by
                minimizing the variance within each cluster. Watch as the algorithm iteratively assigns
                points to the nearest cluster center and updates the centers.
              </p>

              {/* Controls */}
              <div className="glass-effect p-6 rounded-xl mb-8 border border-primary/20">
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

              {/* Algorithm Steps */}
              <div className="space-y-6 mb-12">
                <h2 className="text-2xl font-bold text-foreground">How it Works</h2>

                {[
                  {
                    step: 1,
                    title: 'Initialize Centers',
                    description:
                      'Randomly select K points as the initial cluster centers. These will be updated iteratively.',
                  },
                  {
                    step: 2,
                    title: 'Assign Points',
                    description:
                      'Calculate the distance from each data point to all cluster centers. Assign each point to its nearest center.',
                  },
                  {
                    step: 3,
                    title: 'Update Centers',
                    description:
                      'Calculate the mean of all points in each cluster and move the center to this mean position.',
                  },
                  {
                    step: 4,
                    title: 'Convergence',
                    description:
                      'Repeat steps 2-3 until the cluster centers no longer move significantly or iteration limit is reached.',
                  },
                ].map((item, idx) => (
                  <motion.div
                    key={idx}
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    transition={{ delay: idx * 0.1 }}
                    className="glass-effect p-6 rounded-xl border border-primary/20 hover:border-primary/40 transition-all"
                  >
                    <div className="flex gap-4">
                      <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary/30 border border-primary/50 flex items-center justify-center font-bold text-primary">
                        {item.step}
                      </div>
                      <div>
                        <h3 className="font-semibold text-foreground mb-1">{item.title}</h3>
                        <p className="text-foreground/70">{item.description}</p>
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>

              {/* Code Example */}
              <div className="glass-effect p-6 rounded-xl border border-primary/20">
                <h3 className="text-lg font-semibold text-foreground mb-4">Python Implementation</h3>
                <pre className="bg-background/50 p-4 rounded-lg overflow-x-auto text-sm text-accent">
{`from sklearn.cluster import KMeans
import numpy as np

# Sample data
X = np.random.randn(150, 2)

# Create and fit model
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Get cluster assignments
labels = kmeans.labels_
centers = kmeans.cluster_centers_`}
                </pre>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </main>
  )
}
