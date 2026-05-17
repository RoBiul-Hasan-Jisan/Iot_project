'use client'

import { motion } from 'framer-motion'
import { Header } from '@/components/header'
import { CanvasWrapper } from '@/components/canvas-wrapper'
import { HeroScene } from '@/components/hero-scene'
import { ArrowRight, Sparkles, Brain, Zap } from 'lucide-react'
import Link from 'next/link'

export default function Home() {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.2,
        delayChildren: 0.3,
      },
    },
  }

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.8 },
    },
  }

  return (
    <main className="min-h-screen bg-background overflow-hidden">
      <Header />

      {/* Hero Section */}
      <section className="relative h-screen flex items-center justify-center pt-20">
        {/* 3D Canvas Background */}
        <div className="absolute inset-0 top-20">
          <CanvasWrapper dpr={1.5}>
            <HeroScene />
          </CanvasWrapper>
        </div>

        {/* Content Overlay */}
        <motion.div
          className="relative z-10 text-center max-w-4xl px-6"
          variants={containerVariants}
          initial="hidden"
          animate="visible"
        >
          <motion.div variants={itemVariants} className="flex items-center justify-center gap-2 mb-6">
            <Sparkles className="w-5 h-5 text-accent" />
            <span className="text-accent text-sm font-semibold">Interactive ML Education</span>
          </motion.div>

          <motion.h1
            variants={itemVariants}
            className="text-6xl md:text-7xl font-bold mb-6 bg-gradient-to-r from-primary via-accent to-secondary bg-clip-text text-transparent"
          >
            Visualize Intelligence
          </motion.h1>

          <motion.p
            variants={itemVariants}
            className="text-xl text-foreground/70 mb-8 max-w-2xl mx-auto leading-relaxed"
          >
            Explore machine learning algorithms through interactive 3D visualizations and hands-on
            playgrounds. Understand the fundamentals of AI, from basic decision trees to advanced
            neural networks.
          </motion.p>

          <motion.div variants={itemVariants} className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link
              href="/algorithms"
              className="px-8 py-4 rounded-lg bg-primary/30 border border-primary/50 text-primary hover:bg-primary/40 transition-all font-semibold flex items-center justify-center gap-2 group"
            >
              Explore Algorithms
              <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
            </Link>
            <Link
              href="/playgrounds"
              className="px-8 py-4 rounded-lg bg-accent/20 border border-accent/50 text-accent hover:bg-accent/30 transition-all font-semibold flex items-center justify-center gap-2"
            >
              Try Playgrounds
            </Link>
          </motion.div>
        </motion.div>
      </section>

      {/* Features Section */}
      <section className="relative z-20 py-24 px-6">
        <div className="max-w-6xl mx-auto">
          <motion.h2
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-4xl font-bold mb-16 text-center bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent"
          >
            Why MLVerse?
          </motion.h2>

          <div className="grid md:grid-cols-3 gap-8">
            {[
              {
                icon: <Brain className="w-8 h-8" />,
                title: 'Interactive Learning',
                description: 'Learn by doing with hands-on playgrounds where you can experiment with algorithms in real-time.',
              },
              {
                icon: <Zap className="w-8 h-8" />,
                title: '3D Visualizations',
                description: 'See algorithms come alive with stunning 3D visualizations that make complex concepts intuitive.',
              },
              {
                icon: <Sparkles className="w-8 h-8" />,
                title: 'Comprehensive Guides',
                description: 'From fundamentals to advanced topics, explore detailed guides for each algorithm with code examples.',
              },
            ].map((feature, idx) => (
              <motion.div
                key={idx}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: idx * 0.2 }}
                className="glass-effect p-8 rounded-xl border border-primary/20 hover:border-primary/50 transition-all group"
              >
                <div className="text-accent mb-4 group-hover:scale-110 transition-transform">
                  {feature.icon}
                </div>
                <h3 className="text-xl font-semibold mb-3 text-foreground">{feature.title}</h3>
                <p className="text-foreground/70">{feature.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="relative z-20 border-t border-primary/10 py-12 px-6">
        <div className="max-w-6xl mx-auto text-center text-foreground/50 text-sm">
          <p>MLVerse © 2024 • Making machine learning beautiful and accessible</p>
        </div>
      </footer>
    </main>
  )
}
