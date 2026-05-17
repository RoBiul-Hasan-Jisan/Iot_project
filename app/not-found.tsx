'use client'

import { motion } from 'framer-motion'
import Link from 'next/link'
import { Header } from '@/components/header'
import { Home, ArrowRight } from 'lucide-react'

export default function NotFound() {
  return (
    <main className="min-h-screen bg-background">
      <Header />

      <div className="h-screen flex items-center justify-center px-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center max-w-md"
        >
          <motion.div
            animate={{ scale: [1, 1.1, 1] }}
            transition={{ duration: 2, repeat: Infinity }}
            className="text-8xl font-bold mb-6 bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent"
          >
            404
          </motion.div>

          <h1 className="text-3xl font-bold text-foreground mb-4">Page Not Found</h1>

          <p className="text-lg text-foreground/70 mb-8">
            This page doesn't exist yet. But the MLVerse is still waiting to be explored!
          </p>

          <Link
            href="/"
            className="inline-flex items-center gap-2 px-8 py-4 rounded-lg bg-primary/30 border border-primary/50 text-primary hover:bg-primary/40 transition-all font-semibold"
          >
            <Home className="w-5 h-5" />
            Back to Home
            <ArrowRight className="w-5 h-5" />
          </Link>
        </motion.div>
      </div>
    </main>
  )
}
