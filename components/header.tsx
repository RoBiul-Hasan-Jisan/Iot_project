'use client'

import { motion } from 'framer-motion'
import Link from 'next/link'

export function Header() {
  return (
    <header className="fixed top-0 left-0 right-0 z-50 glass-effect-dark border-b border-primary/20">
      <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
        <Link href="/" className="flex items-center gap-3 group">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-2xl font-bold bg-gradient-to-r from-primary via-accent to-secondary bg-clip-text text-transparent"
          >
            MLVerse
          </motion.div>
        </Link>

        <nav className="hidden md:flex items-center gap-8">
          <Link
            href="/algorithms"
            className="text-sm text-foreground/70 hover:text-primary transition-colors"
          >
            Algorithms
          </Link>
          <Link
            href="/playgrounds"
            className="text-sm text-foreground/70 hover:text-accent transition-colors"
          >
            Playgrounds
          </Link>
          <Link
            href="/guides"
            className="text-sm text-foreground/70 hover:text-secondary transition-colors"
          >
            Guides
          </Link>
        </nav>

        <button className="px-4 py-2 rounded-lg bg-primary/20 border border-primary/40 text-primary hover:bg-primary/30 transition-all text-sm font-medium">
          Explore
        </button>
      </div>
    </header>
  )
}
