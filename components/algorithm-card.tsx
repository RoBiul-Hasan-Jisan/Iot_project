'use client'

import { motion } from 'framer-motion'
import Link from 'next/link'
import { ArrowRight, Code2, Zap } from 'lucide-react'
import { ReactNode } from 'react'

interface AlgorithmCardProps {
  id: string
  name: string
  description: string
  category: 'ml' | 'dl' | 'genai'
  complexity: 'beginner' | 'intermediate' | 'advanced'
  icon: ReactNode
}

const categoryColors = {
  ml: { bg: 'bg-blue-500/10', text: 'text-blue-400', border: 'border-blue-500/20' },
  dl: { bg: 'bg-purple-500/10', text: 'text-purple-400', border: 'border-purple-500/20' },
  genai: { bg: 'bg-pink-500/10', text: 'text-pink-400', border: 'border-pink-500/20' },
}

const complexityColors = {
  beginner: 'text-accent',
  intermediate: 'text-secondary',
  advanced: 'text-primary',
}

const categoryLabels = {
  ml: 'Machine Learning',
  dl: 'Deep Learning',
  genai: 'Gen AI',
}

export function AlgorithmCard({
  id,
  name,
  description,
  category,
  complexity,
  icon,
}: AlgorithmCardProps) {
  const colors = categoryColors[category]

  return (
    <Link href={`/algorithms/${id}`}>
      <motion.div
        whileHover={{ y: -4, scale: 1.02 }}
        className={`glass-effect p-6 rounded-xl border border-primary/10 hover:border-primary/40 transition-all cursor-pointer h-full flex flex-col`}
      >
        <div className="flex items-start justify-between mb-4">
          <div className={`${colors.bg} ${colors.text} p-3 rounded-lg`}>{icon}</div>
          <span className={`${colors.text} text-xs font-semibold px-2 py-1 rounded ${colors.bg}`}>
            {categoryLabels[category]}
          </span>
        </div>

        <h3 className="text-lg font-semibold text-foreground mb-2">{name}</h3>
        <p className="text-sm text-foreground/60 mb-4 flex-grow">{description}</p>

        <div className="flex items-center justify-between pt-4 border-t border-primary/10">
          <span className={`text-xs font-medium capitalize ${complexityColors[complexity]}`}>
            {complexity}
          </span>
          <motion.div whileHover={{ translate: 2 }}>
            <ArrowRight className="w-4 h-4 text-primary" />
          </motion.div>
        </div>
      </motion.div>
    </Link>
  )
}
