'use client'

import { Canvas } from '@react-three/fiber'
import { useEffect, useState } from 'react'

interface CanvasWrapperProps {
  children: React.ReactNode
  className?: string
  dpr?: number
}

export function CanvasWrapper({ children, className = '', dpr = 1.5 }: CanvasWrapperProps) {
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  if (!mounted) {
    return <div className={`bg-background ${className}`} />
  }

  return (
    <Canvas
      dpr={dpr}
      gl={{ antialias: true, alpha: true }}
      camera={{ position: [0, 0, 8], fov: 50 }}
      className={className}
    >
      {children}
    </Canvas>
  )
}
