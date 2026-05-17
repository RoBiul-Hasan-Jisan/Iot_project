'use client'

import { useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import { Mesh } from 'three'
import * as THREE from 'three'

export function HeroScene() {
  const meshRef = useRef<Mesh>(null)
  const torusRef = useRef<Mesh>(null)
  const dotsRef = useRef<Mesh>(null)

  useFrame(() => {
    if (meshRef.current) {
      meshRef.current.rotation.x += 0.002
      meshRef.current.rotation.y += 0.003
    }
    if (torusRef.current) {
      torusRef.current.rotation.z -= 0.001
      torusRef.current.rotation.x += 0.0005
    }
    if (dotsRef.current) {
      dotsRef.current.rotation.y += 0.0015
    }
  })

  return (
    <>
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} intensity={1.5} color="#00ffff" />
      <pointLight position={[-10, -10, 10]} intensity={1} color="#ff00ff" />
      <pointLight position={[0, 0, 10]} intensity={0.8} color="#00ff88" />

      {/* Central cube */}
      <mesh ref={meshRef}>
        <boxGeometry args={[2, 2, 2]} />
        <meshPhongMaterial
          color="#00ffff"
          emissive="#0088ff"
          wireframe={true}
          transparent={true}
          opacity={0.8}
        />
      </mesh>

      {/* Rotating torus */}
      <mesh ref={torusRef} position={[0, 0, 0]}>
        <torusGeometry args={[3, 0.3, 16, 100]} />
        <meshPhongMaterial
          color="#ff00ff"
          emissive="#ff0088"
          wireframe={true}
          transparent={true}
          opacity={0.6}
        />
      </mesh>

      {/* Orbiting particles */}
      <mesh ref={dotsRef}>
        {[...Array(20)].map((_, i) => {
          const angle = (i / 20) * Math.PI * 2
          const radius = 4
          const x = Math.cos(angle) * radius
          const z = Math.sin(angle) * radius
          const y = Math.sin(angle * 2) * 1.5

          return (
            <mesh key={i} position={[x, y, z]}>
              <sphereGeometry args={[0.15, 8, 8]} />
              <meshBasicMaterial color="#00ff88" />
            </mesh>
          )
        })}
      </mesh>
    </>
  )
}
