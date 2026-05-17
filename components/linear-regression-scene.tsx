'use client'

import { useRef, useEffect, useState } from 'react'
import { useFrame } from '@react-three/fiber'
import { Mesh, Group } from 'three'
import * as THREE from 'three'

interface RegressionState {
  points: [number, number, number][]
  line: { start: [number, number, number]; end: [number, number, number] }
}

export function LinearRegressionScene({ learningRate = 0.01, iteration = 0 }: { learningRate?: number; iteration?: number }) {
  const groupRef = useRef<Group>(null)
  const [state, setState] = useState<RegressionState>({
    points: [],
    line: { start: [-3, -2, 0], end: [3, 2, 0] },
  })

  useEffect(() => {
    // Generate random points around a line
    const points: [number, number, number][] = []
    const trueSlope = 0.6
    const trueIntercept = 0.3

    for (let i = 0; i < 50; i++) {
      const x = Math.random() * 6 - 3
      const y = trueSlope * x + trueIntercept + (Math.random() - 0.5) * 1.2
      points.push([x, y, 0])
    }

    // Calculate best fit line with noise based on iteration
    let slope = 0.3 + (learningRate * iteration * 0.5) % 0.6
    let intercept = 0.1 + Math.sin(iteration * 0.05) * 0.2

    const line = {
      start: [-3, slope * -3 + intercept, 0] as [number, number, number],
      end: [3, slope * 3 + intercept, 0] as [number, number, number],
    }

    setState({ points, line })
  }, [learningRate, iteration])

  useFrame(() => {
    if (groupRef.current) {
      groupRef.current.rotation.z += 0.001
    }
  })

  return (
    <group ref={groupRef}>
      <ambientLight intensity={0.6} />
      <pointLight position={[5, 5, 5]} intensity={1.5} color="#00ffff" />
      <pointLight position={[-5, -5, 5]} intensity={0.8} color="#ff00ff" />

      {/* Grid background */}
      {[...Array(7)].map((_, i) => {
        const pos = -3 + i * 1
        return (
          <group key={`grid-${i}`}>
            {/* Vertical lines */}
            <line>
              <bufferGeometry>
                <bufferAttribute
                  attach="attributes-position"
                  count={2}
                  array={new Float32Array([pos, -3.5, -0.1, pos, 3.5, -0.1])}
                  itemSize={3}
                />
              </bufferGeometry>
              <lineBasicMaterial color="#00ffff" transparent opacity={0.1} />
            </line>
            {/* Horizontal lines */}
            <line>
              <bufferGeometry>
                <bufferAttribute
                  attach="attributes-position"
                  count={2}
                  array={new Float32Array([-3.5, pos, -0.1, 3.5, pos, -0.1])}
                  itemSize={3}
                />
              </bufferGeometry>
              <lineBasicMaterial color="#00ffff" transparent opacity={0.1} />
            </line>
          </group>
        )
      })}

      {/* Regression line */}
      <line>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={2}
            array={new Float32Array([...state.line.start, ...state.line.end])}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial color="#ff00ff" linewidth={3} opacity={0.8} transparent />
      </line>

      {/* Data points */}
      {state.points.map((point, idx) => {
        const yPred = (state.line.end[1] - state.line.start[1]) / (state.line.end[0] - state.line.start[0]) * point[0] +
          state.line.start[1]
        const residual = Math.abs(point[1] - yPred)

        return (
          <group key={`point-${idx}`}>
            {/* Residual line */}
            <line>
              <bufferGeometry>
                <bufferAttribute
                  attach="attributes-position"
                  count={2}
                  array={new Float32Array([point[0], point[1], point[2], point[0], yPred, point[2]])}
                  itemSize={3}
                />
              </bufferGeometry>
              <lineBasicMaterial color="#ff0088" transparent opacity={0.3} />
            </line>

            {/* Data point */}
            <mesh position={point}>
              <sphereGeometry args={[0.15, 8, 8]} />
              <meshBasicMaterial
                color="#00ff88"
                transparent
                opacity={0.8}
              />
            </mesh>
          </group>
        )
      })}

      {/* Axes */}
      <group>
        {/* X axis */}
        <line>
          <bufferGeometry>
            <bufferAttribute
              attach="attributes-position"
              count={2}
              array={new Float32Array([-3.5, 0, 0, 3.5, 0, 0])}
              itemSize={3}
            />
          </bufferGeometry>
          <lineBasicMaterial color="#00ffff" opacity={0.5} />
        </line>

        {/* Y axis */}
        <line>
          <bufferGeometry>
            <bufferAttribute
              attach="attributes-position"
              count={2}
              array={new Float32Array([0, -3.5, 0, 0, 3.5, 0])}
              itemSize={3}
            />
          </bufferGeometry>
          <lineBasicMaterial color="#00ffff" opacity={0.5} />
        </line>
      </group>
    </group>
  )
}
