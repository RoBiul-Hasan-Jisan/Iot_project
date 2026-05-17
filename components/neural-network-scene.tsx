'use client'

import { useRef, useEffect, useState } from 'react'
import { useFrame } from '@react-three/fiber'
import { Mesh, Group, Line, LineBasicMaterial, BufferGeometry, Vector3 } from 'three'
import * as THREE from 'three'

interface NetworkState {
  inputActivation: number[]
  hiddenActivation: number[]
  outputActivation: number[]
}

export function NeuralNetworkScene({ input = [0.5, 0.5, 0.5] }: { input?: number[] }) {
  const groupRef = useRef<Group>(null)
  const [networkState, setNetworkState] = useState<NetworkState>({
    inputActivation: input,
    hiddenActivation: [0.5, 0.5, 0.5, 0.5],
    outputActivation: [0.5, 0.5],
  })

  useEffect(() => {
    // Simulate neural network forward pass
    const hidden = networkState.inputActivation.map((_, i) =>
      Math.tanh(
        input[0] * 0.5 +
          input[1] * 0.3 +
          input[2] * 0.2 -
          (i * 0.2) +
          Math.sin(Date.now() * 0.001 + i) * 0.1,
      ),
    )

    const output = [
      sigmoid(hidden.reduce((a, b, i) => a + b * (0.5 - i * 0.1), 0)),
      sigmoid(hidden.reduce((a, b, i) => a + b * (0.3 + i * 0.1), 0)),
    ]

    setNetworkState({
      inputActivation: input,
      hiddenActivation: hidden,
      outputActivation: output,
    })
  }, [input])

  useFrame(() => {
    if (groupRef.current) {
      groupRef.current.rotation.y += 0.005
    }
  })

  const layerPositions = [
    { x: -3, y: 0, neurons: 3 }, // Input
    { x: 0, y: 0, neurons: 4 }, // Hidden
    { x: 3, y: 0, neurons: 2 }, // Output
  ]

  const getNeuronPos = (layer: number, index: number, total: number) => {
    const y = ((index - (total - 1) / 2) * 1.5) / layerPositions[layer].neurons
    return [layerPositions[layer].x, y, 0] as const
  }

  return (
    <group ref={groupRef}>
      <ambientLight intensity={0.5} />
      <pointLight position={[5, 5, 5]} intensity={1.5} color="#00ffff" />
      <pointLight position={[-5, -5, 5]} intensity={0.8} color="#ff00ff" />

      {/* Connections - Input to Hidden */}
      {networkState.inputActivation.map((_, i) =>
        networkState.hiddenActivation.map((_, j) => {
          const from = getNeuronPos(0, i, 3)
          const to = getNeuronPos(1, j, 4)
          const alpha = Math.abs(
            Math.sin(Date.now() * 0.002 + i * j) * (networkState.inputActivation[i] * 0.5 + 0.3),
          )

          return (
            <line key={`0-${i}-${j}`}>
              <bufferGeometry>
                <bufferAttribute
                  attach="attributes-position"
                  count={2}
                  array={new Float32Array([...from, ...to])}
                  itemSize={3}
                />
              </bufferGeometry>
              <lineBasicMaterial color="#00ffff" opacity={alpha} transparent />
            </line>
          )
        }),
      )}

      {/* Connections - Hidden to Output */}
      {networkState.hiddenActivation.map((_, i) =>
        networkState.outputActivation.map((_, j) => {
          const from = getNeuronPos(1, i, 4)
          const to = getNeuronPos(2, j, 2)
          const alpha = Math.abs(
            Math.sin(Date.now() * 0.002 + (i + 4) * j) *
              (networkState.hiddenActivation[i] * 0.5 + 0.3),
          )

          return (
            <line key={`1-${i}-${j}`}>
              <bufferGeometry>
                <bufferAttribute
                  attach="attributes-position"
                  count={2}
                  array={new Float32Array([...from, ...to])}
                  itemSize={3}
                />
              </bufferGeometry>
              <lineBasicMaterial color="#ff00ff" opacity={alpha} transparent />
            </line>
          )
        }),
      )}

      {/* Input neurons */}
      {networkState.inputActivation.map((activation, i) => {
        const pos = getNeuronPos(0, i, 3)
        return (
          <mesh key={`input-${i}`} position={pos}>
            <sphereGeometry args={[0.25 + activation * 0.15, 16, 16]} />
            <meshPhongMaterial
              color="#00ff88"
              emissive="#00ff88"
              emissiveIntensity={activation}
            />
          </mesh>
        )
      })}

      {/* Hidden neurons */}
      {networkState.hiddenActivation.map((activation, i) => {
        const pos = getNeuronPos(1, i, 4)
        return (
          <mesh key={`hidden-${i}`} position={pos}>
            <sphereGeometry args={[0.25 + activation * 0.15, 16, 16]} />
            <meshPhongMaterial
              color="#00ffff"
              emissive="#0088ff"
              emissiveIntensity={activation}
            />
          </mesh>
        )
      })}

      {/* Output neurons */}
      {networkState.outputActivation.map((activation, i) => {
        const pos = getNeuronPos(2, i, 2)
        return (
          <mesh key={`output-${i}`} position={pos}>
            <sphereGeometry args={[0.25 + activation * 0.15, 16, 16]} />
            <meshPhongMaterial
              color="#ff00ff"
              emissive="#ff0088"
              emissiveIntensity={activation}
            />
          </mesh>
        )
      })}
    </group>
  )
}

function sigmoid(x: number) {
  return 1 / (1 + Math.exp(-x))
}
