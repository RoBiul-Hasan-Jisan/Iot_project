'use client'

import { useRef, useState, useEffect } from 'react'
import { useFrame } from '@react-three/fiber'
import { Mesh, BufferGeometry, Points } from 'three'
import * as THREE from 'three'

interface Cluster {
  center: [number, number, number]
  points: [number, number, number][]
  color: string
}

export function KMeansScene({ iteration = 0 }) {
  const pointsRef = useRef<Mesh>(null)
  const centersRef = useRef<Mesh>(null)
  const [clusters, setClusters] = useState<Cluster[]>([])

  useEffect(() => {
    // Initialize clusters
    const k = 3
    const newClusters: Cluster[] = []
    const colors = ['#00ffff', '#ff00ff', '#00ff88']

    for (let i = 0; i < k; i++) {
      newClusters.push({
        center: [Math.random() * 6 - 3, Math.random() * 6 - 3, 0],
        points: [],
        color: colors[i],
      })
    }

    // Generate random points
    const pointCount = 150
    for (let i = 0; i < pointCount; i++) {
      const point: [number, number, number] = [
        Math.random() * 6 - 3,
        Math.random() * 6 - 3,
        0,
      ]

      // Assign to nearest cluster
      let nearest = 0
      let minDist = Infinity
      for (let j = 0; j < k; j++) {
        const dist =
          Math.pow(point[0] - newClusters[j].center[0], 2) +
          Math.pow(point[1] - newClusters[j].center[1], 2)
        if (dist < minDist) {
          minDist = dist
          nearest = j
        }
      }
      newClusters[nearest].points.push(point)
    }

    setClusters(newClusters)
  }, [iteration])

  useFrame(() => {
    if (pointsRef.current && centersRef.current) {
      pointsRef.current.rotation.z += 0.001
      centersRef.current.rotation.z += 0.001
    }
  })

  return (
    <>
      <ambientLight intensity={0.6} />
      <pointLight position={[5, 5, 5]} intensity={1} color="#00ffff" />
      <pointLight position={[-5, -5, 5]} intensity={0.8} color="#ff00ff" />

      {/* Data points */}
      {clusters.map((cluster, clusterIdx) => (
        <group key={clusterIdx} ref={clusterIdx === 0 ? pointsRef : null}>
          {cluster.points.map((point, pointIdx) => (
            <mesh key={`${clusterIdx}-${pointIdx}`} position={point}>
              <sphereGeometry args={[0.1, 8, 8]} />
              <meshBasicMaterial color={cluster.color} transparent opacity={0.7} />
            </mesh>
          ))}

          {/* Cluster center */}
          <mesh position={cluster.center} ref={clusterIdx === 0 ? centersRef : null}>
            <sphereGeometry args={[0.25, 8, 8]} />
            <meshPhongMaterial
              color={cluster.color}
              emissive={cluster.color}
              emissiveIntensity={0.8}
            />
          </mesh>
        </group>
      ))}
    </>
  )
}
