'use client'

import { useRef, useState, useMemo, useId } from 'react'
import { useFrame } from '@react-three/fiber'
import { Html, Line } from '@react-three/drei'
import { Group } from 'three'
import * as THREE from 'three'

// ─── Types ────────────────────────────────────────────────────────────────────

export interface TreeNode {
  id: string          // Changed to string for better uniqueness
  numericalId: number // Keep numerical ID for tree logic
  x: number
  y: number
  z: number
  isLeaf: boolean
  label: string        // question text or class name
  feature?: string     // e.g. "Feature X"
  threshold?: number
  samples: number
  gini: number
  classes: { A: number; B: number; C: number }
  leftId?: string
  rightId?: string
}

export interface TreeEdge {
  from: string
  to: string
  label: string        // "≤ 5" or "> 5"
}

export interface Feature {
  name: string
  min: number
  max: number
  currentValue: number
}

// ─── Dynamic Tree Generator ─────────────────────────────────────────────────

function generateDynamicTree(features: Feature[], treeId: string): { nodes: TreeNode[]; edges: TreeEdge[] } {
  if (features.length === 0) {
    // Return a single leaf node when no features
    return {
      nodes: [{
        id: `${treeId}-node-0`,
        numericalId: 0,
        x: 0, y: 0, z: 0,
        isLeaf: true, label: 'No features available',
        samples: 0, gini: 0,
        classes: { A: 0, B: 0, C: 0 }
      }],
      edges: []
    }
  }

  const nodes: TreeNode[] = []
  const edges: TreeEdge[] = []
  
  // Helper to calculate positions based on depth
  const depthSpacing = 2.2
  const horizontalSpacing = 3.5
  
  function addNode(
    numericalId: number,
    depth: number,
    position: number,
    featureIndex: number,
    parentId?: string,
    edgeLabel?: string
  ): string {
    const isLeaf = featureIndex >= features.length
    const x = position * horizontalSpacing
    const y = 3 - depth * depthSpacing
    const z = 0
    const uniqueId = `${treeId}-node-${depth}-${numericalId}-${Date.now()}-${Math.random()}`
    
    let node: TreeNode
    
    if (isLeaf) {
      // Random sample distribution for demo
      const samples = Math.floor(Math.random() * 100) + 20
      const classA = Math.floor(Math.random() * samples)
      const classB = Math.floor(Math.random() * (samples - classA))
      const classC = samples - classA - classB
      const gini = 1 - Math.pow(classA/samples, 2) - Math.pow(classB/samples, 2) - Math.pow(classC/samples, 2)
      
      node = {
        id: uniqueId,
        numericalId,
        x, y, z,
        isLeaf: true,
        label: `Class ${classA > classB && classA > classC ? 'A' : classB > classC ? 'B' : 'C'}`,
        samples,
        gini,
        classes: { A: classA, B: classB, C: classC }
      }
    } else {
      const feature = features[featureIndex]
      const threshold = (feature.min + feature.max) / 2
      
      node = {
        id: uniqueId,
        numericalId,
        x, y, z,
        isLeaf: false,
        label: `${feature.name} > ${threshold.toFixed(1)}?`,
        feature: feature.name,
        threshold,
        samples: 100,
        gini: 0.66,
        classes: { A: 45, B: 25, C: 30 }
      }
    }
    
    nodes.push(node)
    
    // Create edge from parent
    if (parentId !== undefined && edgeLabel) {
      edges.push({ from: parentId, to: uniqueId, label: edgeLabel })
    }
    
    // Add children for non-leaf nodes
    if (!isLeaf) {
      const leftNumericalId = numericalId * 2 + 1
      const rightNumericalId = numericalId * 2 + 2
      
      // Recursively add children
      const nextIndex = featureIndex + 1
      const leftPos = position - 1 / Math.pow(2, depth + 1)
      const rightPos = position + 1 / Math.pow(2, depth + 1)
      
      const leftId = addNode(leftNumericalId, depth + 1, leftPos, nextIndex, uniqueId, `≤ ${node.threshold?.toFixed(1)}`)
      const rightId = addNode(rightNumericalId, depth + 1, rightPos, nextIndex, uniqueId, `> ${node.threshold?.toFixed(1)}`)
      
      node.leftId = leftId
      node.rightId = rightId
    }
    
    return uniqueId
  }
  
  addNode(0, 0, 0, 0)
  return { nodes, edges }
}

// ─── Colour helpers ───────────────────────────────────────────────────────────

const LEAF_COLOURS: Record<string, string> = {
  'Class A': '#639922',
  'Class B': '#D4537E',
  'Class C': '#EF9F27',
}

function nodeColour(node: TreeNode, isActive: boolean, isHighlighted: boolean): string {
  if (isHighlighted) return '#EF9F27'
  if (isActive) return '#185FA5'
  if (node.isLeaf) return LEAF_COLOURS[node.label] ?? '#639922'
  return '#378ADD'
}

// ─── Sub-components ───────────────────────────────────────────────────────────

interface NodeMeshProps {
  node: TreeNode
  isActive: boolean
  isHighlighted: boolean
  onClick: (node: TreeNode) => void
}

function NodeMesh({ node, isActive, isHighlighted, onClick }: NodeMeshProps) {
  const meshRef = useRef<THREE.Mesh>(null)
  const [hovered, setHovered] = useState(false)

  const color = nodeColour(node, isActive, isHighlighted)
  const radius = node.isLeaf ? 0.38 : 0.44
  const emissiveIntensity = hovered ? 0.6 : isActive || isHighlighted ? 0.5 : 0.25

  useFrame(() => {
    if (!meshRef.current) return
    const target = hovered ? radius * 1.12 : radius
    const curr = meshRef.current.scale.x
    meshRef.current.scale.setScalar(curr + (target - curr) * 0.12)
  })

  return (
    <group position={[node.x, node.y, node.z]}>
      <mesh
        ref={meshRef}
        onClick={(e) => { e.stopPropagation(); onClick(node) }}
        onPointerOver={() => setHovered(true)}
        onPointerOut={() => setHovered(false)}
      >
        <sphereGeometry args={[radius, 24, 24]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={emissiveIntensity}
          roughness={0.4}
          metalness={0.1}
        />
      </mesh>

      {/* Gini ring on decision nodes */}
      {!node.isLeaf && (
        <mesh rotation={[Math.PI / 2, 0, 0]}>
          <torusGeometry args={[radius + 0.12, 0.025, 8, 48, node.gini * Math.PI * 2]} />
          <meshBasicMaterial color={color} opacity={0.5} transparent />
        </mesh>
      )}

      {/* HTML label — always faces camera */}
      <Html
        center
        distanceFactor={6}
        style={{ pointerEvents: 'none', userSelect: 'none' }}
        position={[0, -(radius + 0.55), 0]}
      >
        <div
          style={{
            background: 'rgba(10,12,18,0.82)',
            border: `1px solid ${color}55`,
            borderRadius: 6,
            padding: '3px 8px',
            fontSize: 11,
            fontFamily: 'system-ui, sans-serif',
            color: '#fff',
            whiteSpace: 'nowrap',
            textAlign: 'center',
            lineHeight: 1.4,
          }}
        >
          <div style={{ fontWeight: 600, color }}>{node.label}</div>
          <div style={{ opacity: 0.7, fontSize: 10 }}>
            n={node.samples} · Gini={node.gini.toFixed(2)}
          </div>
        </div>
      </Html>
    </group>
  )
}

interface EdgeLineProps {
  from: TreeNode
  to: TreeNode
  edgeLabel: string
  isHighlighted: boolean
}

function EdgeLine({ from, to, edgeLabel, isHighlighted }: EdgeLineProps) {
  const midX = (from.x + to.x) / 2
  const midY = (from.y + to.y) / 2

  const points = useMemo<[number, number, number][]>(() => [
    [from.x, from.y, from.z],
    [to.x, to.y, to.z],
  ], [from, to])

  return (
    <>
      <Line
        points={points}
        color={isHighlighted ? '#EF9F27' : '#378ADD'}
        lineWidth={isHighlighted ? 2.5 : 1}
        opacity={isHighlighted ? 1 : 0.45}
        transparent
      />
      {/* Edge label */}
      <Html center position={[midX + 0.18, midY, 0]} distanceFactor={8} style={{ pointerEvents: 'none' }}>
        <div
          style={{
            fontSize: 10,
            fontFamily: 'system-ui, sans-serif',
            color: isHighlighted ? '#EF9F27' : '#aaa',
            background: 'rgba(10,12,18,0.6)',
            borderRadius: 3,
            padding: '1px 5px',
            whiteSpace: 'nowrap',
          }}
        >
          {edgeLabel}
        </div>
      </Html>
    </>
  )
}

// ─── Info panel (top-left overlay) ────────────────────────────────────────────

interface InfoPanelProps {
  node: TreeNode | null
}

function InfoPanel({ node }: InfoPanelProps) {
  if (!node) return null
  const total = node.classes.A + node.classes.B + node.classes.C

  return (
    <Html position={[-5.8, 3.2, 0]} distanceFactor={10} style={{ pointerEvents: 'none' }}>
      <div
        style={{
          background: 'rgba(10,12,18,0.88)',
          border: '1px solid #378ADD44',
          borderRadius: 8,
          padding: '10px 14px',
          minWidth: 180,
          fontFamily: 'system-ui, sans-serif',
          color: '#fff',
          fontSize: 12,
          lineHeight: 1.6,
        }}
      >
        <div style={{ fontWeight: 600, color: '#378ADD', marginBottom: 6 }}>
          {node.isLeaf ? '🍃 Leaf node' : '🔀 Decision node'}
        </div>
        <div style={{ opacity: 0.9 }}>{node.label}</div>
        <div style={{ opacity: 0.6, fontSize: 11, marginTop: 4 }}>
          Samples: {node.samples}
        </div>
        <div style={{ opacity: 0.6, fontSize: 11 }}>
          Gini impurity: {node.gini.toFixed(3)}
        </div>
        <div style={{ marginTop: 6, fontSize: 11 }}>
          {(['A', 'B', 'C'] as const).map((cls) => (
            <div key={cls} style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 2 }}>
              <div
                style={{
                  height: 5,
                  borderRadius: 3,
                  background: LEAF_COLOURS[`Class ${cls}`],
                  width: total > 0 ? `${(node.classes[cls] / total) * 80}px` : '0px',
                  transition: 'width 0.3s',
                  minWidth: 4,
                }}
              />
              <span style={{ opacity: 0.7 }}>
                Class {cls}: {node.classes[cls]}
              </span>
            </div>
          ))}
        </div>
      </div>
    </Html>
  )
}

// ─── Main scene ───────────────────────────────────────────────────────────────

interface DecisionTreeSceneProps {
  features?: Feature[]
  highlightedPath?: string[]   // Changed to string IDs
  activeNodeId?: string | null
  onNodeClick?: (node: TreeNode) => void
  autoRotate?: boolean
}

export function DecisionTreeScene({
  features = [],
  highlightedPath = [],
  activeNodeId = null,
  onNodeClick,
  autoRotate = true,
}: DecisionTreeSceneProps) {
  const groupRef = useRef<Group>(null)
  const [selectedNode, setSelectedNode] = useState<TreeNode | null>(null)
  const treeId = useId() // Generate unique ID for this tree instance

  // Generate dynamic tree based on features
  const { nodes, edges } = useMemo(() => generateDynamicTree(features, treeId), [features, treeId])
  const nodeMap = useMemo(() => new Map(nodes.map((n) => [n.id, n])), [nodes])

  useFrame((_, delta) => {
    if (groupRef.current && autoRotate) {
      groupRef.current.rotation.y += delta * 0.08
    }
  })

  function handleNodeClick(node: TreeNode) {
    setSelectedNode((prev) => (prev?.id === node.id ? null : node))
    onNodeClick?.(node)
  }

  // If no nodes, show placeholder
  if (nodes.length === 0) {
    return (
      <group>
        <Html center>
          <div style={{ color: 'white', background: 'rgba(0,0,0,0.7)', padding: '20px', borderRadius: '8px' }}>
            Add features to build your decision tree
          </div>
        </Html>
      </group>
    )
  }

  return (
    <group ref={groupRef}>
      <ambientLight intensity={0.5} />
      <pointLight position={[6, 6, 6]} intensity={1.2} color="#b0d4ff" />
      <pointLight position={[-6, -4, 4]} intensity={0.6} color="#ffd0a0" />
      <pointLight position={[0, -6, 2]} intensity={0.3} color="#ffffff" />

      {/* Edges */}
      {edges.map((edge) => {
        const fromNode = nodeMap.get(edge.from)
        const toNode = nodeMap.get(edge.to)
        if (!fromNode || !toNode) return null
        
        const isHighlighted =
          highlightedPath.includes(edge.from) && highlightedPath.includes(edge.to)
        return (
          <EdgeLine
            key={`${edge.from}-${edge.to}`}
            from={fromNode}
            to={toNode}
            edgeLabel={edge.label}
            isHighlighted={isHighlighted}
          />
        )
      })}

      {/* Nodes */}
      {nodes.map((node) => (
        <NodeMesh
          key={node.id}
          node={node}
          isActive={node.id === activeNodeId}
          isHighlighted={highlightedPath.includes(node.id)}
          onClick={handleNodeClick}
        />
      ))}

      {/* Selected node info panel */}
      <InfoPanel node={selectedNode} />
    </group>
  )
}