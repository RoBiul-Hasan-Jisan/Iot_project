'use client'

import { useState, useCallback, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, PerspectiveCamera } from '@react-three/drei'
import { ChevronLeft, ChevronRight, RotateCcw, Play, Pause, Plus, Trash2 } from 'lucide-react'
import Link from 'next/link'

import { DecisionTreeScene, type Feature } from '@/components/decision-tree-scene'
import { Header } from '@/components/header'

// ─── Types ────────────────────────────────────────────────────────────────────

interface FeatureValues {
  [key: string]: number
}

// ─── Dynamic Prediction Logic ─────────────────────────────────────────────────

function predictPathDynamic(
  features: Feature[],
  featureValues: FeatureValues,
  treeId: string
): { path: string[]; steps: string[]; prediction: string } {
  const path: string[] = []
  const steps: string[] = []
  
  if (features.length === 0) {
    return { path, steps, prediction: 'No features available' }
  }

  let depth = 0
  let numericalId = 0

  while (depth < features.length) {
    const nodeId = `${treeId}-node-${depth}-${numericalId}`
    path.push(nodeId)
    
    const feature = features[depth]
    const value = featureValues[feature.name] ?? (feature.min + feature.max) / 2
    const threshold = (feature.min + feature.max) / 2
    
    if (value > threshold) {
      steps.push(`${feature.name} = ${value.toFixed(1)} > ${threshold.toFixed(1)} → right branch`)
      numericalId = numericalId * 2 + 2
    } else {
      steps.push(`${feature.name} = ${value.toFixed(1)} ≤ ${threshold.toFixed(1)} → left branch`)
      numericalId = numericalId * 2 + 1
    }
    depth++
  }
  
  // Add leaf node to path
  const leafId = `${treeId}-node-${depth}-${numericalId}`
  path.push(leafId)

  // Determine prediction based on path (simplified demo logic)
  const predictions = ['Class A', 'Class B', 'Class C']
  const prediction = predictions[numericalId % predictions.length]
  
  return { path, steps, prediction }
}

// ─── Code snippet ─────────────────────────────────────────────────────────────

const CODE = `from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Prepare data (dynamic features based on your selection)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
tree = DecisionTreeClassifier(
    max_depth=5,           # limit depth → reduce overfitting
    min_samples_split=10,  # min samples to split a node
    criterion='gini'       # or 'entropy' for information gain
)
tree.fit(X_train, y_train)

# Feature importance
importances = tree.feature_importances_
for name, importance in zip(feature_names, importances):
    print(f"{name}: {importance:.3f}")`

// ─── Sub-components ───────────────────────────────────────────────────────────

function SliderRow({
  label,
  value,
  min,
  max,
  onChange,
  onRemove,
  showRemove = false,
}: {
  label: string
  value: number
  min: number
  max: number
  onChange: (v: number) => void
  onRemove?: () => void
  showRemove?: boolean
}) {
  return (
    <div className="flex items-center gap-3 group">
      <span className="w-24 text-sm text-foreground/60 flex-shrink-0">Feature {label}</span>
      <input
        type="range"
        min={min}
        max={max}
        step={0.5}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="flex-1 accent-primary"
      />
      <span className="w-8 text-right text-sm font-medium text-foreground">
        {value.toFixed(1)}
      </span>
      {showRemove && onRemove && (
        <button
          onClick={onRemove}
          className="opacity-0 group-hover:opacity-100 transition-opacity p-1 hover:bg-red-500/20 rounded"
        >
          <Trash2 className="w-4 h-4 text-red-400" />
        </button>
      )}
    </div>
  )
}

function AddFeatureForm({ onAdd }: { onAdd: (name: string, min: number, max: number) => void }) {
  const [name, setName] = useState('')
  const [min, setMin] = useState(0)
  const [max, setMax] = useState(10)

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (name.trim()) {
      onAdd(name.trim(), min, max)
      setName('')
      setMin(0)
      setMax(10)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-3">
      <input
        type="text"
        placeholder="Feature name (e.g., Temperature)"
        value={name}
        onChange={(e) => setName(e.target.value)}
        className="w-full px-3 py-2 rounded-lg border border-primary/20 bg-background/50 text-foreground text-sm focus:outline-none focus:border-primary"
      />
      <div className="flex gap-3">
        <div className="flex-1">
          <label className="text-xs text-foreground/50">Min</label>
          <input
            type="number"
            value={min}
            onChange={(e) => setMin(parseFloat(e.target.value))}
            className="w-full px-2 py-1 rounded border border-primary/20 bg-background/50 text-foreground text-sm"
          />
        </div>
        <div className="flex-1">
          <label className="text-xs text-foreground/50">Max</label>
          <input
            type="number"
            value={max}
            onChange={(e) => setMax(parseFloat(e.target.value))}
            className="w-full px-2 py-1 rounded border border-primary/20 bg-background/50 text-foreground text-sm"
          />
        </div>
      </div>
      <button
        type="submit"
        className="w-full flex items-center justify-center gap-2 px-4 py-2 rounded-lg bg-primary/20 hover:bg-primary/30 text-primary transition-colors text-sm"
      >
        <Plus className="w-4 h-4" />
        Add Feature
      </button>
    </form>
  )
}

const PREDICTION_COLOURS: Record<string, string> = {
  'Class A': 'text-green-400',
  'Class B': 'text-pink-400',
  'Class C': 'text-amber-400',
}

// ─── Build Steps (dynamic based on features) ─────────────────────────────────

function getBuildSteps(featureCount: number) {
  return [
    {
      title: 'Calculate impurity',
      body: 'Measure how mixed the classes are at the root using Gini impurity: 1 − Σpᵢ². A pure node (one class only) scores 0; a 50/50 split scores 0.5.',
      highlight: [],
    },
    {
      title: 'Find the best split',
      body: `Test every feature (${featureCount} feature${featureCount > 1 ? 's' : ''}) at every threshold. Compute the weighted Gini of the two child groups. Pick the feature + threshold with the biggest impurity reduction.`,
      highlight: [],
    },
    {
      title: 'Split the node',
      body: 'Samples with feature ≤ threshold go left; samples with feature > threshold go right.',
      highlight: [],
    },
    {
      title: 'Recurse on children',
      body: `Apply the same process independently to each child. The tree grows to depth ${featureCount}, using one feature per level.`,
      highlight: [],
    },
    {
      title: 'Assign leaf predictions',
      body: 'When all features are used or a stopping criterion is met, the node becomes a leaf. It predicts the majority class of all training samples that landed there.',
      highlight: [],
    },
  ]
}

export default function DecisionTreePage() {
  // Generate a unique ID for this page instance
  const pageId = useMemo(() => Math.random().toString(36).substring(7), [])
  
  // Dynamic features state
  const [features, setFeatures] = useState<Feature[]>([
    { name: 'X', min: 0, max: 10, currentValue: 6 },
    { name: 'Y', min: 0, max: 10, currentValue: 4 },
    { name: 'Z', min: 0, max: 10, currentValue: 3 },
  ])
  
  // Feature values for prediction
  const [featureValues, setFeatureValues] = useState<FeatureValues>({
    X: 6, Y: 4, Z: 3
  })

  // Build-step tab state
  const [step, setStep] = useState(0)
  const [autoRotate, setAutoRotate] = useState(true)

  // Which panel is shown alongside the 3D view
  type Panel = 'predict' | 'build' | 'code'
  const [panel, setPanel] = useState<Panel>('predict')

  const predResult = predictPathDynamic(features, featureValues, pageId)
  const buildSteps = getBuildSteps(features.length)

  const highlightedPath: string[] = panel === 'predict' ? predResult.path : []

  const updateFeatureValue = useCallback(
    (featureName: string) => (v: number) => {
      setFeatureValues((prev) => ({ ...prev, [featureName]: v }))
      // Update current value in features array
      setFeatures((prev) =>
        prev.map((f) => (f.name === featureName ? { ...f, currentValue: v } : f))
      )
    },
    []
  )

  const addFeature = useCallback((name: string, min: number, max: number) => {
    const newFeature: Feature = { name, min, max, currentValue: (min + max) / 2 }
    setFeatures((prev) => [...prev, newFeature])
    setFeatureValues((prev) => ({ ...prev, [name]: (min + max) / 2 }))
  }, [])

  const removeFeature = useCallback((featureName: string) => {
    setFeatures((prev) => prev.filter((f) => f.name !== featureName))
    setFeatureValues((prev) => {
      const newValues = { ...prev }
      delete newValues[featureName]
      return newValues
    })
  }, [])

  return (
    <main className="min-h-screen bg-background">
      <Header />

      <div className="pt-20">
        {/* Back */}
        <div className="px-6 py-4">
          <Link
            href="/algorithms"
            className="inline-flex items-center gap-2 text-primary hover:text-primary/80 transition-colors text-sm"
          >
            <ChevronLeft className="w-4 h-4" />
            Back to algorithms
          </Link>
        </div>

        <div className="grid lg:grid-cols-2 gap-0 px-6 pb-20">
          {/* ── 3-D canvas ── */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="h-[420px] lg:h-screen lg:sticky lg:top-0 rounded-xl overflow-hidden border border-primary/20 bg-black/30"
          >
            <Canvas dpr={[1, 1.5]} gl={{ antialias: true }}>
              <PerspectiveCamera makeDefault position={[0, 0, 11]} fov={50} />
              <OrbitControls
                enablePan={false}
                minDistance={7}
                maxDistance={18}
                autoRotate={false}
              />
              <DecisionTreeScene
                features={features}
                highlightedPath={highlightedPath}
                autoRotate={autoRotate}
              />
            </Canvas>

            {/* Canvas controls */}
            <div className="absolute bottom-3 right-3 flex gap-2">
              <button
                onClick={() => setAutoRotate((r) => !r)}
                className="p-2 rounded-lg bg-black/50 border border-white/10 text-white/70 hover:text-white transition-colors"
                title={autoRotate ? 'Pause rotation' : 'Resume rotation'}
              >
                {autoRotate ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
              </button>
            </div>
          </motion.div>

          {/* ── Right panel ── */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="lg:pl-8 lg:py-16 space-y-8"
          >
            {/* Heading */}
            <div>
              <h1 className="text-4xl lg:text-5xl font-bold mb-3 bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
                Decision Trees
              </h1>
              <p className="text-foreground/60 text-base leading-relaxed">
                Interpretable models that recursively split data on feature thresholds. Each
                internal node asks a yes/no question; each leaf gives a prediction. Works for
                both classification and regression.
              </p>
              <div className="mt-3 text-sm text-foreground/50">
                Currently using <span className="text-primary font-semibold">{features.length}</span> feature{features.length !== 1 ? 's' : ''}
              </div>
            </div>

            {/* Panel switcher */}
            <div className="flex gap-2 border-b border-primary/10 pb-0">
              {(['predict', 'build', 'code'] as Panel[]).map((p) => (
                <button
                  key={p}
                  onClick={() => setPanel(p)}
                  className={`px-4 py-2 text-sm font-medium transition-colors border-b-2 -mb-px ${
                    panel === p
                      ? 'border-primary text-primary'
                      : 'border-transparent text-foreground/50 hover:text-foreground/80'
                  }`}
                >
                  {p === 'predict' ? 'Predict' : p === 'build' ? 'How it is built' : 'Python code'}
                </button>
              ))}
            </div>

            {/* ── Predict panel ── */}
            <AnimatePresence mode="wait">
              {panel === 'predict' && (
                <motion.div
                  key="predict"
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -8 }}
                  className="space-y-5"
                >
                  <p className="text-sm text-foreground/60">
                    Add features below, then drag the sliders to set their values. The tree
                    dynamically updates and highlights the prediction path.
                  </p>

                  {/* Add feature form */}
                  <div className="glass-effect border border-primary/20 rounded-xl p-4">
                    <h3 className="text-sm font-medium text-foreground/70 mb-3">Add New Feature</h3>
                    <AddFeatureForm onAdd={addFeature} />
                  </div>

                  {/* Existing features sliders */}
                  {features.length > 0 && (
                    <div className="glass-effect border border-primary/20 rounded-xl p-5 space-y-4">
                      <div className="flex justify-between items-center">
                        <h3 className="text-sm font-medium text-foreground/70">Feature Values</h3>
                        <span className="text-xs text-foreground/40">
                          {features.length} feature{features.length !== 1 ? 's' : ''}
                        </span>
                      </div>
                      {features.map((feature) => (
                        <SliderRow
                          key={feature.name}
                          label={feature.name}
                          value={featureValues[feature.name] ?? feature.currentValue}
                          min={feature.min}
                          max={feature.max}
                          onChange={updateFeatureValue(feature.name)}
                          onRemove={() => removeFeature(feature.name)}
                          showRemove={features.length > 1}
                        />
                      ))}
                    </div>
                  )}

                  {/* Steps */}
                  {predResult.steps.length > 0 && (
                    <div className="space-y-2">
                      <h3 className="text-sm font-medium text-foreground/70">Decision Path</h3>
                      {predResult.steps.map((s, i) => (
                        <motion.div
                          key={i}
                          initial={{ opacity: 0, x: 10 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: i * 0.05 }}
                          className="flex items-start gap-3 text-sm text-foreground/70"
                        >
                          <span className="mt-0.5 w-5 h-5 rounded-full bg-primary/20 text-primary flex items-center justify-center text-xs flex-shrink-0">
                            {i + 1}
                          </span>
                          {s}
                        </motion.div>
                      ))}
                    </div>
                  )}

                  {/* Result */}
                  {features.length > 0 && (
                    <div className="glass-effect border border-primary/20 rounded-xl p-5 flex items-center gap-4">
                      <div className="flex-1">
                        <p className="text-xs text-foreground/50 mb-1">Prediction</p>
                        <p
                          className={`text-2xl font-bold ${
                            PREDICTION_COLOURS[predResult.prediction] ?? 'text-foreground'
                          }`}
                        >
                          {predResult.prediction}
                        </p>
                      </div>
                      <div className="text-right">
                        <p className="text-xs text-foreground/50 mb-1">Path length</p>
                        <p className="text-2xl font-bold text-foreground">
                          {predResult.steps.length}
                        </p>
                        <p className="text-xs text-foreground/40">splits</p>
                      </div>
                    </div>
                  )}

                  {features.length === 0 && (
                    <div className="text-center py-8 text-foreground/40">
                      <p>Add features above to start building your decision tree</p>
                    </div>
                  )}
                </motion.div>
              )}

              {/* ── Build panel ── */}
              {panel === 'build' && (
                <motion.div
                  key="build"
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -8 }}
                  className="space-y-5"
                >
                  <p className="text-sm text-foreground/60">
                    Step through how the tree is grown. The 3D view highlights the relevant
                    nodes at each stage.
                  </p>

                  {/* Step card */}
                  <AnimatePresence mode="wait">
                    <motion.div
                      key={step}
                      initial={{ opacity: 0, y: 6 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -6 }}
                      className="glass-effect border border-primary/20 rounded-xl p-6"
                    >
                      <div className="flex items-center gap-3 mb-3">
                        <span className="w-7 h-7 rounded-full bg-primary/30 border border-primary/50 text-primary text-sm font-bold flex items-center justify-center">
                          {step + 1}
                        </span>
                        <h3 className="font-semibold text-foreground">
                          {buildSteps[step].title}
                        </h3>
                      </div>
                      <p className="text-sm text-foreground/70 leading-relaxed">
                        {buildSteps[step].body}
                      </p>
                    </motion.div>
                  </AnimatePresence>

                  {/* Step controls */}
                  <div className="flex items-center gap-3">
                    <button
                      onClick={() => setStep((s) => Math.max(0, s - 1))}
                      disabled={step === 0}
                      className="flex items-center gap-1.5 px-4 py-2 rounded-lg border border-primary/20 text-sm text-foreground/70 hover:text-foreground hover:border-primary/40 disabled:opacity-30 disabled:cursor-not-allowed transition-all"
                    >
                      <ChevronLeft className="w-4 h-4" />
                      Previous
                    </button>
                    <div className="flex-1 flex justify-center gap-1.5">
                      {buildSteps.map((_, i) => (
                        <button
                          key={i}
                          onClick={() => setStep(i)}
                          className={`w-2 h-2 rounded-full transition-all ${
                            i === step ? 'bg-primary w-4' : 'bg-primary/25 hover:bg-primary/50'
                          }`}
                        />
                      ))}
                    </div>
                    <button
                      onClick={() => setStep((s) => Math.min(buildSteps.length - 1, s + 1))}
                      disabled={step === buildSteps.length - 1}
                      className="flex items-center gap-1.5 px-4 py-2 rounded-lg border border-primary/20 text-sm text-foreground/70 hover:text-foreground hover:border-primary/40 disabled:opacity-30 disabled:cursor-not-allowed transition-all"
                    >
                      Next
                      <ChevronRight className="w-4 h-4" />
                    </button>
                  </div>

                  <button
                    onClick={() => setStep(0)}
                    className="flex items-center gap-2 text-xs text-foreground/40 hover:text-foreground/60 transition-colors"
                  >
                    <RotateCcw className="w-3 h-3" />
                    Restart
                  </button>
                </motion.div>
              )}

              {/* ── Code panel ── */}
              {panel === 'code' && (
                <motion.div
                  key="code"
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -8 }}
                  className="space-y-5"
                >
                  {/* Complexity badges */}
                  <div className="flex flex-wrap gap-2">
                    {[
                      { label: 'Train', value: 'O(n·d·log n)' },
                      { label: 'Predict', value: 'O(depth)' },
                      { label: 'Space', value: 'O(nodes)' },
                    ].map(({ label, value }) => (
                      <span
                        key={label}
                        className="text-xs px-3 py-1 rounded-full border border-primary/20 text-foreground/60"
                      >
                        {label}: <span className="font-mono text-primary">{value}</span>
                      </span>
                    ))}
                  </div>

                  <div className="glass-effect border border-primary/20 rounded-xl overflow-hidden">
                    <div className="flex items-center justify-between px-4 py-2.5 border-b border-primary/10">
                      <span className="text-xs text-foreground/50 font-mono">
                        sklearn · decision_tree.py
                      </span>
                      <span className="text-xs text-foreground/30">Python</span>
                    </div>
                    <pre className="p-4 overflow-x-auto text-xs leading-relaxed text-accent font-mono bg-background/50">
                      {CODE}
                    </pre>
                  </div>

                  {/* Key advantages */}
                  <div className="space-y-2">
                    <p className="text-xs font-medium text-foreground/50 uppercase tracking-wider">
                      Key advantages
                    </p>
                    {[
                      'No feature scaling needed',
                      'Handles mixed feature types (numerical + categorical)',
                      'Built-in feature importance ranking',
                      'Fully interpretable — you can read every rule',
                      `Supports ${features.length} custom feature${features.length !== 1 ? 's' : ''}`,
                    ].map((a, i) => (
                      <div key={i} className="flex gap-2.5 text-sm text-foreground/70">
                        <span className="text-accent mt-0.5 flex-shrink-0">✓</span>
                        {a}
                      </div>
                    ))}
                  </div>

                  {/* Pitfalls */}
                  <div className="space-y-2">
                    <p className="text-xs font-medium text-foreground/50 uppercase tracking-wider">
                      Watch out for
                    </p>
                    {[
                      'Overfitting on deep, unconstrained trees — use max_depth',
                      'High variance: small data changes can flip the whole structure',
                      'Poor extrapolation outside training range',
                    ].map((a, i) => (
                      <div key={i} className="flex gap-2.5 text-sm text-foreground/70">
                        <span className="text-red-400 mt-0.5 flex-shrink-0">!</span>
                        {a}
                      </div>
                    ))}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        </div>
      </div>
    </main>
  )
}