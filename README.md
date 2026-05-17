# MLVerse - Interactive Machine Learning Education Platform

MLVerse is a beautiful, interactive platform for learning machine learning algorithms through stunning 3D visualizations and hands-on playgrounds. Explore fundamental concepts in an engaging, visual way.

## Features

### 🎨 Interactive 3D Visualizations
- **Decision Trees**: See how trees recursively split data to make decisions
- **K-Means Clustering**: Watch unsupervised learning group similar data points
- **Neural Networks**: Visualize neurons, layers, and forward propagation
- **Linear Regression**: See how algorithms fit lines to data with residuals
- **And More**: Additional algorithms coming soon

### 🎮 Interactive Playgrounds
- Tune hyperparameters and see results in real-time
- Experiment with neural network architectures
- Explore how learning rates affect optimization
- Build and test decision trees interactively

### 📚 Comprehensive Guides
- ML fundamentals and core concepts
- Deep dives into specific algorithms
- Feature engineering best practices
- Model selection strategies
- Code examples and implementations

## Technology Stack

- **Frontend**: Next.js 16 with React 19
- **3D Graphics**: Three.js with React Three Fiber
- **Animations**: Framer Motion
- **Styling**: Tailwind CSS with custom cyberpunk theme
- **Icons**: Lucide React

## Getting Started

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd mlverse

# Install dependencies
pnpm install

# Start the development server
pnpm dev
```

The application will be available at `http://localhost:3000`.

## Project Structure

```
/app
  /algorithms         # Algorithm detail pages
    /decision-tree
    /linear-regression
    /k-means
  /playgrounds        # Interactive playground pages
  /guides             # Educational content
  layout.tsx          # Root layout
  page.tsx            # Homepage
  globals.css         # Global styles and design tokens

/components
  header.tsx                    # Navigation header
  canvas-wrapper.tsx            # 3D canvas setup
  hero-scene.tsx               # Homepage hero visualization
  decision-tree-scene.tsx      # Decision tree 3D scene
  linear-regression-scene.tsx  # Linear regression visualization
  kmeans-scene.tsx             # K-Means clustering visualization
  neural-network-scene.tsx     # Neural network visualization
  algorithm-card.tsx           # Reusable algorithm card component
```

## Design System

The platform uses a cyberpunk aesthetic with neon accents:

- **Primary Color**: Neon Cyan (`oklch(0.65 0.25 200)`)
- **Secondary Color**: Neon Magenta (`oklch(0.6 0.28 310)`)
- **Accent Color**: Neon Lime (`oklch(0.75 0.25 130)`)
- **Background**: Dark Blue (`oklch(0.08 0 0)`)

All colors use OKLCH color space for consistent luminosity and saturation.

## Planned Features

### Phase 1 (MVP - Current)
- Core algorithm visualizations
- Basic playgrounds
- Learning guides

### Phase 2
- More algorithms (SVM, Random Forest, etc.)
- Advanced hyperparameter tuning
- Dataset upload and training

### Phase 3
- Model comparison tools
- Performance profiling challenges
- Community features
- Dark/Light mode toggle

## Performance Optimizations

- Lazy-loaded 3D scenes
- Optimized shader rendering
- React component memoization
- Canvas DPR adjustment for high-DPI displays

## Browser Support

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+
- Requires WebGL support

## Contributing

Contributions are welcome! Areas for contribution:
- New algorithm visualizations
- Additional playgrounds
- Educational content
- Bug fixes and optimizations

## License

This project is open source and available under the MIT License.

## Feedback & Support

Have ideas, found bugs, or want to contribute? Please create an issue or reach out!

---

**MLVerse**: Making machine learning beautiful and accessible through interactive visualization.
