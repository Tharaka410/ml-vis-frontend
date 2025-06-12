// ml-frontend/components/algorithms/som.tsx
"use client";

import { useState, useEffect, useRef, useCallback } from "react";

export default function SOMPage() {
  const [params, setParams] = useState({
    gridSize: 10,
    learningRate: 0.1,
    iterations: 100,
    sigma: 1.0,
  });

  const [currentFrame, setCurrentFrame] = useState(0);
  const [isAnimating, setIsAnimating] = useState(false);
  const animationFrameRef = useRef<number | null>(null); // Use a ref to store the animation frame ID
  const canvasRef = useRef<HTMLCanvasElement | null>(null); // Ref for the canvas element

  const paramControls = [
    {
      name: "gridSize",
      label: "Grid Size",
      min: 5,
      max: 20,
      step: 1,
    },
    {
      name: "learningRate",
      label: "Learning Rate",
      min: 0.01,
      max: 0.5,
      step: 0.01,
    },
    {
      name: "iterations",
      label: "Iterations",
      min: 10,
      max: 200,
      step: 10,
    },
    {
      name: "sigma",
      label: "Neighborhood Radius",
      min: 0.5,
      max: 3,
      step: 0.1,
    },
  ];

  // Fixed dataset for visualization (2D for 2D SOM)
  // Generating a ring-like distribution for better visualization of SOM's topological mapping
  const fixedData = Array.from({ length: 200 }, () => {
    const angle = Math.random() * 2 * Math.PI;
    const radius = 0.6 + Math.random() * 0.2; // Data points clustered in a ring
    return {
      x: Math.cos(angle) * radius,
      y: Math.sin(angle) * radius,
    };
  });

  // This function draws the SOM and data points on the canvas
  const renderSOM = useCallback(
    (
      ctx: CanvasRenderingContext2D,
      width: number,
      height: number,
      currentParams: typeof params,
      frame: number = 0
    ) => {
      const { gridSize, learningRate, iterations, sigma } = currentParams;

      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = "#0a0a0a"; // Dark background
      ctx.fillRect(0, 0, width, height);

      const margin = 40;
      const plotWidth = width - 2 * margin;
      const plotHeight = height - 2 * margin;

      // Coordinate conversion helpers: Normalize data [-1, 1] to canvas coordinates
      const toCanvas = (x: number, y: number) => {
        return {
          x: margin + ((x + 1) / 2) * plotWidth,
          y: margin + (1 - (y + 1) / 2) * plotHeight, // Invert Y-axis for standard plot view
        };
      };

      // Initialize SOM grid (weights) only once per simulation run or param change.
      // This is crucial: in this direct canvas approach, we re-simulate the training
      // for each frame to get the correct state at that `frame`.
      const grid: { x: number; y: number; i: number; j: number }[] = [];
      for (let i = 0; i < gridSize; i++) {
        for (let j = 0; j < gridSize; j++) {
          grid.push({
            x: Math.random() * 2 - 1, // [-1, 1]
            y: Math.random() * 2 - 1, // [-1, 1]
            i, // Grid row index
            j, // Grid column index
          });
        }
      }

      let lastBMU: (typeof grid[0]) | null = null;
      let lastBMUInputPoint: (typeof fixedData[0]) | null = null;

      // Simulate SOM training up to the current frame to get the state for this frame
      const effectiveIterations = Math.min(frame, iterations);

      for (let iter = 0; iter < effectiveIterations; iter++) {
        // Pick a random data point from the fixed dataset
        const point = fixedData[Math.floor(Math.random() * fixedData.length)];

        // Find BMU (Best Matching Unit)
        let bmuIndex = 0;
        let minDist = Infinity;
        for (let k = 0; k < grid.length; k++) {
          const node = grid[k];
          const dist = Math.hypot(point.x - node.x, point.y - node.y);
          if (dist < minDist) {
            minDist = dist;
            bmuIndex = k;
          }
        }

        lastBMU = grid[bmuIndex];
        lastBMUInputPoint = point; // Store the input point that caused this BMU

        // Calculate decaying learning rate and neighborhood radius
        const lrDecay = learningRate * Math.exp(-iter / iterations);
        const sigmaDecay = sigma * Math.exp(-iter / iterations);

        // Update BMU and its neighbors
        for (let k = 0; k < grid.length; k++) {
          const node = grid[k];
          const gridDist = Math.hypot(node.i - lastBMU.i, node.j - lastBMU.j); // Distance in the SOM grid
          const influence = Math.exp(-(gridDist ** 2) / (2 * sigmaDecay ** 2));

          node.x += lrDecay * influence * (point.x - node.x);
          node.y += lrDecay * influence * (point.y - node.y);
        }
      }

      // --- Drawing ---

      // Draw Axes and Bounding Box
      ctx.strokeStyle = "#444"; // Darker gray
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.rect(margin, margin, plotWidth, plotHeight); // Bounding box
      ctx.moveTo(width / 2, margin); // Y-axis
      ctx.lineTo(width / 2, height - margin);
      ctx.moveTo(margin, height / 2); // X-axis
      ctx.lineTo(width - margin, height / 2);
      ctx.stroke();

      // Draw data points
      for (const point of fixedData) {
        const { x, y } = toCanvas(point.x, point.y);
        ctx.fillStyle = "rgba(96, 165, 250, 0.7)"; // Blue, slightly transparent
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, 2 * Math.PI);
        ctx.fill();
      }

      // Draw grid connections
      ctx.strokeStyle = "rgba(255,255,255,0.4)"; // Faint white lines
      ctx.lineWidth = 1;
      for (let i = 0; i < gridSize; i++) {
        for (let j = 0; j < gridSize; j++) {
          const idx = i * gridSize + j;
          const node = grid[idx];
          const nodeCanvas = toCanvas(node.x, node.y);

          // Connect to right neighbor
          if (j < gridSize - 1) {
            const right = grid[i * gridSize + j + 1];
            const rightCanvas = toCanvas(right.x, right.y);
            ctx.beginPath();
            ctx.moveTo(nodeCanvas.x, nodeCanvas.y);
            ctx.lineTo(rightCanvas.x, rightCanvas.y);
            ctx.stroke();
          }

          // Connect to bottom neighbor
          if (i < gridSize - 1) {
            const down = grid[(i + 1) * gridSize + j];
            const downCanvas = toCanvas(down.x, down.y);
            ctx.beginPath();
            ctx.moveTo(nodeCanvas.x, nodeCanvas.y);
            ctx.lineTo(downCanvas.x, downCanvas.y);
            ctx.stroke();
          }
        }
      }

      // Draw neurons (nodes)
      for (const node of grid) {
        const { x, y } = toCanvas(node.x, node.y);
        ctx.fillStyle = "#ff3366"; // Pink for regular nodes
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, 2 * Math.PI);
        ctx.fill();
      }

      // Highlight BMU with green circle and pulse
      if (lastBMU) {
        const { x, y } = toCanvas(lastBMU.x, lastBMU.y);
        ctx.strokeStyle = "#22c55e"; // Green for BMU
        ctx.lineWidth = 3;
        const pulseSize = 6 + Math.sin(frame * 0.1) * 2; // Simple pulsing effect
        ctx.beginPath();
        ctx.arc(x, y, pulseSize, 0, 2 * Math.PI);
        ctx.stroke();
      }

      // Highlight the input data point that activated the BMU
      if (lastBMUInputPoint) {
        const { x, y } = toCanvas(lastBMUInputPoint.x, lastBMUInputPoint.y);
        ctx.fillStyle = "#FFD700"; // Gold for BMU input point
        ctx.beginPath();
        ctx.arc(x, y, 5, 0, 2 * Math.PI);
        ctx.fill();

        // Draw a dashed line from BMU input point to BMU node
        if (lastBMU) {
            const bmuCanvas = toCanvas(lastBMU.x, lastBMU.y);
            ctx.strokeStyle = "#FFD700"; // Gold line
            ctx.lineWidth = 1.5;
            ctx.setLineDash([5, 5]); // Dashed line
            ctx.beginPath();
            ctx.moveTo(x, y);
            ctx.lineTo(bmuCanvas.x, bmuCanvas.y);
            ctx.stroke();
            ctx.setLineDash([]); // Reset line dash
        }
      }
    },
    [] // Dependencies: fixedData is constant, so no need to include it in the deps array.
  );

  // Effect to draw on canvas whenever params or currentFrame changes
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    renderSOM(ctx, canvas.width, canvas.height, params, currentFrame);
  }, [params, currentFrame, renderSOM]); // Re-render when params or frame changes

  // Animation loop
  useEffect(() => {
    let lastFrameTime = 0;
    const frameInterval = 16; // approximately 60fps

    const animate = (timestamp: number) => {
      const elapsed = timestamp - lastFrameTime;
      if (elapsed > frameInterval) {
        lastFrameTime = timestamp;

        setCurrentFrame(prev => {
          const next = prev + 1;
          if (next <= params.iterations) {
            animationFrameRef.current = requestAnimationFrame(animate); // Continue animating
            return next;
          } else {
            setIsAnimating(false); // Stop animation when complete
            return params.iterations; // Cap at max iterations
          }
        });
      } else {
         animationFrameRef.current = requestAnimationFrame(animate); // Request next frame immediately if not enough time passed
      }
    };

    if (isAnimating && currentFrame < params.iterations) {
      animationFrameRef.current = requestAnimationFrame(animate);
    } else if (!isAnimating && animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
    }

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [isAnimating, currentFrame, params.iterations]);


  const handleAnimateToggle = () => {
    if (currentFrame >= params.iterations) {
      // If animation completed, restart from beginning
      setCurrentFrame(0);
      setIsAnimating(true);
    } else {
      // Otherwise toggle animation state
      setIsAnimating(prev => !prev);
    }
  };

  return (
    <div className="space-y-8 bg-black text-white p-4 min-h-screen">
      <div>
        <h1 className="text-4xl font-bold mb-4">Self-Organizing Map (2D Visualization)</h1>
        <p className="text-lg text-gray-400 mb-6">
          A type of artificial neural network that maps high-dimensional data onto a low-dimensional (2D) grid while preserving topological relationships.
        </p>
      </div>

      <div className="flex flex-col lg:flex-row gap-8">
        {/* Left Section: Canvas and Controls */}
        <div className="lg:w-3/5 flex flex-col gap-4"> {/* Use flex-col to stack canvas and controls */}
          {/* Canvas Section */}
          <div className="relative h-[600px] border border-gray-700 rounded-lg overflow-hidden">
            <canvas
              ref={canvasRef}
              width={1000} // Fixed width for the canvas
              height={600} // Fixed height for the canvas
              className="w-full max-w-full h-auto"
              style={{ aspectRatio: `1000/600` }} // Maintain aspect ratio
            />

            <div className="absolute bottom-4 left-4 flex gap-2">
              <button
                onClick={handleAnimateToggle}
                className="px-4 py-2 bg-black text-white rounded border border-white hover:bg-white hover:text-black transition"
              >
                {isAnimating ? "Pause Animation" : "Start Animation"}
              </button>
              {/* Iteration slider for manual control */}
              <input
                  type="range"
                  min={0}
                  max={params.iterations}
                  step={1}
                  value={currentFrame}
                  onChange={(e) => {
                      setIsAnimating(false); // Pause animation when manually moving slider
                      setCurrentFrame(Number(e.target.value));
                  }}
                  className="w-48 ml-4 slider"
                  style={{ "--value": `${(currentFrame / params.iterations) * 100}%` } as React.CSSProperties}
              />
            </div>

            {/* Progress indicator */}
            <div className="absolute bottom-4 right-4 w-1/3">
              <div className="w-full bg-gray-700 rounded-full h-2.5">
                <div
                  className="bg-blue-600 h-2.5 rounded-full"
                  style={{ width: `${(currentFrame / params.iterations) * 100}%` }}
                ></div>
              </div>
              <p className="text-sm text-gray-400 mt-1 text-right">
                Progress: {currentFrame}/{params.iterations} iterations
              </p>
            </div>
          </div>

          {/* Controls Section - Moved below the graph */}
          <section className="controls-section p-4 bg-black-900">
            <h3 className="text-xl font-semibold text-white mb-4">Controls</h3>

            {paramControls.map((ctl) => {
              const val = params[ctl.name as keyof typeof params];
              const pct = ((Number(val) - ctl.min) / (ctl.max - ctl.min)) * 100;
              return (
                <div key={ctl.name} className="control-row">
                  <span className="control-label">{ctl.label}</span>
                  <input
                    type="range"
                    min={ctl.min}
                    max={ctl.max}
                    step={ctl.step}
                    value={val}
                    onChange={(e) =>
                      setParams((p) => ({
                        ...p,
                        [ctl.name]: Number(e.target.value),
                      }))
                    }
                    className="slider"
                    style={{ "--value": `${pct}%` } as React.CSSProperties}
                  />
                  <span className="control-value">{val}</span>
                </div>
              );
            })}
          </section>
        </div>

        {/* Right Section: Text Content */}
        <div className="space-y-6 lg:w-2/5">
          <div>
            <h2 className="text-2xl font-semibold mb-2">How It Works</h2>
            <p className="text-gray-400">
              Self-Organizing Maps (SOMs) are a type of artificial neural network that transform complex,
              high-dimensional data into a simpler low-dimensional representation while preserving the topological
              structure of the data. Here, we visualize how a 2D map organizes itself to represent 2D input data.
            </p>
          </div>

          <div>
            <h3 className="text-xl font-semibold mb-2">Key Concepts</h3>
            <ul className="list-disc list-inside space-y-2 text-gray-400">
              <li>
                <strong>Competitive Learning:</strong> Neurons compete to be activated, with only one winning neuron
                (Best Matching Unit or BMU) being activated for each input.
              </li>
              <li>
                <strong>Best Matching Unit (BMU):</strong> The neuron whose weights are most similar to the input
                vector.
              </li>
              <li>
                <strong>Neighborhood Function:</strong> Determines how much neurons around the BMU are adjusted. Its influence decays over time.
              </li>
              <li>
                <strong>Topology Preservation:</strong> Similar inputs activate neurons that are close to each other in
                the grid, allowing the map to capture the underlying structure of the data.
              </li>
            </ul>
          </div>

          <div>
            <h3 className="text-xl font-semibold mb-2">Algorithm Steps (Iterative Training)</h3>
            <ol className="list-decimal list-inside space-y-2 text-gray-400">
              <li>Initialize a grid of neurons with random weights (positions in data space).</li>
              <li>For each input vector, find the BMU (neuron with weights most similar to the input).</li>
              <li>Update the BMU and its neighbors to make them more similar to the input vector. The amount of update depends on learning rate and neighborhood influence.</li>
              <li>Gradually reduce the learning rate and neighborhood radius over time (annealing).</li>
              <li>Repeat until convergence or a maximum number of iterations.</li>
            </ol>
          </div>

          <div>
            <h3 className="text-xl font-semibold mb-2">Color Guide</h3>
            <ul className="list-disc list-inside space-y-2 text-gray-400">
              <li><span className="text-blue-400">Blue Circles:</span> Input data points.</li>
              <li><span className="text-pink-400">Pink Circles:</span> SOM neurons (nodes of the 2D grid).</li>
              <li><span className="text-green-400">Green Pulsing Circle:</span> Best Matching Unit (BMU) for the current input data point.</li>
              <li><span className="text-yellow-400">Gold Circle:</span> The specific input data point that was chosen to train the current iteration's BMU.</li>
              <li><span className="text-white">Faint White Lines:</span> Connections forming the SOM grid.</li>
              <li><span className="text-yellow-400">Gold Dashed Line:</span> Connection from the selected input data point to its BMU.</li>
              <li><span className="text-gray-600">Gray Lines/Box:</span> Background axes and plot boundaries.</li>
            </ul>
          </div>

          <div>
            <h3 className="text-xl font-semibold mb-2">Applications</h3>
            <p className="text-gray-400">SOMs are widely used in:</p>
            <ul className="list-disc list-inside space-y-2 text-gray-400">
              <li>Data visualization and dimensionality reduction </li>
              <li>Cluster analysis and pattern recognition</li>
              <li>Image and speech processing</li>
              <li>Bioinformatics (gene expression analysis)</li>
              <li>Fraud detection</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Global styles */}
      <style jsx global>{`
        .controls-section {
          width: 100%;
          max-width: 800px; /* Adjust max-width to fit under the canvas */
          margin-left: auto;
          margin-right: auto;
        }
        .control-row {
          display: flex;
          align-items: center;
          margin-bottom: 1rem;
        }
        .control-label {
          width: 150px;
          color: white;
          font-weight: 500;
        }
        .control-value {
          width: 40px;
          text-align: right;
          color: white;
          margin-left: 0.5rem;
        }
        .slider {
          -webkit-appearance: none;
          appearance: none;
          width: 100%;
          height: 8px;
          border-radius: 4px;
          background: linear-gradient(
            to right,
            #ff3860 0%,
            #ff3860 var(--value),
            #333333 var(--value),
            #333333 100%
          );
          margin: 0 0.75rem;
        }
        .slider::-moz-range-track {
          background: transparent;
        }
        .slider::-webkit-slider-thumb {
          -webkit-appearance: none;
          appearance: none;
          width: 18px;
          height: 18px;
          border-radius: 50%;
          background: black;
          border: 2px solid #ff3860;
          margin-top: -5px;
          cursor: pointer;
        }
        .slider::-moz-range-thumb {
          width: 18px;
          height: 18px;
          border-radius: 50%;
          background: #ff3860;
          border: 2px solid white;
          cursor: pointer;
        }
      `}</style>
    </div>
  );
}