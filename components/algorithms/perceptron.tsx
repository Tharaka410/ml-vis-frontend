"use client";

import { useEffect, useRef, useState, useMemo, useCallback } from "react";
import { BlockMath } from "react-katex";
import "katex/dist/katex.min.css";

// --- Type Definitions ---
type Layer = { weights: number[][]; biases: number[] };

type ActivationFunction = {
  label: string;
  func: (x: number) => number; // Keeping for compatibility with non-softmax functions
  derivative: (y: number) => number;
  formula: string;
};

type Activations = {
  [key: string]: ActivationFunction;
};

// --- Constants ---
const activations: Activations = {
  sigmoid: {
    label: "Sigmoid",
    func: (x: number) => 1 / (1 + Math.exp(-x)),
    derivative: (y: number) => y * (1 - y),
    formula: "\\sigma(x) = \\frac{1}{1 + e^{-x}}",
  },
  relu: {
    label: "ReLU",
    func: (x: number) => Math.max(0, x),
    derivative: (y: number) => (y > 0 ? 1 : 0),
    formula: "\\text{ReLU}(x) = \\max(0, x)",
  },
  identity: {
    label: "Identity",
    func: (x: number) => x,
    derivative: (_: number) => 1,
    formula: "\\phi(x) = x",
  },
  softmax: { // Added Softmax
    label: "Softmax",
    func: (x: number) => Math.exp(x), // Simplified representation for scalar display
    derivative: (_: number) => 1, // Placeholder derivative for display purposes
    formula: "\\text{Softmax}(z_i) = \\frac{e^{z_i}}{\\sum_j e^{z_j}}",
  },
};

const BASE_URL = "http://localhost:8000";

// --- Helper for Range Input Style ---
interface SliderStyle extends React.CSSProperties {
  "--value": string;
}

// --- Main Component ---
export default function PerceptronVisualizer() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number | null>(null); // Initialize with null for animation frame ID
  // ADDED: useRef to hold the latest networkState without triggering useEffect
  const networkRef = useRef<Layer[]>([]); 

  const [config, setConfig] = useState({
    hiddenLayers: 2,
    iterations: 10,
    activation: "sigmoid",
    learningRate: 0.1,
    outputNodes: 1,
  });
  const [networkState, setNetworkState] = useState<Layer[]>([]);
  const [currentActivations, setCurrentActivations] = useState<number[][]>([]);
  const [currentDeltas, setCurrentDeltas] = useState<number[][]>([]);
  const [isNetworkInitialized, setIsNetworkInitialized] = useState(false);
  const [isClient, setIsClient] = useState(false);

  // Constants that do not change and thus don't need to be in state or dependencies if used directly
  const inputSize = 2;
  const hiddenSize = 3;
  const input = useMemo(() => [0.5, -0.3], []); // Input values for the network
  const target = useMemo(() => Array(config.outputNodes).fill(1), [config.outputNodes]); // Target output for training

  // Mark component as mounted on client after hydration
  useEffect(() => {
    setIsClient(true);
  }, []);

  // ADDED: Effect to keep networkRef.current in sync with networkState
  useEffect(() => {
    networkRef.current = networkState;
  }, [networkState]);

  const initializeNetwork = useCallback(async (currentConfig: typeof config) => {
    setIsNetworkInitialized(false);
    try {
        const response = await fetch(`${BASE_URL}/initialize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
  inputSize: inputSize,
  hiddenSize: hiddenSize,
  outputNodes: currentConfig.outputNodes,
  hiddenLayers: currentConfig.hiddenLayers,
  activation: currentConfig.activation,
  learningRate: currentConfig.learningRate, // <-- ADD THIS
}),
});
        if (!response.ok) {
            throw new Error(`Network response was not ok: ${response.statusText}`);
        }
        const data = await response.json();
        console.log("FRONTEND: Initial network layers from /initialize:", data);

        setNetworkState(data);
        networkRef.current = data; 

        const initialActivations: number[][] = [];
        initialActivations.push(input); // [0] = Input layer activations

        setCurrentActivations(initialActivations);
        const backendLayerCount = currentConfig.hiddenLayers + 1;
        setCurrentDeltas(Array(backendLayerCount).fill(null).map(() => [])); 
        for (let i = 0; i < backendLayerCount; i++) {
  const layerSize = data[i].biases.length;
  initialActivations.push(Array(layerSize).fill(0)); // z
  initialActivations.push(Array(layerSize).fill(0)); // a
}

        setIsNetworkInitialized(true);
    } catch (error) {
        console.error("Error initializing network:", error);
        setIsNetworkInitialized(false);
    }
}, [input]);

  const draw = useCallback((layers: Layer[], acts: number[][], deltas: number[][]) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const w = (canvas.width = 800);
    const h = (canvas.height = 620);
    const nodeRadius = 25; 
    const marginX = 50; 
    ctx.clearRect(0, 0, w, h);

    const totalVisualLayers = config.hiddenLayers + 2; 
    const drawingWidth = w - marginX * 2;
    const xGap = totalVisualLayers > 1 ? drawingWidth / (totalVisualLayers - 1) : 0;

    // Pre-calculate positions for all nodes in each *visual* layer
    const positions: { x: number; y: number }[][] = [];
    for (let li = 0; li < totalVisualLayers; li++) {
      let layerNodeCount: number;
      let activationsToDisplay: number[];

      if (li === 0) { // Input layer
        layerNodeCount = input.length;
        activationsToDisplay = acts[0]; // acts[0] is the input activations
      } else { 
        const backendLayerIndex = li - 1; // 0 for first hidden, 1 for second, etc.
        const actsIndexForActivatedOutput = (2 * backendLayerIndex) + 2; 
        if (acts[actsIndexForActivatedOutput]) {
            layerNodeCount = acts[actsIndexForActivatedOutput].length;
            activationsToDisplay = acts[actsIndexForActivatedOutput];
        } else {
            console.warn(`Activations for visual layer ${li} (backend layer ${backendLayerIndex}) not found at acts[${actsIndexForActivatedOutput}].`);
            layerNodeCount = 0;
            activationsToDisplay = [];
        }
      }

      const yGap = h / (layerNodeCount + 1); // Vertical spacing for nodes within a layer
      const layerPositions: { x: number; y: number }[] = [];
      for (let ni = 0; ni < layerNodeCount; ni++) {
        layerPositions.push({
          x: marginX + li * xGap,
          y: (ni + 1) * yGap,
        });
      }
      positions.push(layerPositions);
    }

    ctx.font = "11px monospace";
    ctx.textAlign = "center";

    for (let l = 0; l < layers.length; l++) {
      const currentBackendLayer = layers[l];      
      const visualInputLayerIndex = l;
      const visualOutputLayerIndex = l + 1;

      if (!positions[visualInputLayerIndex] || !positions[visualOutputLayerIndex]) {
          console.warn(`Missing positions for drawing connections between visual layers ${visualInputLayerIndex} and ${visualOutputLayerIndex}.`);
          continue; // Skip drawing connections for this layer if positions are missing
      }

      const nextLayerDeltas = deltas[l]; 

      positions[visualInputLayerIndex].forEach((p0, i) => // p0 is the position of a node in the current visual layer
        positions[visualOutputLayerIndex].forEach((p1, j) => { // p1 is the position of a node in the next visual layer
          ctx.strokeStyle = "rgba(255,255,255,0.3)";
          ctx.beginPath();
          ctx.moveTo(p0.x, p0.y);
          ctx.lineTo(p1.x, p1.y);
          ctx.stroke();

          const mx = (p0.x + p1.x) / 2;
          const my = (p0.y + p1.y) / 2;
          const dx = p1.x - p0.x;
          const dy = p1.y - p0.y;
          const len = Math.hypot(dx, dy) || 1; 
          const offsetBase = 20; // Base offset distance
          const offsetFactor = (j - i) * 10; // Dynamic offset based on node indices
          const ox = (-dy / len) * (offsetBase + offsetFactor);
          const oy = (dx / len) * (offsetBase + offsetFactor);

          ctx.fillStyle = "white";
          if (currentBackendLayer.weights && currentBackendLayer.weights[j] && currentBackendLayer.weights[j][i] !== undefined) {
             ctx.fillText(currentBackendLayer.weights[j][i].toFixed(2), mx + ox, my + oy);
          } else {
             ctx.fillText("N/A", mx + ox, my + oy); // Fallback for missing weight
          }

          if (nextLayerDeltas && l < deltas.length && nextLayerDeltas[j] !== undefined) {
            const deltaValue = nextLayerDeltas[j];
            const deltaStrength = Math.abs(deltaValue);

            const arrowLength = 15;
            const arrowHeadSize = 5;

            const angle = Math.atan2(p0.y - p1.y, p0.x - p1.x);
            const startArrowX = p1.x + nodeRadius * Math.cos(angle);
            const startArrowY = p1.y + nodeRadius * Math.sin(angle);

            ctx.save();
            ctx.translate(startArrowX, startArrowY);
            ctx.rotate(angle);

            ctx.beginPath();
            ctx.moveTo(0, 0);
            ctx.lineTo(arrowLength, 0);
            ctx.lineTo(arrowLength - arrowHeadSize, arrowHeadSize);
            ctx.moveTo(arrowLength, 0);
            ctx.lineTo(arrowLength - arrowHeadSize, -arrowHeadSize);

            ctx.strokeStyle = deltaValue > 0 ? `rgba(0, 255, 0, ${Math.min(1, deltaStrength * 5)})` : `rgba(255, 0, 0, ${Math.min(1, deltaStrength * 5)})`;
            ctx.lineWidth = 2;
            ctx.stroke();
            ctx.restore();
          }
        })
      );

      if (!positions[visualOutputLayerIndex]) {
          console.warn(`Missing positions for drawing biases for visual layer ${visualOutputLayerIndex}.`);
          continue; // Skip drawing biases for this layer
      }

      const nextLayerNodeCount = positions[visualOutputLayerIndex].length;
      const midPointIndex = (nextLayerNodeCount - 1) / 2;
      const yGapForNextLayer = h / (nextLayerNodeCount + 1); 

      positions[visualOutputLayerIndex].forEach((p, j) => { // Iterate through nodes of the next visual layer
        const biasOffsetX = (j - midPointIndex) * 50; // Horizontal offset for bias node
        const biasOffsetY = yGapForNextLayer / 2 + 15; // Vertical offset for bias node

        const bx = p.x + biasOffsetX;
        const by = p.y - biasOffsetY;

        ctx.strokeStyle = "orange"; // Color for bias connections
        ctx.beginPath();
        ctx.moveTo(bx, by);
        ctx.lineTo(p.x, p.y); // Draw line from bias node to the neuron
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(bx, by, nodeRadius * 0.6, 0, 2 * Math.PI); // Draw bias node (smaller circle)
        ctx.fillStyle = "orange";
        ctx.fill();
        ctx.strokeStyle = "white";
        ctx.stroke();

        ctx.fillStyle = "white";
        if (currentBackendLayer.biases && currentBackendLayer.biases[j] !== undefined) {
          ctx.fillText(currentBackendLayer.biases[j].toFixed(2), bx, by + 4);
        } else {
          ctx.fillText("N/A", bx, by + 4); // Fallback for missing bias
        }
      });

      if (l < deltas.length) {
        positions[visualOutputLayerIndex].forEach((p, j) => {
          if (deltas[l] && deltas[l][j] !== undefined) {
            const deltaValue = deltas[l][j];
            ctx.fillStyle = deltaValue > 0 ? "lightgreen" : "salmon";
            ctx.fillText(`Δ: ${deltaValue.toFixed(4)}`, p.x, p.y + nodeRadius + 15);
          }
        });
      }
    }

    positions.forEach((layerPos, visualLayerIndex) => {
      if (!layerPos) return; // Add check to prevent forEach on undefined
      layerPos.forEach((p, nodeIndex) => {
        const color =
          visualLayerIndex === 0
            ? "#22c55e"
            : visualLayerIndex === totalVisualLayers - 1
            ? "#f87171"
            : "#0ea5e9";

        ctx.beginPath();
        ctx.arc(p.x, p.y, nodeRadius, 0, 2 * Math.PI);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.strokeStyle = "white";
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.fillStyle = "white";
        
        let activationValue: number = 0; // Initialize with default
        if (visualLayerIndex === 0) { // Input layer
            if (acts[0] && acts[0][nodeIndex] !== undefined) {
                activationValue = acts[0][nodeIndex];
            }
        } else { // Hidden and Output layers: get activated 'a' value
            const backendLayerIndex = visualLayerIndex - 1;
            const actsIndexForActivatedOutput = (2 * backendLayerIndex) + 2;
            if (acts[actsIndexForActivatedOutput] && acts[actsIndexForActivatedOutput][nodeIndex] !== undefined) {
                activationValue = acts[actsIndexForActivatedOutput][nodeIndex];
            }
        }

        ctx.fillText(activationValue.toFixed(2), p.x, p.y + 6);
      });
    });

    // --- Display Mean Squared Error (MSE) ---
    // The final activations (an) from the backend are at the very end of the acts array.
    const outputActivations = acts[acts.length - 1];

    let loss = 0;
    if (outputActivations && outputActivations.length > 0) {
        loss = outputActivations.reduce((sum, v, i) => sum + (v - target[i]) ** 2, 0) / outputActivations.length;
    } else {
        console.warn("outputActivations is undefined or empty, cannot calculate MSE.");
        loss = Infinity;
    }

    ctx.fillStyle = "white";
    ctx.font = "17px sans-serif";
    ctx.textAlign = "end";
    ctx.fillText(`MSE: ${loss.toFixed(4)}`, w - 20, h - 20); 
    
  }, [target, config.hiddenLayers, input]); // 'target', 'config.hiddenLayers', and 'input' are dependencies.

  // Effect to initialize the network when config or client status changes
  useEffect(() => {
    if (isClient) {
      initializeNetwork(config);
    }
  }, [isClient, config]); 

  useEffect(() => {
    if (animRef.current) {
      cancelAnimationFrame(animRef.current);
    }

    if (!isNetworkInitialized || networkState.length === 0 || !isClient) {
      return;
    }

    let currentIterations = 0;

    const step = async () => {
      if (currentIterations >= config.iterations) {
        console.log("Training complete for this configuration.");
        return;
      }

      try {
        const response = await fetch(`${BASE_URL}/train`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
    config: {
        inputSize: inputSize, // Explicitly pass inputSize
        hiddenSize: hiddenSize, // Explicitly pass hiddenSize
        outputNodes: config.outputNodes,
        hiddenLayers: config.hiddenLayers,
        activation: config.activation,
        learningRate: config.learningRate, // Include learningRate
        // Remove momentum and batchSize
    },
    input: [input],
    target: [target],
    network: networkRef.current,
}),
        });
        if (!response.ok) {
          throw new Error(`Training network response was not ok: ${response.statusText}`);
        }
        const data = await response.json();

        setNetworkState(data.network);
        const constructedActivations: number[][] = [input, ...data.full_activations];
        setCurrentActivations(constructedActivations);
        setCurrentDeltas(data.deltas);

        currentIterations++;

        if (currentIterations < config.iterations) {
          animRef.current = requestAnimationFrame(step);
        }
      } catch (error) {
        console.error("Error during training step:", error);
        setIsNetworkInitialized(false);
      }
    };

    animRef.current = requestAnimationFrame(step);

    return () => {
      if (animRef.current) {
        cancelAnimationFrame(animRef.current);
      }
    };
  }, [
    isClient,
    config.activation,
    config.iterations,
    config.learningRate,
    config.hiddenLayers,
    config.outputNodes,
    target,
    isNetworkInitialized,
    input,
    hiddenSize // hiddenSize is not directly used in this effect after the fix, but harmless to keep
  ]);

  useEffect(() => {
    // Only draw if initialized, network data exists, current activations exist, and on the client
    if (isNetworkInitialized && networkState.length > 0 && currentActivations.length > 0 && isClient) {
      draw(networkState, currentActivations, currentDeltas);
    }
  }, [isNetworkInitialized, networkState, currentActivations, currentDeltas, draw, isClient]);

  // Component render
  return (
    <div className="flex p-6">
      <div className="flex-shrink-0">
        <h2 className="text-3xl font-bold text-white mb-4">MLP Visualizer</h2>
        {/* Only render canvas and controls if on client to avoid hydration mismatch */}
        {isClient ? (
          <>
            <div className="p-4 bg-black border rounded w-[860px]">
              <canvas ref={canvasRef} className="w-full" />
            </div>

            <div className="mt-4 w-[880px] space-y-2">
              <div className="controls-section">
                {[
                  {
                    label: "Hidden Layers",
                    key: "hiddenLayers",
                    min: 0, // Adjusted min to 0 to allow direct input to output
                    max: 5,
                    step: 1,
                    value: config.hiddenLayers,
                    // Calculation for slider percentage should be robust for min > 0
                    percent: ((config.hiddenLayers - 0) / (5 - 0)) * 100,
                  },
                  {
                    label: "Iterations",
                    key: "iterations",
                    min: 1,
                    max: 20,
                    step: 1,
                    value: config.iterations,
                    percent: ((config.iterations - 1) / (20 - 1)) * 100,
                  },
                  {
                    label: "Learning Rate",
                    key: "learningRate",
                    min: 0.01,
                    max: 1,
                    step: 0.01,
                    value: config.learningRate,
                    percent: ((config.learningRate - 0.01) / (1 - 0.01)) * 100,
                  },
                  {
                    label: "Output Nodes",
                    key: "outputNodes",
                    min: 1,
                    max: 5,
                    step: 1,
                    value: config.outputNodes,
                    percent: ((config.outputNodes - 1) / (5 - 1)) * 100,
                  },
                ].map(({ label, key, min, max, step, value, percent }) => (
                  <div className="control-row" key={key}>
                    <label className="control-label">{label}</label>
                    <input
                      type="range"
                      min={min}
                      max={max}
                      step={step}
                      value={value}
                      className="slider"
                      onChange={(e) => setConfig((prev) => ({ ...prev, [key]: +e.target.value }))}
                      style={{ "--value": `${percent}%` } as SliderStyle}
                    />
                    <span className="control-value">
                      {key === "learningRate" ? value.toFixed(2) : value}
                    </span>
                  </div>
                ))}

                <div className="control-row">
                  <label className="control-label">Activation</label>
                  <select
                    className="styled-select"
                    value={config.activation}
                    onChange={(e) => setConfig((prev) => ({ ...prev, activation: e.target.value }))}
                  >
                    {Object.entries(activations).map(([key, { label }]) => (
                      <option key={key} value={key}>
                        {label}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            </div>
          </>
        ) : (
          <div className="p-4 bg-black border rounded w-[860px] h-[620px] flex items-center justify-center text-white">
            Loading visualizer...
          </div>
        )}
      </div>

      <div className="ml-8 w-1/3 space-y-4">
        <h3 className="text-2xl font-semibold text-white">What is a Perceptron?</h3>
        <p className="text-gray-300">
          A perceptron is the simplest artificial neural network for binary classification,
          computing a weighted sum plus bias and applying an activation function.
        </p>
        <h3 className="text-xl font-semibold text-white">Formula</h3>
        <BlockMath math="y = \phi\left(\sum_i w_i x_i + b\right)" />
        <h3 className="text-xl font-semibold text-white">Activation Functions</h3>
        {Object.entries(activations).map(([key, { label, formula }]) => (
          <div key={key} className="text-gray-300">
            <strong>{label}:</strong>
            <BlockMath math={formula} />
          </div>
        ))}
        <h3 className="text-xl font-semibold text-white">Data Types</h3>
        <ul className="list-disc list-inside text-gray-300">
          <li>Binary inputs</li>
          <li>Continuous features</li>
          <li>Labels {"{-1,+1}"} or {"{0,1}"}</li>
        </ul>
        <h3 className="text-xl font-semibold text-white">Controls</h3>
        <p className="text-gray-300">
          <strong>Hidden Layers:</strong> Depth (0–5)<br />
          <strong>Iterations:</strong> Epochs<br />
          <strong>Learning Rate:</strong> Step size<br />
          <strong>Output Nodes:</strong> Number of outputs<br />
          <strong>Activation:</strong> Sigmoid, ReLU, Identity, or Softmax
        </p>
        <h3 className="text-xl font-semibold text-white">Applications</h3>
        <ul className="list-disc list-inside text-gray-300">
          <li>Linearly separable classification</li>
          <li>Feature detection</li>
          <li>Base for deep networks</li>
        </ul>
      </div>
      <style jsx global>{`
        .controls-section {
          margin-top: 1.5rem;
          width: 100%;
          max-width: 800px;
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

        .styled-select {
          flex: 1;
          margin: 0 0.75rem;
          padding: 6px 8px;
          background: #222;
          color: white;
          border: 1px solid #555;
          border-radius: 4px;
          appearance: none;
        }
        .styled-select::-ms-expand {
          display: none;
        }
      `}</style>
    </div>
  );
}