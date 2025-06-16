"use client";

import { useEffect, useRef, useState, useMemo, useCallback } from "react";
import { BlockMath } from "react-katex";
import "katex/dist/katex.min.css";

type Layer = { weights: number[][]; biases: number[] };

interface NetworkConfig {
  modelType: "custom" | "sklearn";
  inputSize: number;
  hiddenSize: number;
  outputNodes: number;
  hiddenLayers: number;
  activation: "sigmoid" | "relu" | "identity" | "softmax";
  outputActivation: "sigmoid" | "relu" | "identity" | "softmax";
  learningRate: number;
}

type ActivationFunction = {
  label: string;
  func: (x: number) => number;
  derivative: (y: number) => number;
  formula: string;
};

type Activations = {
  [key: string]: ActivationFunction;
};

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
  softmax: {
    label: "Softmax",
    func: (x: number) => {
      console.warn("Softmax activation expects an array, applying to single value might not be intended.");
      const expX = Math.exp(x);
      return expX / (expX + 1);
    },
    derivative: (_: number) => 1,
    formula: "\\text{Softmax}(x_i) = \\frac{e^{x_i}}{\\sum_j e^{x_j}}",
  },
};

const initialNetworkConfig: NetworkConfig = {
  modelType: "custom",
  inputSize: 2,
  hiddenSize: 4,
  outputNodes: 1,
  hiddenLayers: 1,
  activation: "sigmoid",
  outputActivation: "sigmoid",
  learningRate: 0.1,
};

const MIN_NODES = 1;
const MAX_NODES = 10;
const MIN_HIDDEN_SIZE = 1;
const MAX_HIDDEN_SIZE = 20;

const NODE_RADIUS = 15;
const LAYER_SPACING = 150;
const NODE_SPACING = 50;

export default function Perceptron() {
  const [networkConfig, setNetworkConfig] = useState<NetworkConfig>(initialNetworkConfig);
  const [network, setNetwork] = useState<Layer[] | null>(null);
  const [inputData, setInputData] = useState<number[]>([]);
  const [targetData, setTargetData] = useState<number[]>([]);
  const [trainError, setTrainError] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const canvasRef = useRef<HTMLCanvasElement>(null);

  const networkStructure = useMemo(() => {
    if (!network) return [];

    const structure = [];

    structure.push({ type: 'input', nodes: networkConfig.inputSize });

    if (networkConfig.modelType === "custom") {
      for (let i = 0; i < networkConfig.hiddenLayers; i++) {
          const layerIndexInNetworkState = i;
          if (network[layerIndexInNetworkState]) {
            structure.push({ type: 'hidden', nodes: networkConfig.hiddenSize });
          }
      }
    }
    structure.push({ type: 'output', nodes: networkConfig.outputNodes });

    return structure;
  }, [network, networkConfig.inputSize, networkConfig.hiddenLayers, networkConfig.outputNodes, networkConfig.modelType]);


  const initializeNetwork = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch("https://ml-vis-lbhl.onrender.com/initialize", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ config: networkConfig }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Failed to initialize network.");
      }

      const initializedNetwork: Layer[] = await response.json();
      setNetwork(initializedNetwork);
      setTrainError(null);
    } catch (err: any) {
      console.error("Initialization error:", err);
      setError(err.message || "Failed to initialize network.");
    } finally {
      setIsLoading(false);
    }
  }, [networkConfig]);

  const trainNetwork = useCallback(async () => {
    if (!network || inputData.length === 0 || targetData.length === 0) {
      setError("Please initialize network and set input/target data.");
      return;
    }

    try {
      const classes = networkConfig.modelType === "sklearn" ? [0, 1] : undefined;

      const response = await fetch("https://ml-vis-lbhl.onrender.com/train", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          config: networkConfig,
          network: network,
          input: [inputData],
          target: [targetData],
          classes: classes,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Failed to train network.");
      }

      const result: { network: Layer[]; error: number; message?: string } = await response.json();
      setNetwork(result.network);
      setTrainError(result.error);
    } catch (err: any) {
      console.error("Training error:", err);
      setTrainError(null);
      setError(err.message || "Failed to train network.");
    }
  }, [network, networkConfig, inputData, targetData]);

  useEffect(() => {
    initializeNetwork();
  }, [initializeNetwork]);


  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !network) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const totalLayers = networkStructure.length;
    const maxNodesInAnyLayer = Math.max(...networkStructure.map(layer => layer.nodes));

    const drawingWidth = (totalLayers - 1) * LAYER_SPACING + NODE_RADIUS * 2;
    const drawingHeight = maxNodesInAnyLayer * NODE_SPACING;

    const startX = (canvas.width - drawingWidth) / 2;
    const startY = (canvas.height - drawingHeight) / 2;

    const getActivations = async () => {
        if (!inputData.length || !network) return null;
        try {
            const response = await fetch("https://ml-vis-lbhl.onrender.com/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    config: networkConfig,
                    network: network,
                    input: inputData
                })
            });
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || "Failed to get predictions.");
            }
            const result = await response.json();
            return result.activations;
        } catch (err) {
            console.error("Prediction error:", err);
            return null;
        }
    };


    let currentLayerX = startX;
    const layerPositions: { x: number, y: number[] }[] = [];

    networkStructure.forEach((layerInfo, layerIndex) => {
        const numNodes = layerInfo.nodes;
        const layerHeight = numNodes * NODE_SPACING;
        const layerStartY = startY + (drawingHeight - layerHeight) / 2;

        const nodePositionsY: number[] = [];

        for (let i = 0; i < numNodes; i++) {
            const nodeX = currentLayerX + NODE_RADIUS;
            const nodeY = layerStartY + i * NODE_SPACING + NODE_RADIUS;
            nodePositionsY.push(nodeY);

            ctx.beginPath();
            ctx.arc(nodeX, nodeY, NODE_RADIUS, 0, Math.PI * 2);
            ctx.fillStyle = layerInfo.type === 'input' ? '#4CAF50' : (layerInfo.type === 'output' ? '#F44336' : '#2196F3');
            ctx.fill();
            ctx.strokeStyle = '#333';
            ctx.lineWidth = 1;
            ctx.stroke();

            ctx.fillStyle = 'white';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.font = '12px Arial';
            let label = '';
            if (layerInfo.type === 'input') {
                label = inputData[i] !== undefined ? inputData[i].toFixed(1) : 'Input';
            }
            ctx.fillText(label, nodeX, nodeY);
        }
        layerPositions.push({ x: currentLayerX + NODE_RADIUS, y: nodePositionsY });
        currentLayerX += LAYER_SPACING;
    });

    for (let i = 0; i < layerPositions.length - 1; i++) {
        const currentLayer = layerPositions[i];
        const nextLayer = layerPositions[i + 1];

        const weights = network[i]?.weights;

        if (weights) {
            for (let j = 0; j < currentLayer.y.length; j++) {
                for (let k = 0; k < nextLayer.y.length; k++) {
                    const weight = weights[k]?.[j];

                    if (weight !== undefined) {
                        ctx.beginPath();
                        ctx.moveTo(currentLayer.x + NODE_RADIUS, currentLayer.y[j]);
                        ctx.lineTo(nextLayer.x - NODE_RADIUS, nextLayer.y[k]);

                        ctx.strokeStyle = weight > 0 ? '#0F0' : (weight < 0 ? '#F00' : '#888');
                        ctx.lineWidth = Math.min(Math.abs(weight) * 2 + 0.5, 5);
                        ctx.stroke();
                    }
                }
            }
        }
    }
  }, [network, inputData, networkConfig, networkStructure]);


  const renderActivationDetails = (activationType: "activation" | "outputActivation") => {
    const act = activations[networkConfig[activationType]];
    return (
      <div className="activation-details">
        <h4>{activationType === "activation" ? "Hidden Layer Activation" : "Output Layer Activation"}</h4>
        <p>Name: {act.label}</p>
        <p>Formula:</p>
        <BlockMath math={act.formula} />
      </div>
    );
  };

  return (
    <div className="container">
      <style jsx global>{`
        body {
          margin: 0;
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
            Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji",
            "Segoe UI Symbol";
          background-color: #1a1a1a;
          color: white;
        }

        .container {
          display: flex;
          flex-direction: column;
          align-items: center;
          padding: 2rem;
          min-height: 100vh;
          background-color: #1a1a1a;
        }

        .header {
          text-align: center;
          margin-bottom: 2rem;
        }

        .main-content {
          display: flex;
          flex-direction: row;
          gap: 2rem;
          width: 100%;
          max-width: 1200px;
          justify-content: center;
        }

        .controls-panel {
          flex: 0 0 300px;
          background-color: #2a2a2a;
          padding: 1.5rem;
          border-radius: 8px;
          box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
          display: flex;
          flex-direction: column;
          gap: 1.2rem;
        }

        .control-group {
          display: flex;
          align-items: center;
          justify-content: space-between;
          margin-bottom: 0.5rem;
        }

        .control-group label {
          font-weight: bold;
          white-space: nowrap;
          margin-right: 1rem;
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
          outline: none;
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
        
        .slider::-ms-thumb {
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
          position: relative;
        }

        .styled-select select {
          width: 100%;
          padding: 0.5rem 1rem;
          border-radius: 4px;
          background-color: #333;
          color: white;
          border: 1px solid #ff3860;
          appearance: none;
          -webkit-appearance: none;
          cursor: pointer;
          font-size: 1rem;
        }

        .styled-select::after {
          content: '▼';
          position: absolute;
          right: 10px;
          top: 50%;
          transform: translateY(-50%);
          color: white;
          pointer-events: none;
        }

        .data-input-group {
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
        }

        .data-input-group input {
          padding: 0.5rem;
          border-radius: 4px;
          border: 1px solid #333;
          background-color: #444;
          color: white;
        }

        .buttons {
          display: flex;
          gap: 1rem;
          margin-top: 1.5rem;
          justify-content: center;
        }

        .buttons button {
          padding: 0.75rem 1.5rem;
          border: none;
          border-radius: 4px;
          cursor: pointer;
          font-size: 1rem;
          font-weight: bold;
          transition: background-color 0.2s ease;
        }

        .buttons button:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .buttons .train-button {
          background-color: #ff3860;
          color: white;
        }

        .buttons .train-button:hover:not(:disabled) {
          background-color: #e02f52;
        }

        .buttons .reset-button {
          background-color: #555;
          color: white;
        }

        .buttons .reset-button:hover:not(:disabled) {
          background-color: #777;
        }

        .visualization-panel {
          flex: 1;
          display: flex;
          flex-direction: column;
          align-items: center;
          background-color: #2a2a2a;
          padding: 1.5rem;
          border-radius: 8px;
          box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }

        canvas {
          background-color: #1a1a1a;
          border-radius: 8px;
          box-shadow: inset 0 0 5px rgba(0, 0, 0, 0.5);
          margin-bottom: 1.5rem;
        }

        .info-box {
          background-color: #3a3a3a;
          padding: 1rem;
          border-radius: 6px;
          width: 100%;
          margin-top: 1rem;
        }

        .info-box p {
          margin: 0.5rem 0;
          color: #bbb;
        }

        .error-message {
          color: #ff6b6b;
          background-color: #3a1a1a;
          padding: 0.75rem;
          border-radius: 4px;
          margin-top: 1rem;
          text-align: center;
        }

        .loading-message {
          color: #6bb0ff;
          margin-top: 1rem;
          text-align: center;
        }

        .activation-details {
            background-color: #3a3a3a;
            padding: 1rem;
            border-radius: 6px;
            margin-top: 1rem;
            width: 100%;
        }
        .activation-details h4 {
            margin-top: 0;
            color: #ff3860;
        }
        .activation-details p {
            margin: 0.5rem 0;
            color: #bbb;
        }
        .activation-details .katex {
            font-size: 1.2rem;
            color: #eee;
        }
        .note {
            font-size: 0.85rem;
            color: #aaa;
            margin-top: 0.5rem;
            padding: 0 1rem;
        }
      `}</style>

      <header className="header">
        <h1>Perceptron Visualizer</h1>
      </header>

      <div className="main-content">
        <div className="controls-panel">
          <h2>Network Configuration</h2>

          <div className="control-group">
            <label>Model Type:</label>
            <div className="styled-select">
              <select
                value={networkConfig.modelType}
                onChange={(e) =>
                  setNetworkConfig((prev) => ({
                    ...prev,
                    modelType: e.target.value as "custom" | "sklearn",
                  }))
                }
              >
                <option value="custom">Custom Perceptron (MLP)</option>
                <option value="sklearn">Sklearn Perceptron</option>
              </select>
            </div>
          </div>

          <div className="control-group">
            <label>Input Nodes:</label>
            <input
              type="range"
              min={MIN_NODES}
              max={MAX_NODES}
              value={networkConfig.inputSize}
              onChange={(e) =>
                setNetworkConfig((prev) => ({
                  ...prev,
                  inputSize: parseInt(e.target.value),
                }))
              }
              className="slider"
              style={{ "--value": `${((networkConfig.inputSize - MIN_NODES) / (MAX_NODES - MIN_NODES)) * 100}%` } as React.CSSProperties}
            />
            <span className="control-value">{networkConfig.inputSize}</span>
          </div>

          {networkConfig.modelType === "custom" && (
            <>
              <div className="control-group">
                <label>Hidden Layers:</label>
                <input
                  type="range"
                  min="0"
                  max="5"
                  value={networkConfig.hiddenLayers}
                  onChange={(e) =>
                    setNetworkConfig((prev) => ({
                      ...prev,
                      hiddenLayers: parseInt(e.target.value),
                    }))
                  }
                  className="slider"
                  style={{ "--value": `${(networkConfig.hiddenLayers / 5) * 100}%` } as React.CSSProperties}
                />
                <span className="control-value">{networkConfig.hiddenLayers}</span>
              </div>

              <div className="control-group">
                <label>Hidden Size:</label>
                <input
                  type="range"
                  min={MIN_HIDDEN_SIZE}
                  max={MAX_HIDDEN_SIZE}
                  value={networkConfig.hiddenSize}
                  onChange={(e) =>
                    setNetworkConfig((prev) => ({
                      ...prev,
                      hiddenSize: parseInt(e.target.value),
                    }))
                  }
                  className="slider"
                  style={{ "--value": `${((networkConfig.hiddenSize - MIN_HIDDEN_SIZE) / (MAX_HIDDEN_SIZE - MIN_HIDDEN_SIZE)) * 100}%` } as React.CSSProperties}
                />
                <span className="control-value">{networkConfig.hiddenSize}</span>
              </div>
            </>
          )}

          <div className="control-group">
            <label>Output Nodes:</label>
            <input
              type="range"
              min={MIN_NODES}
              max={MAX_NODES}
              value={networkConfig.outputNodes}
              onChange={(e) =>
                setNetworkConfig((prev) => ({
                  ...prev,
                  outputNodes: parseInt(e.target.value),
                }))
              }
              className="slider"
              style={{ "--value": `${((networkConfig.outputNodes - MIN_NODES) / (MAX_NODES - MIN_NODES)) * 100}%` } as React.CSSProperties}
            />
            <span className="control-value">{networkConfig.outputNodes}</span>
          </div>

          {networkConfig.modelType === "custom" && (
            <div className="control-group">
              <label>Hidden Activation:</label>
              <div className="styled-select">
                <select
                  value={networkConfig.activation}
                  onChange={(e) =>
                    setNetworkConfig((prev) => ({
                      ...prev,
                      activation: e.target.value as NetworkConfig["activation"],
                    }))
                  }
                >
                  {Object.keys(activations).map((key) => (
                    <option key={key} value={key}>
                      {activations[key].label}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          )}

          <div className="control-group">
            <label>Output Activation:</label>
            <div className="styled-select">
              <select
                value={networkConfig.outputActivation}
                onChange={(e) =>
                  setNetworkConfig((prev) => ({
                    ...prev,
                    outputActivation: e.target.value as NetworkConfig["outputActivation"],
                  }))
                }
              >
                {Object.keys(activations).map((key) => (
                  <option key={key} value={key}>
                    {activations[key].label}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="control-group">
            <label>Learning Rate:</label>
            <input
              type="range"
              min="0.001"
              max="1.0"
              step="0.001"
              value={networkConfig.learningRate}
              onChange={(e) =>
                setNetworkConfig((prev) => ({
                  ...prev,
                  learningRate: parseFloat(e.target.value),
                }))
              }
              className="slider"
              style={{ "--value": `${(networkConfig.learningRate / 1.0) * 100}%` } as React.CSSProperties}
            />
            <span className="control-value">{networkConfig.learningRate.toFixed(3)}</span>
          </div>

          {networkConfig.modelType === "sklearn" && (
            <p className="note">
              *Sklearn Perceptron is a single-layer binary classifier. Sklearn MLPClassifier supports hidden layers.
              Output activation will be applied for display purposes only if using Sklearn Perceptron.
            </p>
          )}

          <hr style={{ borderColor: '#444' }}/>

          <h2>Data Input</h2>
          <div className="data-input-group">
            <label>Input Data (comma-separated):</label>
            <input
              type="text"
              value={inputData.join(',')}
              onChange={(e) => setInputData(e.target.value.split(',').map(Number))}
              placeholder="e.g., 0.5, 0.8"
            />
          </div>
          <div className="data-input-group">
            <label>Target Data (comma-separated):</label>
            <input
              type="text"
              value={targetData.join(',')}
              onChange={(e) => setTargetData(e.target.value.split(',').map(Number))}
              placeholder="e.g., 0.1"
            />
          </div>

          <div className="buttons">
            <button onClick={trainNetwork} disabled={isLoading || !network} className="train-button">
              {isLoading ? "Training..." : "Train One Step"}
            </button>
            <button onClick={initializeNetwork} disabled={isLoading} className="reset-button">
              Reset Network
            </button>
          </div>

          {isLoading && <p className="loading-message">Initializing Network...</p>}
          {error && <p className="error-message">Error: {error}</p>}
          {trainError !== null && (
            <div className="info-box">
              <p>Last Train MSE: {trainError.toFixed(6)}</p>
            </div>
          )}
        </div>

        <div className="visualization-panel">
          <h2>Network Visualization</h2>
          <canvas
            ref={canvasRef}
            width={700}
            height={500}
            style={{ border: '1px solid #ff3860' }}
          ></canvas>

          {networkConfig.modelType === "custom" && renderActivationDetails("activation")}
          {renderActivationDetails("outputActivation")}
        </div>
      </div>
    </div>
  );
}