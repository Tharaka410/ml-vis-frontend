"use client";

import { useEffect, useRef, useState, useMemo, useCallback } from "react";
import { BlockMath } from "react-katex";
import "katex/dist/katex.min.css";

type Layer = { weights: number[][]; biases: number[] };

interface NetworkConfig {
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
      return Math.exp(x) / (1 + Math.exp(x));
    },
    derivative: (y: number) => y * (1 - y),
    formula: "\\text{Softmax}(x_i) = \\frac{e^{x_i}}{\\sum_j e^{x_j}}",
  },
};

export default function PerceptronPage() {
  const [networkConfig, setNetworkConfig] = useState<NetworkConfig>({
    inputSize: 2,
    hiddenSize: 4,
    outputNodes: 1,
    hiddenLayers: 1,
    activation: "sigmoid",
    outputActivation: "sigmoid",
    learningRate: 0.1,
  });

  const [currentNetwork, setCurrentNetwork] = useState<Layer[]>([]);
  const [trainingData, setTrainingData] = useState<{
    input: number[][];
    target: number[][];
  }>({ input: [], target: [] });
  const [trainingError, setTrainingError] = useState<number | null>(null);
  const [trainingAccuracy, setTrainingAccuracy] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingIteration, setTrainingIteration] = useState(0);
  const [totalErrorHistory, setTotalErrorHistory] = useState<number[]>([]);
  const [layerDeltas, setLayerDeltas] = useState<number[][]>([]);
  const [fullActivations, setFullActivations] = useState<number[][]>([]);
  const animationFrameId = useRef<number | null>(null);

  const totalLayers = networkConfig.hiddenLayers + 1;

  const generateRandomData = useCallback(() => {
    const numSamples = 50;
    const input = Array(numSamples)
      .fill(0)
      .map(() => [Math.random() * 2 - 1, Math.random() * 2 - 1]);
    const target = input.map(([x, y]) =>
      x > 0 && y > 0 ? [1] : x < 0 && y < 0 ? [1] : [0]
    );
    setTrainingData({ input, target });
  }, []);

  useEffect(() => {
    generateRandomData();
  }, [generateRandomData]);

  const initializeNetwork = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch(
        "https://ml-vis-lbhl.onrender.com/initialize",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(networkConfig),
        }
      );
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(`Network response was not ok: ${errorData.detail || response.statusText}`);
      }
      const data: Layer[] = await response.json();
      setCurrentNetwork(data);
      setTrainingError(null);
      setTrainingAccuracy(null);
      setTotalErrorHistory([]);
      setTrainingIteration(0);
      setLayerDeltas([]);
      setFullActivations([]);
    } catch (err: any) {
      console.error("Error initializing network:", err);
      setError(`Error initializing network: ${err.message}`);
    } finally {
      setIsLoading(false);
    }
  }, [networkConfig]);

  useEffect(() => {
    initializeNetwork();
  }, [initializeNetwork]);

  const trainNetworkOneIteration = useCallback(async () => {
    setError(null);
    if (!currentNetwork.length || !trainingData.input.length) {
      setError("Network or training data not initialized.");
      return;
    }

    const randomIndex = Math.floor(Math.random() * trainingData.input.length);
    const inputSample = trainingData.input[randomIndex];
    const targetSample = trainingData.target[randomIndex];

    try {
      const response = await fetch("https://ml-vis-lbhl.onrender.com/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          input: [inputSample],
          target: [targetSample],
          network: currentNetwork,
          config: networkConfig,
        }),
      });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(`Network response was not ok: ${errorData.detail || response.statusText}`);
      }
      const data = await response.json();
      setCurrentNetwork(data.network);
      setTrainingError(data.error);
      setTrainingAccuracy(data.accuracy);
      setTrainingIteration((prev) => prev + 1);
      setTotalErrorHistory((prev) => [...prev, data.error]);
      setLayerDeltas(data.deltas || []);
      setFullActivations(data.full_activations || []);
    } catch (err: any) {
      console.error("Error training network:", err);
      setError(`Error training network: ${err.message}`);
      setIsTraining(false);
    }
  }, [currentNetwork, trainingData, networkConfig]);

  const startTraining = useCallback(() => {
    if (isTraining) return;
    setIsTraining(true);
    const animate = () => {
      trainNetworkOneIteration();
      animationFrameId.current = requestAnimationFrame(animate);
    };
    animationFrameId.current = requestAnimationFrame(animate);
  }, [isTraining, trainNetworkOneIteration]);

  const stopTraining = useCallback(() => {
    if (animationFrameId.current) {
      cancelAnimationFrame(animationFrameId.current);
    }
    setIsTraining(false);
  }, []);

  useEffect(() => {
    return () => {
      if (animationFrameId.current) {
        cancelAnimationFrame(animationFrameId.current);
      }
    };
  }, []);

  const getLayerOutput = useCallback(
    (input: number[], layerIndex: number, network: Layer[]) => {
      let currentInput = input;
      for (let i = 0; i <= layerIndex; i++) {
        const layer = network[i];
        const weights = layer.weights;
        const biases = layer.biases;

        const rawOutput = weights.map((row) => {
          return row.reduce(
            (sum, weight, idx) => sum + weight * currentInput[idx],
            0
          );
        });

        const layerActivationName =
          i === totalLayers - 1
            ? networkConfig.outputActivation
            : networkConfig.activation;

        currentInput = rawOutput.map((val) =>
          activations[layerActivationName].func(val)
        );
      }
      return currentInput;
    },
    [activations, networkConfig.activation, networkConfig.outputActivation, totalLayers]
  );

  const renderNeuronConnections = (
    layerIndex: number,
    numInputNeurons: number,
    numOutputNeurons: number
  ) => {
    const connections = [];
    const svgWidth = 100; // Assuming 100% width
    const svgHeight = 100; // Assuming 100% height
    const xOffset = (svgWidth / (totalLayers + 1)) * layerIndex + 10;
    const nextXOffset = (svgWidth / (totalLayers + 1)) * (layerIndex + 1) + 10;

    for (let i = 0; i < numInputNeurons; i++) {
      for (let j = 0; j < numOutputNeurons; j++) {
        const weight = currentNetwork[layerIndex]?.weights[j]?.[i];
        if (weight !== undefined) {
          const opacity = Math.min(1, Math.abs(weight));
          const color = weight > 0 ? "rgba(0, 255, 0," : "rgba(255, 0, 0,"; // Green for positive, Red for negative
          const x1 = xOffset;
          const y1 = (svgHeight / (numInputNeurons + 1)) * (i + 1);
          const x2 = nextXOffset;
          const y2 = (svgHeight / (numOutputNeurons + 1)) * (j + 1);

          connections.push(
            <line
              key={`conn-${layerIndex}-${i}-${j}`}
              x1={`${x1}%`}
              y1={`${y1}%`}
              x2={`${x2}%`}
              y2={`${y2}%`}
              stroke={`${color} ${opacity})`}
              strokeWidth={Math.max(0.5, Math.abs(weight) * 3)}
              className="transition-all duration-100 ease-in-out"
            />
          );

          // Add text for weight
          const midX = (x1 + x2) / 2;
          const midY = (y1 + y2) / 2;
          connections.push(
            <text
              key={`weight-text-${layerIndex}-${i}-${j}`}
              x={`${midX}%`}
              y={`${midY}%`}
              fill="white"
              fontSize="8"
              textAnchor="middle"
              dominantBaseline="central"
              className="transition-all duration-100 ease-in-out"
            >
              {weight.toFixed(2)}
            </text>
          );
        }
      }
    }
    return connections;
  };

  const renderNeurons = (numNeurons: number, layerIndex: number) => {
    const neurons = [];
    const fullActLayer = fullActivations?.[layerIndex * 2 + 1]; // Raw outputs (Z)
    const activatedLayer = fullActivations?.[layerIndex * 2 + 2]; // Activated outputs (A)

    for (let i = 0; i < numNeurons; i++) {
      const bias = currentNetwork[layerIndex]?.biases?.[i];
      const rawOutput = fullActLayer?.[i];
      const activatedOutput = activatedLayer?.[i];
      const delta = layerDeltas?.[layerIndex]?.[i];

      const neuronYPos = (100 / (numNeurons + 1)) * (i + 1);

      neurons.push(
        <div
          key={`neuron-${layerIndex}-${i}`}
          className="relative w-12 h-12 bg-neutral-800 rounded-full flex items-center justify-center text-xs text-white border-2 border-neutral-700 shadow-md transform transition-all duration-100 ease-in-out"
          style={{
            top: `${neuronYPos}%`,
            left: "50%",
            transform: "translate(-50%, -50%)",
            backgroundColor: activatedOutput !== undefined
                ? `rgba(100, 200, 255, ${Math.min(1, Math.abs(activatedOutput))})` // Lighter blue for active
                : 'rgb(82, 82, 82)', // Darker gray for inactive
            borderColor: delta !== undefined
                ? (delta > 0 ? 'rgba(0, 255, 0, 0.8)' : 'rgba(255, 0, 0, 0.8)') // Green for positive delta, red for negative
                : 'rgb(100, 100, 100)', // Default border
            borderWidth: delta !== undefined
                ? Math.max(1, Math.abs(delta) * 5)
                : 2,
          }}
        >
          {/* Neuron Value Labels */}
          {rawOutput !== undefined && (
            <span className="absolute -left-10 text-xs">Z: {rawOutput.toFixed(2)}</span>
          )}
          {activatedOutput !== undefined && (
            <span className="absolute -right-10 text-xs">A: {activatedOutput.toFixed(2)}</span>
          )}
          {delta !== undefined && (
            <span className="absolute -bottom-4 text-xs">Δ: {delta.toFixed(2)}</span>
          )}
        </div>
      );

      // Separate circle for Bias
      if (bias !== undefined) {
        neurons.push(
          <div
            key={`bias-${layerIndex}-${i}`}
            className="absolute w-6 h-6 bg-yellow-600 rounded-full flex items-center justify-center text-xs text-white border border-yellow-400"
            style={{
              top: `${neuronYPos}%`,
              left: "calc(50% - 40px)", // Position to the left of the neuron
              transform: "translate(-50%, -50%)",
            }}
          >
            {bias.toFixed(2)}
          </div>
        );
      }
    }
    return neurons;
  };

  const renderInputNeurons = (numNeurons: number) => {
    const neurons = [];
    const inputLayerActivations = fullActivations?.[0]; // Input values are the first "activations"

    for (let i = 0; i < numNeurons; i++) {
      const inputValue = inputLayerActivations?.[i];
      neurons.push(
        <div
          key={`input-neuron-${i}`}
          className="relative w-12 h-12 bg-blue-900 rounded-full flex items-center justify-center text-xs text-white border-2 border-blue-700 shadow-md"
          style={{
            top: `${(100 / (numNeurons + 1)) * (i + 1)}%`,
            left: "50%",
            transform: "translate(-50%, -50%)",
          }}
        >
          {inputValue !== undefined && (
            <span className="absolute inset-0 flex items-center justify-center">
              X{i + 1}: {inputValue.toFixed(2)}
            </span>
          )}
        </div>
      );
    }
    return neurons;
  };

  const renderOutputNeurons = (numNeurons: number) => {
    const neurons = [];
    const outputLayerActivations = fullActivations?.[(totalLayers * 2)]; // Last layer's activated output

    for (let i = 0; i < numNeurons; i++) {
      const activatedOutput = outputLayerActivations?.[i];
      const delta = layerDeltas?.[totalLayers - 1]?.[i];

      neurons.push(
        <div
          key={`output-neuron-${i}`}
          className="relative w-12 h-12 bg-purple-900 rounded-full flex items-center justify-center text-xs text-white border-2 border-purple-700 shadow-md transform transition-all duration-100 ease-in-out"
          style={{
            top: `${(100 / (numNeurons + 1)) * (i + 1)}%`,
            left: "50%",
            transform: "translate(-50%, -50%)",
            backgroundColor: activatedOutput !== undefined
                ? `rgba(150, 50, 255, ${Math.min(1, Math.abs(activatedOutput))})` // Lighter purple for active
                : 'rgb(82, 82, 82)', // Darker gray for inactive
            borderColor: delta !== undefined
                ? (delta > 0 ? 'rgba(0, 255, 0, 0.8)' : 'rgba(255, 0, 0, 0.8)') // Green for positive delta, red for negative
                : 'rgb(100, 100, 100)', // Default border
            borderWidth: delta !== undefined
                ? Math.max(1, Math.abs(delta) * 5)
                : 2,
          }}
        >
          {activatedOutput !== undefined && (
            <span className="absolute inset-0 flex items-center justify-center">
              Y{i + 1}: {activatedOutput.toFixed(2)}
            </span>
          )}
          {delta !== undefined && (
            <span className="absolute -bottom-4 text-xs">Δ: {delta.toFixed(2)}</span>
          )}
        </div>
      );
    }
    return neurons;
  };

  const currentOutput = useMemo(() => {
    if (currentNetwork.length && trainingData.input.length) {
      const lastActivation = fullActivations?.[(totalLayers * 2)];
      if (lastActivation) {
        return lastActivation;
      }
    }
    return [];
  }, [currentNetwork, trainingData, fullActivations, totalLayers]);

  return (
    <div className="flex flex-col min-h-screen bg-black text-gray-100 font-sans p-4"> {/* Changed main bg to black */}
      <h1 className="text-4xl font-extrabold text-center text-blue-400 mb-8">
        MLP Visualizer
      </h1>

      {error && <div className="text-red-500 text-center mb-4">{error}</div>}
      {isLoading && (
        <div className="text-blue-300 text-center mb-4">
          Initializing Network...
        </div>
      )}

      {/* Network Visualization (Canvas) - Bigger and on top */}
      <div className="bg-neutral-900 p-6 rounded-lg shadow-xl relative h-[700px] mb-8"> {/* Increased height */}
        <h2 className="text-2xl font-bold text-blue-300 mb-4 text-center">
          Network Visualization
        </h2>
        <svg
          className="absolute inset-0 w-full h-full"
          style={{ overflow: "visible" }}
        >
          {currentNetwork.map((layer, layerIndex) => {
            const numInputNeurons =
              layerIndex === 0
                ? networkConfig.inputSize
                : networkConfig.hiddenSize;
            const numOutputNeurons =
              layerIndex === totalLayers - 1
                ? networkConfig.outputNodes
                : networkConfig.hiddenSize;
            return renderNeuronConnections(
              layerIndex,
              numInputNeurons,
              numOutputNeurons
            );
          })}
        </svg>

        <div className="flex justify-around items-center h-full">
          <div
            className="flex flex-col justify-around items-center h-full relative"
            style={{ flexBasis: "10%" }}
          >
            <div className="text-lg font-semibold absolute -top-8">Input</div>
            {renderInputNeurons(networkConfig.inputSize)}
          </div>

          {Array.from({ length: networkConfig.hiddenLayers }).map(
            (_, layerIndex) => (
              <div
                key={`hidden-layer-${layerIndex}`}
                className="flex flex-col justify-around items-center h-full relative"
                style={{ flexBasis: `${80 / totalLayers}%` }}
              >
                <div className="text-lg font-semibold absolute -top-8">
                  Hidden {layerIndex + 1}
                </div>
                {renderNeurons(networkConfig.hiddenSize, layerIndex)}
              </div>
            )
          )}

          <div
            className="flex flex-col justify-around items-center h-full relative"
            style={{ flexBasis: "10%" }}
          >
            <div className="text-lg font-semibold absolute -top-8">Output</div>
            {renderOutputNeurons(networkConfig.outputNodes)}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8"> {/* Container for controls and text */}
        {/* Controls Column */}
        <div className="flex flex-col gap-6">
          <div className="bg-neutral-900 p-6 rounded-lg shadow-xl">
            <h2 className="text-2xl font-bold text-blue-300 mb-4">
              Network Configuration
            </h2>
            <div className="space-y-4">
              <div>
                <label htmlFor="inputSize" className="block text-sm font-medium">
                  Input Neurons: {networkConfig.inputSize}
                </label>
                <input
                  type="range"
                  id="inputSize"
                  min="1"
                  max="10"
                  value={networkConfig.inputSize}
                  onChange={(e) =>
                    setNetworkConfig({
                      ...networkConfig,
                      inputSize: parseInt(e.target.value),
                    })
                  }
                  className="w-full h-2 rounded-lg appearance-none cursor-pointer bg-neutral-800"
                />
              </div>

              <div>
                <label htmlFor="hiddenLayers" className="block text-sm font-medium">
                  Hidden Layers: {networkConfig.hiddenLayers}
                </label>
                <input
                  type="range"
                  id="hiddenLayers"
                  min="0"
                  max="5"
                  value={networkConfig.hiddenLayers}
                  onChange={(e) =>
                    setNetworkConfig({
                      ...networkConfig,
                      hiddenLayers: parseInt(e.target.value),
                    })
                  }
                  className="w-full h-2 rounded-lg appearance-none cursor-pointer bg-neutral-800"
                />
              </div>

              <div>
                <label htmlFor="hiddenSize" className="block text-sm font-medium">
                  Neurons per Hidden Layer: {networkConfig.hiddenSize}
                </label>
                <input
                  type="range"
                  id="hiddenSize"
                  min="1"
                  max="10"
                  value={networkConfig.hiddenSize}
                  onChange={(e) =>
                    setNetworkConfig({
                      ...networkConfig,
                      hiddenSize: parseInt(e.target.value),
                    })
                  }
                  className="w-full h-2 rounded-lg appearance-none cursor-pointer bg-neutral-800"
                />
              </div>

              <div>
                <label htmlFor="outputNodes" className="block text-sm font-medium">
                  Output Neurons: {networkConfig.outputNodes}
                </label>
                <input
                  type="range"
                  id="outputNodes"
                  min="1"
                  max="10"
                  value={networkConfig.outputNodes}
                  onChange={(e) =>
                    setNetworkConfig({
                      ...networkConfig,
                      outputNodes: parseInt(e.target.value),
                    })
                  }
                  className="w-full h-2 rounded-lg appearance-none cursor-pointer bg-neutral-800"
                />
              </div>

              <div>
                <label htmlFor="learningRate" className="block text-sm font-medium">
                  Learning Rate: {networkConfig.learningRate.toFixed(3)}
                </label>
                <input
                  type="range"
                  id="learningRate"
                  min="0.001"
                  max="0.5"
                  step="0.001"
                  value={networkConfig.learningRate}
                  onChange={(e) =>
                    setNetworkConfig({
                      ...networkConfig,
                      learningRate: parseFloat(e.target.value),
                    })
                  }
                  className="w-full h-2 rounded-lg appearance-none cursor-pointer bg-neutral-800"
                />
              </div>

              <div>
                <label htmlFor="activation" className="block text-sm font-medium">
                  Hidden Activation:
                </label>
                <select
                  id="activation"
                  value={networkConfig.activation}
                  onChange={(e) =>
                    setNetworkConfig({
                      ...networkConfig,
                      activation: e.target.value as NetworkConfig["activation"],
                    })
                  }
                  className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-neutral-700 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md bg-neutral-800 text-white"
                >
                  {Object.keys(activations).map((key) => (
                    <option key={key} value={key}>
                      {activations[key].label}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label htmlFor="outputActivation" className="block text-sm font-medium">
                  Output Activation:
                </label>
                <select
                  id="outputActivation"
                  value={networkConfig.outputActivation}
                  onChange={(e) =>
                    setNetworkConfig({
                      ...networkConfig,
                      outputActivation: e.target.value as NetworkConfig["outputActivation"],
                    })
                  }
                  className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-neutral-700 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md bg-neutral-800 text-white"
                >
                  {Object.keys(activations).map((key) => (
                    <option key={key} value={key}>
                      {activations[key].label}
                    </option>
                  ))}
                </select>
              </div>

              <button
                onClick={initializeNetwork}
                className="w-full bg-green-600 hover:bg-green-700 text-white font-bold py-2 px-4 rounded transition duration-200"
                disabled={isLoading || isTraining}
              >
                Reinitialize Network
              </button>
            </div>
          </div>

          <div className="bg-neutral-900 p-6 rounded-lg shadow-xl">
            <h2 className="text-2xl font-bold text-blue-300 mb-4">
              Training Controls & Metrics
            </h2>
            <div className="space-y-4">
              <button
                onClick={isTraining ? stopTraining : startTraining}
                className={`w-full font-bold py-2 px-4 rounded transition duration-200 ${
                  isTraining
                    ? "bg-red-600 hover:bg-red-700"
                    : "bg-blue-600 hover:bg-blue-700"
                }`}
                disabled={isLoading || !currentNetwork.length}
              >
                {isTraining ? "Stop Training" : "Start Training"}
              </button>
              <button
                onClick={trainNetworkOneIteration}
                className="w-full bg-yellow-600 hover:bg-yellow-700 text-white font-bold py-2 px-4 rounded transition duration-200"
                disabled={isLoading || isTraining || !currentNetwork.length}
              >
                Train One Iteration
              </button>
              <button
                onClick={generateRandomData}
                className="w-full bg-purple-600 hover:bg-purple-700 text-white font-bold py-2 px-4 rounded transition duration-200"
              >
                Generate New Data
              </button>
              <div className="text-lg">
                <strong>Iteration:</strong> {trainingIteration}
              </div>
              <div className="text-lg">
                <strong>Current Error (MSE):</strong>{" "}
                {trainingError !== null ? trainingError.toFixed(6) : "N/A"}
              </div>
              <h3 className="text-xl font-bold text-blue-200 mt-6">
                Error History
              </h3>
              <div className="max-h-48 overflow-y-auto bg-neutral-800 p-3 rounded">
                <ul className="text-sm">
                  {totalErrorHistory.slice(-20).map((error, idx) => (
                    <li key={idx} className="mb-1">
                      Iteration {trainingIteration - (totalErrorHistory.length - idx) + 1}: {error.toFixed(6)}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        </div>

        {/* Text Column */}
        <div className="bg-neutral-900 p-6 rounded-lg shadow-xl">
          <div className="space-y-4">
            <h3 className="text-2xl font-semibold text-white">What is a Perceptron?</h3>
            <p className="text-gray-300">
              A perceptron is the simplest artificial neural network for binary classification,
              computing a weighted sum plus bias and applying an activation function.
            </p>
            <h3 className="text-xl font-semibold text-white">Formula</h3>
            <BlockMath math="y = \\phi\\left(\\sum_i w_i x_i + b\\right)" />
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
            <h2 className="text-2xl font-bold text-blue-300 mb-4">
              Activation Formulas
            </h2>
            <div className="space-y-4">
            {Object.keys(activations).map((key) => (
              <div key={key}>
                <h3 className="text-xl font-semibold text-blue-200">
                  {activations[key].label}
                </h3>
                <BlockMath math={activations[key].formula} />
                <p className="text-sm mt-2">
                  Derivative:{" "}
                  <BlockMath
                    math={
                      key === "softmax"
                        ? "f'(y) = y(1-y) \\quad \\text{(Simplified for MSE)}"
                        : activations[key].derivative.toString().replace("(_: number) => ", "").replace("(y: number) => ", "")
                    }
                  />
                </p>
              </div>
            ))}
          </div>
          </div>
        </div>
      </div>

      <div className="bg-neutral-900 p-6 rounded-lg shadow-xl mt-6">
        <h2 className="text-2xl font-bold text-blue-300 mb-4 text-center">
          Network Weights and Biases
        </h2>
        <div className="space-y-6">
          {currentNetwork.map((layer, index) => (
            <div key={index} className="bg-neutral-800 p-4 rounded-lg">
              <h3 className="text-xl font-semibold text-blue-200 mb-3">
                Layer {index + 1} ({index === totalLayers - 1 ? "Output" : "Hidden"})
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <h4 className="text-lg font-medium mb-2">Weights:</h4>
                  <div className="max-h-48 overflow-y-auto bg-neutral-700 p-2 rounded text-sm">
                    {layer.weights.map((row, rIdx) => (
                      <div key={rIdx} className="whitespace-pre">
                        [{row.map((w) => w.toFixed(4)).join(", ")}]
                      </div>
                    ))}
                  </div>
                </div>
                <div>
                  <h4 className="text-lg font-medium mb-2">Biases:</h4>
                  <div className="max-h-48 overflow-y-auto bg-neutral-700 p-2 rounded text-sm">
                    [{layer.biases.map((b) => b.toFixed(4)).join(", ")}]
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
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