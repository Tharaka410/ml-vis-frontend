"use client";

import React, { useEffect, useRef, useState, useCallback } from "react";
import "katex/dist/katex.min.css";
import { BlockMath } from "react-katex";
import dynamic from "next/dynamic"; 
import { ScatterData } from "plotly.js";
const DynamicPlotlyGraph = dynamic(() => import("./plotlygraph"), {
  ssr: false, 
  loading: () => <p className="text-blue-400">Loading chart...</p>, 
});

interface LogisticRegressionDataResponse {
  X: number[][];
  y: number[];
  feature_names: string[];
}

interface LogisticRegressionHistoryResponse {
  weights_history: number[][]; // Each inner array is [w1, w2, ..., wn, bias]
  loss_history: number[];
  final_predictions: number[];
  final_loss: number;
}

type Params = {
  learningRate: number;
  iterations: number;
  decisionBoundary: number;
};

export default function LogisticRegressionPage() {
  const [params, setParams] = useState<Params>({
    learningRate: 0.1,
    iterations: 100,
    decisionBoundary: 0.5,
  });

  const [isAnimating, setIsAnimating] = useState(false);
  const [frame, setFrame] = useState(0);
  const reqRef = useRef<number | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const mainPlotRef = useRef<HTMLDivElement | null>(null);
  const lossPlotRef = useRef<HTMLDivElement | null>(null);
  const sigmoidPlotRef = useRef<HTMLDivElement | null>(null);

  const [dataX, setDataX] = useState<number[][]>([]);
  const [dataY, setDataY] = useState<number[]>([]);
  const [featureNames, setFeatureNames] = useState<string[]>([]);
  const [weightsHistory, setWeightsHistory] = useState<number[][]>([]);
  const [lossHistory, setLossHistory] = useState<number[]>([]);
  const [finalLoss, setFinalLoss] = useState<number | null>(null);
  const [finalPredictions, setFinalPredictions] = useState<number[]>([]); // Added for completeness, though not explicitly used in this render logic

  // paramControls now uses keyof Params
  const paramControls: {
    name: keyof Params;
    label: string;
    min: number;
    max: number;
    step: number;
    type?: 'range' | 'select'; 
    options?: { value: any; label: string }[];
  }[] = [
    { name: "learningRate", label: "Learning Rate", min: 0.01, max: 0.5, step: 0.01, type: 'range' },
    { name: "iterations", label: "Iterations", min: 10, max: 200, step: 10, type: 'range' },
    { name: "decisionBoundary", label: "Decision Boundary", min: 0.1, max: 0.9, step: 0.05, type: 'range' },
  ];

  // Derive currentLoss from lossHistory and frame
  const currentLoss = lossHistory.length > 0 && frame < lossHistory.length
    ? lossHistory[frame]
    : lossHistory.length > 0 ? lossHistory[lossHistory.length - 1] : 0;

  const handleAnimateToggle = () => {
    setIsAnimating((prev) => !prev);
  };


  const fetchTrainHistory = useCallback(async (X: number[][], y: number[]) => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch("${process.env.NEXT_PUBLIC_API_URL}/logistic-regression/train-history", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          X: X,
          y: y,
          learning_rate: params.learningRate,
          iterations: params.iterations,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const result: LogisticRegressionHistoryResponse = await response.json();
      setWeightsHistory(result.weights_history);
      setLossHistory(result.loss_history);
      setFinalLoss(result.final_loss);
      setFinalPredictions(result.final_predictions);
      setIsLoading(false);

    } catch (err: any) {
      console.error("Failed to fetch train history:", err);
      setError(err.message || "Failed to fetch training history.");
      setIsLoading(false);
    }
  }, [params.learningRate, params.iterations]);

  const fetchData = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch("${process.env.NEXT_PUBLIC_API_URL}/logistic-regression/data", {
        method: "GET", // Changed to GET as no body is needed for synthetic data
        headers: {
          "Content-Type": "application/json",
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const result: LogisticRegressionDataResponse = await response.json();
      setDataX(result.X);
      setDataY(result.y);
      setFeatureNames(result.feature_names);
      setFrame(0); // Reset animation frame
      setIsAnimating(false); // Stop animation when new data is fetched
      // setIsLoading(false); // This will be set by fetchTrainHistory after it completes

    } catch (err: any) {
      console.error("Failed to fetch data:", err);
      setError(err.message || "Failed to fetch data.");
      setIsLoading(false);
    }
  }, []);


  // Effect to fetch initial data and re-fetch when relevant params change (though numFeatures is removed)
  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // Effect to fetch training history when dataX/dataY or relevant params change
  useEffect(() => {
    if (dataX.length > 0 && dataY.length > 0) {
      fetchTrainHistory(dataX, dataY);
    }
  }, [dataX, dataY, fetchTrainHistory]);

  // Effect for animation loop
  useEffect(() => {
    if (!isAnimating) {
      cancelAnimationFrame(reqRef.current!);
      return;
    }

    const animate = () => {
      setFrame((prevFrame) => {
        if (prevFrame < params.iterations - 1) {
          reqRef.current = requestAnimationFrame(animate);
          return prevFrame + 1;
        } else {
          setIsAnimating(false);
          return prevFrame; 
        }
      });
    };

    reqRef.current = requestAnimationFrame(animate);

    return () => {
      if (reqRef.current) {
        cancelAnimationFrame(reqRef.current);
      }
    };
  }, [isAnimating, params.iterations]);


  // --- EFFECT FOR MAIN DATA VISUALIZATION (2D Plotly) ---
  const mainPlotData: Plotly.Data[] = React.useMemo(() => {
    if (!dataX.length || !dataY.length || !weightsHistory.length) {
      return [];
    }

    const currentWeights = weightsHistory[frame] || weightsHistory[weightsHistory.length - 1];
    if (!currentWeights || currentWeights.length < 3) {
        console.warn("Current weights or number of features is invalid for plotting.");
        return [];
    }
    const weights = currentWeights.slice(0, 2); // w1, w2
    const bias = currentWeights[2]; // bias

    const x_min = Math.min(...dataX.map(d => d[0]));
    const x_max = Math.max(...dataX.map(d => d[0]));

    const x1_line = [x_min, x_max];
    let y1_line: number[] = [];

    if (weights[1] !== 0) {
      y1_line = x1_line.map(x => (-weights[0] * x - bias) / weights[1]);
    } else if (weights[0] !== 0) {
      y1_line = [Math.min(...dataX.map(d => d[1])), Math.max(...dataX.map(d => d[1]))];
      x1_line[0] = -bias / weights[0];
      x1_line[1] = -bias / weights[0];
    } else {
      y1_line = [0, 0];
    }

    const trace0 = {
      x: dataX.filter((_, i) => dataY[i] === 0).map(d => d[0]),
      y: dataX.filter((_, i) => dataY[i] === 0).map(d => d[1]),
      mode: 'markers',
      type: 'scatter',
      name: 'Class 0',
      marker: { color: 'blue' },
    };

    const trace1 = {
      x: dataX.filter((_, i) => dataY[i] === 1).map(d => d[0]),
      y: dataX.filter((_, i) => dataY[i] === 1).map(d => d[1]),
      mode: 'markers',
      type: 'scatter',
      name: 'Class 1',
      marker: { color: 'red' },
    };

    const decisionBoundaryTrace = {
      x: x1_line,
      y: y1_line,
      mode: 'lines',
      type: 'scatter',
      name: 'Decision Boundary',
      line: { color: 'green', width: 2 },
    };

    return [trace0, trace1, decisionBoundaryTrace] as Plotly.Data[];
  }, [dataX, dataY, featureNames, weightsHistory, frame]); // Dependencies for data calculation

  const mainPlotLayout = React.useMemo(() => ({
    title: `Logistic Regression (Frame: <span class="math-inline">\{frame\}/</span>{params.iterations})`,
    xaxis: { title: featureNames[0] || 'Feature 1' },
    yaxis: { title: featureNames[1] || 'Feature 2' },
    hovermode: 'closest',
    height: 400,
    margin: { t: 50, b: 50, l: 50, r: 50 },
    paper_bgcolor: "black",
    plot_bgcolor: "black",
    font: { color: "white" },
  })as Partial<Plotly.Layout>, [frame, params.iterations, featureNames]); 


  // Data for Loss Plot
  const lossPlotData: Plotly.Data[] = React.useMemo(() => {
    if (!lossHistory.length) {
      return [];
    }
    return [{
      x: Array.from({ length: lossHistory.length }, (_, i) => i + 1),
      y: lossHistory,
      mode: 'lines',
      type: 'scatter',
      name: 'Loss',
      line: { color: 'purple' },
    }];
  }, [lossHistory]);

  const lossPlotLayout = React.useMemo(() => ({
    title: { text: 'Loss History' },
    xaxis: { title: 'Iteration' },
    yaxis: { title: 'Loss' },
    height: 250,
    margin: { t: 50, b: 50, l: 50, r: 50 },
    paper_bgcolor: "black",
    plot_bgcolor: "black",
    font: { color: "white" },
  }) as Partial<Plotly.Layout>, []);


  // Data for Sigmoid Plot
  const sigmoidPlotData = React.useMemo(() => {
    if (!dataX.length || !dataY.length || !weightsHistory.length) {
      return [];
    }

    const currentWeights = weightsHistory[frame] || weightsHistory[weightsHistory.length - 1];
    if (!currentWeights || currentWeights.length < 3) {
        console.warn("Current weights or number of features is invalid for sigmoid plotting.");
        return [];
    }
    const weights = currentWeights.slice(0, 2); // w1, w2
    const bias = currentWeights[2]; // bias

    const linearOutputs = dataX.map(d => weights[0] * d[0] + weights[1] * d[1] + bias);
    const predictedProbs = linearOutputs.map(z => 1 / (1 + Math.exp(-z)));

    const z_range = Array.from({ length: 100 }, (_, i) => -10 + (i * 20 / 99));
    const sigmoid_curve_y = z_range.map(z => 1 / (1 + Math.exp(-z)));

    const traceSigmoidPoints0 = {
      x: linearOutputs.filter((_, i) => dataY[i] === 0),
      y: predictedProbs.filter((_, i) => dataY[i] === 0),
      mode: 'markers',
      type: 'scatter',
      name: 'Class 0 Data Points',
      marker: { color: 'blue', size: 6 },
    };

    const traceSigmoidPoints1 = {
      x: linearOutputs.filter((_, i) => dataY[i] === 1),
      y: predictedProbs.filter((_, i) => dataY[i] === 1),
      mode: 'markers',
      type: 'scatter',
      name: 'Class 1 Data Points',
      marker: { color: 'red', size: 6 },
    };

    const sigmoidCurveTrace = {
      x: z_range,
      y: sigmoid_curve_y,
      mode: 'lines',
      type: 'scatter',
      name: 'Sigmoid Function',
      line: { color: 'orange', width: 3 },
    };

    const decisionBoundaryLine = {
      x: [0, 0],
      y: [0, 1],
      mode: 'lines',
      type: 'scatter',
      name: 'Decision Threshold (Z=0)',
      line: { color: 'green', dash: 'dash', width: 2 },
    };

    return [traceSigmoidPoints0, traceSigmoidPoints1, sigmoidCurveTrace, decisionBoundaryLine] as Plotly.Data[];
  }, [dataX, dataY, weightsHistory, frame]);

  const sigmoidPlotLayout = React.useMemo(() => ({
    title: { text: `Sigmoid Curve (Frame: ${frame}/${params.iterations})` }, // Corrected title type
    xaxis: { title: { text: 'Linear Output (z = w · x + b)' } }, // Corrected title type
    yaxis: { title: { text: 'Predicted Probability σ(z)' }, range: [0, 1] },
    hovermode: 'closest',
    height: 350,
    margin: { t: 50, b: 50, l: 50, r: 50 },
    paper_bgcolor: "black",
    plot_bgcolor: "black",
    font: { color: "white" },
  })as Partial<Plotly.Layout>, [frame, params.iterations]);


  return (
    <div className="space-y-8 bg-black text-white p-4 min-h-screen">
      <div>
        <h1 className="text-4xl font-bold mb-4">Logistic Regression (Binary Classification)</h1>
        <p className="text-lg text-gray-400 mb-6">
          A statistical model that uses a logistic function to model a binary dependent variable. Here, we visualize its training process using gradient descent.
        </p>
      </div>

      {isLoading && <p className="text-blue-400">Loading data and training history...</p>}
      {error && <p className="text-red-500">Error: {error}</p>}

      <div className="flex flex-col lg:flex-row gap-8">
        {/* Left Section: Visualizations */}
        <div className="lg:w-3/5 flex flex-col gap-4">
          {/* Data Points & Decision Boundary Plotly Div */}
          <div className="relative h-[400px] border border-gray-700 rounded-lg overflow-hidden bg-gray-800">
            <h3 className="text-lg font-semibold mb-2 p-2">Data Points & Decision Boundary ({featureNames[0] || 'Feature 1'} vs {featureNames[1] || 'Feature 2'})</h3>
            {/* Use the dynamic component here */}
            {dataX.length > 0 && dataY.length > 0 && weightsHistory.length > 0 && (
              <DynamicPlotlyGraph
                divRef={mainPlotRef}
                plotData={mainPlotData}
                layout={mainPlotLayout}
              />
            )}
          </div>

          {/* Sigmoid Curve Plotly Div */}
          <div className="relative h-[350px] border border-gray-700 rounded-lg overflow-hidden bg-gray-800">
            <h3 className="text-lg font-semibold mb-2 p-2">Predicted Probability vs. Linear Output (Sigmoid)</h3>
            {dataX.length > 0 && dataY.length > 0 && weightsHistory.length > 0 && (
              <DynamicPlotlyGraph
                divRef={sigmoidPlotRef}
                plotData={sigmoidPlotData}
                layout={sigmoidPlotLayout}
              />
            )}
          </div>

          {/* Loss Graph Plotly Div */}
          <div className="relative h-[250px] border border-gray-700 rounded-lg overflow-hidden bg-gray-800">
            <h3 className="text-lg font-semibold mb-2 p-2">Classification Error (Binary Cross-Entropy Loss)</h3>
            {lossHistory.length > 0 && (
              <DynamicPlotlyGraph
                divRef={lossPlotRef}
                plotData={lossPlotData}
                layout={lossPlotLayout}
              />
            )}
          </div>

          {/* Animation & Progress Controls */}
          <div className="flex items-center justify-between mt-4">
            <button
              onClick={handleAnimateToggle}
              className="px-4 py-2 bg-black text-white rounded border border-white hover:bg-white hover:text-black transition"
            >
              {isAnimating ? "Pause" : "Play"}
            </button>
            <input
              type="range"
              min={0}
              max={params.iterations}
              step={1}
              value={frame}
              onChange={(e) => {
                setIsAnimating(false); // Pause animation when manually moving slider
                setFrame(Number(e.target.value));
              }}
              className="w-full mx-4 slider"
              // Ensure '--value' is correctly set for slider styling
              style={{ "--value": `${(frame / params.iterations) * 100}%` } as React.CSSProperties}
            />
            <span className="text-sm text-gray-400">
              {frame}/{params.iterations} iterations
            </span>
          </div>
          <div className="w-full bg-gray-700 rounded-full h-2.5 mt-2">
            <div
              className="bg-blue-600 h-2.5 rounded-full transition-all duration-100 ease-linear"
              style={{ width: `${(frame / params.iterations) * 100}%` }}
            ></div>
          </div>
          <p className="text-sm text-gray-400 mt-1 text-right">
            Current Loss: {currentLoss.toFixed(4)}
          </p>

          {/*Controls Section*/}
          <section className="controls-section p-4 border bg-black-900">
            <h3 className="text-xl font-semibold text-white mb-4">Controls</h3>

            {paramControls.map((ctl) => {
              if (ctl.type === 'select' && ctl.options) { // Render select/dropdown for options
                return (
                  <div key={ctl.name} className="control-row">
                    <span className="control-label">{ctl.label}</span>
                    <select
                      value={params[ctl.name as keyof Params].toString()} // Convert to string for value prop
                      onChange={(e) => {
                        let value: string | number = e.target.value;
                        if (typeof params[ctl.name] === 'number') { // Convert back to number if original type is number
                            value = Number(value);
                        }
                        setParams((p) => ({ ...p, [ctl.name]: value as Params[keyof Params] }));
                        setIsAnimating(false); // Pause animation when changing params
                      }}
                      className="ml-4 p-2 rounded bg-gray-800 border border-gray-600 text-white"
                    >
                      {ctl.options.map((option) => (
                        <option key={option.value.toString()} value={option.value.toString()}>
                          {option.label}
                        </option>
                      ))}
                    </select>
                  </div>
                );
              } else { // Render range slider for min/max/step controls
                const val = params[ctl.name as keyof Params] as number;
                const pct = ((val - (ctl.min || 0)) / ((ctl.max || 1) - (ctl.min || 0))) * 100;
                return (
                  <div key={ctl.name} className="control-row">
                    <span className="control-label">{ctl.label}</span>
                    <input
                      type="range"
                      min={ctl.min}
                      max={ctl.max}
                      step={ctl.step}
                      value={val}
                      onChange={(e) => {
                        setParams((p) => ({
                          ...p,
                          [ctl.name]: Number(e.target.value),
                        }));
                        setIsAnimating(false); // Pause animation when changing params
                      }}
                      className="slider"
                      style={{ "--value": `${pct}%` } as React.CSSProperties}
                    />
                    <span className="control-value">{val}</span>
                  </div>
                );
              }
            })}
          </section>
        </div>

        {/* Right Section: Text Content & Controls */}
        <div className="space-y-6 lg:w-2/5">
          <div>
            <h2 className="text-2xl font-semibold mb-2">How It Works</h2>
            <p className="text-gray-400">
              Logistic Regression models the probability of a binary outcome (0 or 1). It uses a sigmoid function to
              squash the linear combination of features into a probability between 0 and 1. During training, the model
              adjusts its weights (coefficients) and bias using Gradient Descent to minimize the Binary Cross-Entropy
              loss, which measures the difference between predicted probabilities and true labels.
            </p>
          </div>

          <div>
            <h3 className="text-xl font-semibold mb-2">Key Concepts</h3>
            <ul className="list-disc list-inside space-y-2 text-gray-400">
              <li>
                <strong>Sigmoid Function:</strong> Transforms linear output into a probability (0-1).
              </li>
              <li>
                <strong>Decision Boundary:</strong> The line (or hyperplane in higher dimensions) that separates the
                classes. For 2 features, it's a straight line.
              </li>
              <li>
                <strong>Binary Cross-Entropy Loss:</strong> A common loss function for binary classification, which the
                model aims to minimize.
              </li>
              <li>
                <strong>Gradient Descent:</strong> An optimization algorithm used to iteratively adjust weights by
                moving in the direction opposite to the gradient of the loss function.
              </li>
            </ul>
          </div>

          <div>
            <h3 className="text-xl font-semibold mb-2">Math Behind It</h3>
            <BlockMath math={`P(y=1|x) = \\sigma(w \\cdot x + b)`} />
            <BlockMath math={`\\sigma(z) = \\frac{1}{1 + e^{-z}}`} />
            <BlockMath math={`\\text{Loss} = -\\frac{1}{N} \\sum_{i=1}^N [y_i \\log(p_i) + (1-y_i) \\log(1-p_i)]`} />
            <BlockMath math={`w_{new} = w_{old} - \\alpha \\frac{\\partial \\text{Loss}}{\\partial w}`} />
            <p className="text-gray-400 text-sm mt-2">
              Where: <br />
              `P(y=1|x)` is the probability of class 1 given features `x`. <br />
              `\\sigma` is the sigmoid function. <br />
              `w` are the weights (coefficients), `b` is the bias (intercept). <br />
              `N` is the number of samples. <br />
              `y_i` is the true label for sample `i`, `p_i` is the predicted probability. <br />
              `\\alpha` is the learning rate.
            </p>
          </div>

          <div>
            <h3 className="text-xl font-semibold mb-2">Color Guide</h3>
            <ul className="list-disc list-inside space-y-2 text-gray-400">
              <li><span className="text-blue-400">Blue Circles:</span> Class 0 Data Points (e.g., Benign Tumor)</li>
              <li><span className="text-red-400">Red Circles:</span> Class 1 Data Points (e.g., Malignant Tumor)</li>
              <li><span className="text-green-400">Green Line/Surface:</span> Decision Boundary</li>
              <li><span className="text-purple-400">Purple Line:</span> Binary Cross-Entropy Loss over Iterations</li>
              <li><span className="text-orange-400">Orange Line:</span> Sigmoid Function Curve</li>
            </ul>
          </div>

          <div>
            <h3 className="text-xl font-semibold mb-2">Applications</h3>
            <p className="text-gray-400">Logistic Regression is widely used in:</p>
            <ul className="list-disc list-inside space-y-2 text-gray-400">
              <li>Medical diagnosis (e.g., predicting disease presence)</li>
              <li>Credit scoring and fraud detection</li>
              <li>Email spam detection</li>
              <li>Marketing (e.g., predicting customer churn)</li>
            </ul>
          </div>
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
      `}</style>
    </div>
  );
}