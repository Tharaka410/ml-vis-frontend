"use client"

import React, { useEffect, useRef, useState } from "react";
import { BlockMath } from "react-katex";
import Chart from 'chart.js/auto'; // Import Chart.js core
import { LineController, LinearScale, CategoryScale, PointElement, LineElement, Title, Tooltip, Legend } from 'chart.js';

// Register the components Chart.js needs for a line chart
Chart.register(LineController, LinearScale, CategoryScale, PointElement, LineElement, Title, Tooltip, Legend);

type Params = {
  // learningRate is now primarily for the simulated history animation, not direct GD
  learningRate: number;
  iterations: number; // Number of animation steps for the simulated history
  noise: number;
};

export default function LinearRegressionPage() {
  const [params, setParams] = useState<Params>({
    learningRate: 0.1,
    iterations: 100, // Default animation iterations
    noise: 0.2,
  });

  const [isAnimating, setIsAnimating] = useState(false);
  const [currentAnimationFrame, setCurrentAnimationFrame] = useState(0); // Renamed from 'frame' for clarity
  const reqRef = useRef<number | null>(null);

  const [dataPoints, setDataPoints] = useState<Array<{ x: number; y: number }>>([]);
  const [finalRegressionCoef, setFinalRegressionCoef] = useState<number>(0); // Final 'w' from scikit-learn
  const [finalRegressionIntercept, setFinalRegressionIntercept] = useState<number>(0); // Final 'b' from scikit-learn
  const [finalMse, setFinalMse] = useState<number>(0); // Final MSE from scikit-learn

  const [simulatedMseHistory, setSimulatedMseHistory] = useState<number[]>([]);
  const [simulatedWHistory, setSimulatedWHistory] = useState<number[]>([]);
  const [simulatedBHistory, setSimulatedBHistory] = useState<number[]>([]);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const errorChartRef = useRef<HTMLCanvasElement | null>(null);
  const chartInstance = useRef<Chart | null>(null);

  const paramControls = [
    { name: "iterations", label: "Animation Steps", min: 10, max: 200, step: 10 },
    { name: "noise", label: "Noise", min: 0, max: 1, step: 0.05 },
  ] as const;

  // Generate data points and fetch results from backend
  useEffect(() => {
    const newPoints: { x: number; y: number }[] = [];
    for (let i = 0; i < 100; i++) {
      const x = Math.random() * 2 - 1;
      const y = 2 * x + 1 + (Math.random() * 2 - 1) * params.noise;
      newPoints.push({ x, y });
    }
    setDataPoints(newPoints);

    // Reset states when data changes
    setFinalRegressionCoef(0);
    setFinalRegressionIntercept(0);
    setFinalMse(0);
    setSimulatedMseHistory([]);
    setSimulatedWHistory([]);
    setSimulatedBHistory([]);
    setCurrentAnimationFrame(0); // Reset animation frame

    const fetchAllRegressionResults = async () => {
      if (newPoints.length === 0) return;

      const X_data = newPoints.map(p => [p.x]);
      const y_data = newPoints.map(p => p.y);

      try {
        const finalResponse = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/linear-regression`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ X: X_data, y: y_data }),
        });
        const finalResult = await finalResponse.json();
        console.log("Backend final model result:", finalResult);

        if (finalResult && Array.isArray(finalResult.coef) && typeof finalResult.intercept === 'number') {
            setFinalRegressionCoef(finalResult.coef[0]);
            setFinalRegressionIntercept(finalResult.intercept);
            setFinalMse(finalResult.final_mse);
        } else {
            console.error("Malformed final regression response:", finalResult);
        }

        // Fetch simulated history for animation and error graph
        const historyResponse = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/linear-regression-history`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ X: X_data, y: y_data, iterations: params.iterations }),
        });
        const historyResult = await historyResponse.json();
        console.log("Backend history result:", historyResult);

        if (historyResult && Array.isArray(historyResult.mse_history) &&
            Array.isArray(historyResult.w_history) && Array.isArray(historyResult.b_history)) {
            setSimulatedMseHistory(historyResult.mse_history);
            setSimulatedWHistory(historyResult.w_history);
            setSimulatedBHistory(historyResult.b_history);
        } else {
            console.error("Malformed history regression response:", historyResult);
        }

      } catch (error) {
        console.error("Error fetching linear regression results:", error);
      }
    };

    fetchAllRegressionResults();
  }, [params.noise, params.iterations]); // Re-run when noise or animation steps change

  // Animation loop
  useEffect(() => {
    let animationId: number | null = null;

    const animate = () => {
      setCurrentAnimationFrame(prev => {
        if (prev >= params.iterations -1) { 
          setIsAnimating(false);
          return prev;
        }
        animationId = requestAnimationFrame(animate);
        return prev + 1;
      });
    };

    if (isAnimating && simulatedMseHistory.length > 0) { // Only start if history is loaded
      animationId = requestAnimationFrame(animate);
    } else {
      if (reqRef.current !== null) cancelAnimationFrame(reqRef.current); 
      reqRef.current = null; 
    }

    // Cleanup function
    return () => {
      if (animationId !== null) cancelAnimationFrame(animationId);
    };
  }, [isAnimating, params.iterations, simulatedMseHistory.length]);

  // Render linear regression graph
  useEffect(() => {
    const c = canvasRef.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    if (!ctx) return;

    // Use simulated w and b from history for current animation frame
    const currentW = simulatedWHistory[currentAnimationFrame] !== undefined
                     ? simulatedWHistory[currentAnimationFrame]
                     : finalRegressionCoef; // Fallback to final if history not ready

    const currentB = simulatedBHistory[currentAnimationFrame] !== undefined
                     ? simulatedBHistory[currentAnimationFrame]
                     : finalRegressionIntercept; // Fallback to final if history not ready


    renderLinearRegression(ctx, c.width, c.height, dataPoints, currentW, currentB,
                           currentAnimationFrame, params.iterations,
                           simulatedMseHistory[currentAnimationFrame] !== undefined
                           ? simulatedMseHistory[currentAnimationFrame]
                           : finalMse); // Pass current MSE from history or final MSE
  }, [dataPoints, currentAnimationFrame, params.iterations, simulatedWHistory, simulatedBHistory, simulatedMseHistory, finalRegressionCoef, finalRegressionIntercept, finalMse]);

  // Render error chart
  useEffect(() => {
    const errCtx = errorChartRef.current?.getContext("2d");
    if (!errCtx) return;

    if (chartInstance.current) {
      chartInstance.current.destroy(); // Destroy existing chart
    }

    chartInstance.current = new Chart(errCtx, {
      type: 'line',
      data: {
        labels: Array.from({ length: simulatedMseHistory.length }, (_, i) => i + 1),
        datasets: [{
          label: 'Mean Squared Error',
          data: simulatedMseHistory,
          borderColor: '#ff3860',
          tension: 0.1,
          pointRadius: 0
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            title: {
              display: true,
              text: 'Iteration',
              color: 'white'
            },
            ticks: { color: 'white' },
            grid: { color: '#333' }
          },
          y: {
            title: {
              display: true,
              text: 'MSE',
              color: 'white'
            },
            ticks: { color: 'white' },
            grid: { color: '#333' }
          }
        },
        plugins: {
          legend: {
            labels: {
              color: 'white'
            }
          }
        }
      }
    });

  }, [simulatedMseHistory]);


  const renderLinearRegression = (
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    points: { x: number; y: number }[],
    w: number,
    b: number,
    currentIteration: number,
    totalIterations: number,
    mse: number
  ) => {
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, width, height);

    const margin = 40;
    const plotWidth = width - margin * 2;
    const plotHeight = height - margin * 2;

    // Draw axes
    ctx.strokeStyle = "#666";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(margin, height / 2);
    ctx.lineTo(width - margin, height / 2);
    ctx.moveTo(width / 2, margin);
    ctx.lineTo(width / 2, height - margin);
    ctx.stroke();

    // Plot line using w and b
    ctx.strokeStyle = "#ff3860";
    ctx.lineWidth = 4;
    ctx.beginPath();
    const x_min_plot = -1;
    const x_max_plot = 1;

    const y_start = w * x_min_plot + b;
    const y_end = w * x_max_plot + b;

    const cx_start = margin + ((x_min_plot + 1) / 2) * plotWidth;
    const cy_start = height / 2 - y_start * plotHeight / 2;
    const cx_end = margin + ((x_max_plot + 1) / 2) * plotWidth;
    const cy_end = height / 2 - y_end * plotHeight / 2;

    ctx.moveTo(cx_start, cy_start);
    ctx.lineTo(cx_end, cy_end);
    ctx.stroke();

    // Points
    for (const pt of points) {
      const cx = margin + ((pt.x + 1) / 2) * plotWidth;
      const cy = height / 2 - pt.y * plotHeight / 2;
      ctx.beginPath();
      ctx.arc(cx, cy, 4, 0, Math.PI * 2);
      ctx.fillStyle = "#34d399";
      ctx.fill();
    }

    ctx.fillStyle = "#fff";
    ctx.font = "14px sans-serif";
    ctx.fillText(`Weight (w): ${w.toFixed(3)}`, margin + 10, margin + 20);
    ctx.fillText(`Bias (b): ${b.toFixed(3)}`, margin + 10, margin + 40);
    ctx.fillText(`Animation Progress: ${currentIteration}/${totalIterations}`, margin + 10, margin + 60);
    ctx.fillText(`MSE: ${mse.toFixed(3)}`, margin + 10, margin + 80);
  };

  return (
    <div className="space-y-8 p-4 bg-black text-white">
      <h1 className="text-4xl font-bold">Linear Regression</h1>
      <p className="text-gray-300">
        A simple model that predicts continuous values using a linear relationship.
      </p>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2">
          <canvas
            ref={canvasRef}
            width={1000}
            height={600}
            className="w-full border border-gray-700"
          />
          <button
            onClick={() => {
              setIsAnimating((a) => {
                if (!a) { // If starting animation
                    setCurrentAnimationFrame(0); // Reset animation counter
                    // Note: simulatedMseHistory etc. are reset in the data useEffect when noise changes
                }
                return !a;
              });
            }}
            className="mt-2 px-4 py-2 bg-black text-white rounded border border-white hover:bg-white hover:text-black transition"
          >
            {isAnimating ? "Pause Animation" : "Start Animation"}
          </button>

          <section className="controls-section">
            <h3 className="text-xl font-semibold text-white mb-4">Controls</h3>

            {paramControls.map((ctl) => {
              const val = params[ctl.name];
              const pct = ((val - ctl.min) / (ctl.max - ctl.min)) * 100;
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

        <div className="space-y-6">
          <h2 className="text-2xl font-semibold">How It Works</h2>
          <p>
            Linear regression fits a straight line to data by minimizing the mean squared error between predictions and true values.
          </p>
          <h3 className="text-xl font-semibold">Formula</h3>
          <div className="text-2xl">
  <BlockMath math="y = wx + b" />
</div>

          <p>
            It learns <code>w</code> and <code>b</code> by finding the optimal fit to the data using algorithms like Ordinary Least Squares (used by scikit-learn). The animation demonstrates a simulated iterative approach.
          </p>
          <h3 className="text-xl font-semibold">Applications</h3>
          <ul className="list-disc list-inside">
            <li>Forecasting</li>
            <li>Risk analysis</li>
            <li>Real estate pricing</li>
            <li>Trend prediction</li>
          </ul>
        </div>
      </div>

      <div className="mt-8">
        <h2 className="text-2xl font-semibold mb-4">Error (MSE) Over Animation Progress</h2>
        <div className="bg-gray-800 p-4 rounded-lg" style={{ height: '400px' }}>
          <canvas ref={errorChartRef}></canvas>
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