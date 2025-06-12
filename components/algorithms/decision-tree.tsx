"use client";

import { useState, useEffect, useRef } from "react";
import "katex/dist/katex.min.css";
import { BlockMath } from "react-katex";
import { FiRefreshCw } from "react-icons/fi";

type Node = {
  feature: number | null;
  threshold: number | null;
  left: Node | null;
  right: Node | null;
  value: number | null;
  impurity: number | null;
};

type DatasetType = "iris" | "wine" | "digits";

const classColors: Record<number, string> = {
  0: "red",
  1: "blue",
  2: "green",
  3: "orange",
  4: "violet",
  5: "fuchsia",
  6: "brown",
  7: "gray",
  8: "indigo",   // Replaces invalid 'cyan'
  9: "lime"
};

const featureNamesMap: Record<DatasetType, string[]> = {
  iris: ["sepalLength", "sepalWidth", "petalLength", "petalWidth"],
  wine: ["alcohol", "malic acid", "ash", "alcalinity", "magnesium", "phenols", "flavanoids", "nonflavanoid phenols", "proanthocyanins", "color intensity", "hue", "OD280/OD315", "proline"],
  digits: Array.from({ length: 64 }, (_, i) => `pixel${i}`)
};

const DecisionTreeVisualizer = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [dataset, setDataset] = useState<DatasetType>("iris");
  const [data, setData] = useState<any[]>([]);
  const [tree, setTree] = useState<Node | null>(null);
  const [featureNames, setFeatureNames] = useState<string[]>([]);

  const [params, setParams] = useState({
    maxDepth: 3,
    minSamplesSplit: 2,
    criterion: "gini",
    treeHeight: 600
  });
  const [selectedDataset, setSelectedDataset] = useState("iris"); // default
  
 useEffect(() => {
  if (!selectedDataset) return;

  fetch("${process.env.NEXT_PUBLIC_API_URL}/build_tree", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      dataset: selectedDataset,
      params: {
        max_depth: params.maxDepth,
        min_samples_split: params.minSamplesSplit,
        criterion: params.criterion,
      },
    }),
  })
    .then((res) => res.json())
    .then((json) => {
      setTree(json.tree);
      setFeatureNames(json.feature_names); // you’ll need a useState for this
    })
    .catch((err) => {
      console.error("Failed to fetch tree:", err);
    });
}, [params, selectedDataset]);



  // Draw tree
  useEffect(() => {
    if (!tree) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const drawNode = (node: Node | null, x: number, y: number, dx: number, dy: number) => {
      if (!node) return;

      const radius = 20;

      if (node.left) {
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x - dx, y + dy);
        ctx.strokeStyle = "#ccc";
        ctx.stroke();
        drawNode(node.left, x - dx, y + dy, dx / 2, dy);
      }

      if (node.right) {
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x + dx, y + dy);
        ctx.strokeStyle = "#ccc";
        ctx.stroke();
        drawNode(node.right, x + dx, y + dy, dx / 2, dy);
      }

      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      const isLeaf = !node.left && !node.right;
      const classColor = typeof node.value === "number" ? classColors[node.value] ?? "gray" : "gray";
      ctx.fillStyle = isLeaf ? classColor : "purple";


      ctx.fill();
      ctx.strokeStyle = "#fff";
      ctx.stroke();

      ctx.fillStyle = "#fff";
      ctx.font = "12px Arial";
      ctx.textAlign = "center";
      if (typeof node.impurity === "number") {
  ctx.fillText(node.impurity.toFixed(2), x, y + 4);
}


      if (node.feature !== null && node.threshold !== null) {
        const name = featureNames[node.feature] ?? `f${node.feature}`;
        ctx.fillText(name, x, y - 35);
        ctx.fillText(`≤ ${node.threshold?.toFixed?.(2) ?? ""}`, x, y - 22);
      }
    };

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawNode(tree, canvas.width / 2, 50, canvas.width / 5, 80);
  }, [tree]);

  return (
    <div className="p-4 space-y-10">
      <h1 className="text-4xl font-bold mb-4">Decision Tree Visualizer</h1>
      <p className="text-lg text-muted-foreground">
        A decision tree classifies values by splitting data based on features available in each dataset. Please pick the dataset below.
      </p>

      

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <canvas
            ref={canvasRef}
            width={700}
            height={params.treeHeight - 50} 
            className="w-full border border-gray-300 rounded-lg bg-black"
          />

<section className="controls-section">
          <h3 className="text-xl font-semibold text-white mb-4">Controls</h3>
          <div className="control-row">
            <span className="control-label">Select Dataset </span>
        <select
          value={selectedDataset}
          onChange={(e) => setSelectedDataset(e.target.value)}
          className="styled-select w-80 mx-10" 
        >   
          <option value="iris">Iris</option>
          <option value="wine">Wine</option>
          <option value="digits">Digits</option>
        </select>
      <br/><br/>
          </div>
          <div className="control-row">
            
            <span className="control-label">Max Depth</span>
            <input
              type="range"
              min="1"
              max="6"
              value={params.maxDepth}
              onChange={(e) =>
                setParams({ ...params, maxDepth: +e.target.value })
              }
              style={
                {
                  "--value": `${((params.maxDepth - 1) / 5) * 100}%`,
                } as React.CSSProperties
              }
              className="slider"
            />
            <span className="control-value">{params.maxDepth}</span>
          </div>

          <div className="control-row">
            <span className="control-label">Min Samples Split</span>
            <input
              type="range"
              min="2"
              max="10"
              value={params.minSamplesSplit}
              onChange={(e) =>
                setParams({ ...params, minSamplesSplit: +e.target.value })
              }
              style={
                {
                  "--value": `${((params.minSamplesSplit - 2) / 8) * 100}%`,
                } as React.CSSProperties
              }
              className="slider"
            />
            <span className="control-value">{params.minSamplesSplit}</span>
          </div>

          <div className="control-row">
            <span className="control-label">Criterion</span>
            <select
              value={params.criterion}
              onChange={(e) =>
                setParams({ ...params, criterion: e.target.value })
              }
              className="styled-select"
            >
              <option value="gini">Gini</option>
              <option value="entropy">Entropy</option>
              <option value="gain_ratio">Gain Ratio</option>
            </select>
          </div>

          <button
            className="reset-btn"
            onClick={() =>
              setParams({
                ...params,
                maxDepth: 3,
                minSamplesSplit: 2,
                criterion: "Gini",
              })
            }
          >
            <FiRefreshCw className="inline-block mr-2 h-5 w-5" />
            Reset
          </button>
        </section>
      </div>

      {/* ——— here’s your embedded CSS ——— */}
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
        .reset-btn {
          background: black;
          color: white;
          padding: 0.5rem 1rem;
          border: 1px solid #444;
          border-radius: 4px;
          cursor: pointer;
        }
        .reset-btn:hover {
          border-color: #666;
        }
      `}</style>

        <div className="space-y-6">
          <section>
            <h2 className="text-2xl font-semibold">How It Works</h2>
            <p className="text-muted-foreground">
              Each node shows the impurity score, and the predicted class is indicated by color.
              If a split occurs, the feature used is shown above the node.
            </p>
          </section>

          <section>
            <h3 className="text-xl font-semibold">Formulas</h3>
            <p className="text-muted-foreground">Gini Impurity:</p>
            <BlockMath math="Gini = 1 - \sum_{i=1}^{n} p_i^2" />
            <p className="text-muted-foreground">Information Gain:</p>
            <BlockMath math="Gain = Impurity(parent) - \sum_{k} \frac{n_k}{n} Impurity(k)" />
            <p className="text-muted-foreground">Gain Ratio:</p>
            <BlockMath math={`
\\text{SplitInfo}_A(D) = - \\sum_{j=1}^{v} \\frac{|D_j|}{|D|} \\log_2\\left(\\frac{|D_j|}{|D|}\\right)
`} />


            <BlockMath math="Gain Ratio(A) = \frac{Gain(A)}{SplitInfo(A)}" />
          </section>

          

          <section>
  <h3 className="text-xl font-semibold">Node Colors</h3>

  <p className="font-semibold mb-2">
  <span className="text-purple-600">Purple:</span>
  <span className="text-gray-400"> Internal (split) node</span>
</p>

  <div className="space-y-4 text-muted-foreground">
    <div>
      <h4 className="font-semibold">Iris Dataset</h4>
      <ul className="list-disc list-inside">
        <li><span className="text-red-600 font-semibold">Red:</span> Setosa</li>
        <li><span className="text-blue-600 font-semibold">Blue:</span> Versicolor</li>
        <li><span className="text-green-600 font-semibold">Green:</span> Virginica</li>
      </ul>
    </div>

    <div>
      <h4 className="font-semibold">Digits Dataset</h4>
      <ul className="list-disc list-inside">
        <li><span className="text-red-600 font-semibold">Red:</span> Digit 0</li>
        <li><span className="text-blue-600 font-semibold">Blue:</span> Digit 1</li>
        <li><span className="text-green-600 font-semibold">Green:</span> Digit 2</li>
        <li><span className="text-orange-600 font-semibold">Orange:</span> Digit 3</li>
        <li><span className="text-violet-600 font-semibold">Violet:</span> Digit 4</li>
        <li><span className="text-pink-600 font-semibold">Pink:</span> Digit 5</li>
        <li><span className="text-amber-600 font-semibold">Brown:</span> Digit 6</li>
        <li><span className="text-gray-600 font-semibold">Gray:</span> Digit 7</li>
        <li><span className="text-indigo-600 font-semibold">Indigo:</span> Digit 8</li>
        <li><span className="text-lime-600 font-semibold">Lime:</span> Digit 9</li>
      </ul>
    </div>

    <div>
      <h4 className="font-semibold">Wine Dataset</h4>
      <ul className="list-disc list-inside">
        <li><span className="text-red-600 font-semibold">Red:</span> Class 0</li>
        <li><span className="text-blue-600 font-semibold">Blue:</span> Class 1</li>
        <li><span className="text-green-600 font-semibold">Green:</span> Class 2</li>
      </ul>
    </div>
  </div>
</section>


          <section>
            <h3 className="text-xl font-semibold">Interactive Controls</h3>
            <ul className="list-disc list-inside text-muted-foreground">
              <li><strong>Max Depth:</strong> Controls the maximum depth of the tree. Increasing the depth allows the tree to make more splits, but may lead to overfitting.</li>
              <li><strong>Min Samples Split:</strong> Defines the minimum number of samples required to split an internal node. A higher value can lead to simpler trees.</li>
              <li><strong>Criterion:</strong> The function used to measure the quality of a split.</li>
            </ul>
          </section>

          <section>
            <h3 className="text-xl font-semibold">Applications</h3>
            <p className="text-muted-foreground">Decision trees are widely used in:</p>
            <ul className="list-disc list-inside text-muted-foreground">
              <li>Banking (fraud detection, credit scoring)</li>
              <li>Medicine (disease diagnosis, treatment recommendation)</li>
              <li>E‑commerce (customer segmentation, recommendation systems)</li>
              <li>Computer Vision (object classification, image segmentation)</li>
            </ul>
          </section>

        </div>
      </div>
    </div>
  );
}

export default DecisionTreeVisualizer
