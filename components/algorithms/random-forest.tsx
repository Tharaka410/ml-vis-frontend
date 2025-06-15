// random-forest.tsx
"use client"
import { useState, useEffect, useRef } from "react";
import "katex/dist/katex.min.css";
import { BlockMath } from "react-katex";
import { FiRefreshCw } from 'react-icons/fi';

type DatasetType = "iris" | "wine" | "digits";

type Point = {
  [key: string]: number | string;
  label: string;
};

type Node = {
  name: string;
  left?: Node;
  right?: Node;
  samples?: number;
  value?: number;
  impurity?: number;
  feature?: string;
  threshold?: number;
};

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

// Mock data moved outside component to ensure it's always available
const mockData: Record<DatasetType, Point[]> = {
  iris: Array.from({ length: 150 }, (_, i) => ({
    sepalLength: Math.random() * 3 + 4,
    sepalWidth: Math.random() * 2 + 2,
    petalLength: Math.random() * 4 + 1,
    petalWidth: Math.random() * 1 + 0.1,
    label: (i % 3).toString(),
  })),
  wine: Array.from({ length: 178 }, (_, i) => ({
    alcohol: Math.random() * 5 + 11,
    'malic acid': Math.random() * 2 + 0.5,
    ash: Math.random() * 1 + 1.5,
    alcalinity: Math.random() * 5 + 10,
    magnesium: Math.random() * 50 + 70,
    phenols: Math.random() * 2 + 1,
    flavanoids: Math.random() * 2 + 0.5,
    'nonflavanoid phenols': Math.random() * 0.5 + 0.1,
    proanthocyanins: Math.random() * 1 + 0.5,
    'color intensity': Math.random() * 5 + 2,
    hue: Math.random() * 0.5 + 0.5,
    'OD280/OD315': Math.random() * 2 + 1.5,
    proline: Math.random() * 800 + 300,
    label: (i % 3).toString(),
  })),
  digits: Array.from({ length: 1797 }, (_, i) => {
    const pixels: { [key: string]: number } = {};
    for (let p = 0; p < 64; p++) {
      pixels[`pixel${p}`] = Math.floor(Math.random() * 16);
    }
    return { ...pixels, label: (i % 10).toString() };
  }),
};

// Helper function to transform backend node to frontend Node - MODIFIED
const transformNode = (backendNode: any): Node => { // Removed featureNames param as it's not needed for feature string
  let name: string;
  let feature: string | undefined;
  let threshold: number | undefined;
  let value: number | undefined;

  if (backendNode.feature === null) { // It's a leaf node
    name = `Leaf: ${backendNode.value}`;
    value = backendNode.value === null ? undefined : backendNode.value;
  } else { 
    feature = backendNode.feature;
    threshold = backendNode.threshold === null ? undefined : backendNode.threshold;
    name = `${feature} <= ${threshold?.toFixed(2) || 'N/A'}`;
  }

  const newNode: Node = {
    name: name,
    impurity: backendNode.impurity === null ? undefined : backendNode.impurity,
    samples: backendNode.samples === null ? undefined : backendNode.samples,
    feature: feature,
    threshold: threshold,
    value: value
  };

  if (backendNode.left) {
    newNode.left = transformNode(backendNode.left); // Removed featureNames param
  }
  if (backendNode.right) {
    newNode.right = transformNode(backendNode.right); // Removed featureNames param
  }
  return newNode;
};


// The predict function for a single tree
function predict(tree: Node, point: Point): number {
  try {
    let node = tree;
    while (node && !node.name.startsWith("Leaf:")) {
      const feature = node.feature;
      const threshold = node.threshold;

      if (feature === undefined || threshold === undefined) {
         const parts = node.name.split(" <= ");
         if (parts.length < 2) throw new Error("Invalid node name format for split node");
         const parsedFeatureName = parts[0];
         const parsedThreshold = parseFloat(parts[1]);

         if (isNaN(parsedThreshold)) throw new Error("Invalid threshold in node name");

         const value = point[parsedFeatureName] as number;
         if (isNaN(value)) throw new Error(`Feature ${parsedFeatureName} not found in point or is not a number`);

         node = value <= parsedThreshold ? node.left! : node.right!;
      } else {
         const value = point[feature] as number;
         if (isNaN(value)) throw new Error(`Feature ${feature} not found in point or is not a number`);
         node = value <= threshold ? node.left! : node.right!;
      }

      if (!node) throw new Error("Invalid tree structure: next node is undefined");
    }
    if (node.name.startsWith("Leaf:")) {
      const classValueStr = node.name.split(": ")[1];
      const classValue = parseInt(classValueStr);
      if (isNaN(classValue)) throw new Error("Leaf node name has invalid class value");
      return classValue;
    } else {
      throw new Error("Prediction reached a non-leaf node without a split condition");
    }
  } catch (error) {
    console.error("Prediction error:", error);
    return -1;
  }
}

// Function to render a single decision tree on a canvas - KEPT AS IS, added classColors argument
function renderDecisionTree(ctx: CanvasRenderingContext2D, width: number, height: number, treeIndex: number, forest: Node[], classColors: Record<number, string>) {
  try {
    ctx.clearRect(0, 0, width, height);
    const tree = forest[treeIndex];
    if (!tree) return;

    ctx.font = "14px Arial";
    ctx.fillStyle = "#fff";
    ctx.textAlign = "center";

    function drawNode(node: Node | undefined, x: number, y: number, dx: number, dy: number) {
      if (!node) return;

      const isLeaf = node.name.startsWith("Leaf:");
      const classValue = isLeaf ? parseInt(node.name.split(": ")[1]) : null;

      if (node.left) {
        ctx.strokeStyle = "white";
        ctx.beginPath();
        ctx.moveTo(x, y + 20);
        ctx.lineTo(x - dx, y + dy - 20);
        ctx.stroke();
        drawNode(node.left, x - dx, y + dy, dx / 2, dy);
      }
      if (node.right) {
        ctx.strokeStyle = "white";
        ctx.beginPath();
        ctx.moveTo(x, y + 20);
        ctx.lineTo(x + dx, y + dy - 20);
        ctx.stroke();
        drawNode(node.right, x + dx, y + dy, dx / 2, dy);
      }

      ctx.beginPath();
      ctx.arc(x, y, 20, 0, Math.PI * 2);
      ctx.fillStyle = isLeaf && classValue !== null ? classColors[classValue] : "purple";
      ctx.fill();

      ctx.strokeStyle = "#fff";
      ctx.stroke();

      ctx.fillStyle = "#fff";
      ctx.font = "12px Arial";

      if (isLeaf) {
        if (node.impurity !== undefined) {
            ctx.fillText(`${node.impurity.toFixed(2)}`, x, y + 5);
        }
      } else {
        const featureToDisplay = node.feature || node.name.split(" <= ")[0];
        const thresholdToDisplay = node.threshold !== undefined ? node.threshold.toFixed(2) : node.name.split(" <= ")[1];

        ctx.fillText(featureToDisplay, x, y - 40);
        ctx.fillText(`≤ ${thresholdToDisplay}`, x, y - 25);
        if (node.impurity !== undefined) {
            ctx.fillText(`${node.impurity.toFixed(2)}`, x, y + 5);
        }
      }
    }

    drawNode(forest[treeIndex], width / 2, 50, width / 5, 100);
  } catch (error) {
    console.error("Rendering error:", error);
    ctx.fillStyle = "red";
    ctx.fillText("Rendering Error", width / 2, height / 2);
  }
}

export default function RandomForestPage() {
  const [params, setParams] = useState({
    dataset: "iris" as DatasetType,
    numberOfTrees: 10,
    maxDepth: 5,
    minSamplesSplit: 2,
    criterion: "gini",
    subsampleRatio: 0.8,
    featureSubsetRatio: 0.7,
  });

  const [dataset, setDataset] = useState<Point[]>([]);
  const [forest, setForest] = useState<Node[]>([]);
  const [selectedIndex, setSelectedIndex] = useState<number>(0);
  const [prediction, setPrediction] = useState<number | null>(null);
  const [individualVotes, setIndividualVotes] = useState<number[]>([]);
  const [targetNames, setTargetNames] = useState<string[]>([]);

  // Initialize canvasRefs.current as an empty array at the start
  const canvasRefs = useRef<Array<HTMLCanvasElement | null>>([]);

  useEffect(() => {
    setDataset(mockData[params.dataset]);
    setSelectedIndex(0); // Reset selected index when dataset changes
  }, [params.dataset]);

useEffect(() => {
  console.log("Sending forest fetch request...");
  fetch(`${process.env.NEXT_PUBLIC_API_URL}/build_forest`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      dataset: params.dataset,
      params: { // This now correctly matches the backend's ForestParams
        n_trees: params.numberOfTrees, // Use n_trees
        subsample_ratio: params.subsampleRatio, // Use subsample_ratio
        feature_subset: params.featureSubsetRatio, // Use feature_subset
        max_depth: params.maxDepth,
        min_samples_split: params.minSamplesSplit,
        criterion: params.criterion,
      },
    }),
  })
    .then((res) => {
      if (!res.ok) { // Check for HTTP errors (e.g., 400, 500)
        return res.json().then(errorData => {
          throw new Error(errorData.detail || 'Backend error');
        });
      }
      return res.json();
    })
    .then((data) => {
      console.log("Full response:", data);

      if (!data || !Array.isArray(data.trees)) {
        console.error("Expected 'trees' array in response, but got:", data);
        setForest([]);
        setTargetNames([]);
        return;
      }
      // MODIFIED: Call transformNode without featureNames, as it's not needed
      const transformedForest = data.trees.map((t: any) =>
          transformNode(t.tree) // Call transformNode with only the backend tree node
      );
      setForest(transformedForest);
      setTargetNames(data.target_names || []);
    })
    .catch((err) => console.error("Failed to load forest:", err));
}, [params]);

  // Render individual trees when forest changes
  useEffect(() => {
    // Dynamically resize canvasRefs.current to match numberOfTrees
    canvasRefs.current = Array(params.numberOfTrees).fill(null).map((_, i) => canvasRefs.current[i] || null);

    if (forest.length > 0 && canvasRefs.current.length === forest.length) {
      const timer = setTimeout(() => {
        forest.forEach((_, i) => {
          const canvas = canvasRefs.current[i];
          if (canvas) {
            const ctx = canvas.getContext("2d");
            if (ctx) renderDecisionTree(ctx, canvas.width, canvas.height, i, forest, classColors);
          }
        });
      }, 100);

      return () => clearTimeout(timer);
    }
  }, [forest, params.numberOfTrees]); // Added params.numberOfTrees to dependency array to trigger re-render on change


  const handlePredict = async () => {
    const selectedPoint = dataset[selectedIndex];
    const recordToSend: number[] = [];
    const currentFeatureNames = featureNamesMap[params.dataset];

    currentFeatureNames.forEach(featureName => {
      if (typeof selectedPoint[featureName] === 'number') {
        recordToSend.push(selectedPoint[featureName] as number);
      } else {

        recordToSend.push(0); // or some default value
      }
    });

    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/predict_forest`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          dataset: params.dataset,
          params: { // Matches PredictionTreeParams in backend
            n_trees: params.numberOfTrees,
            max_depth: params.maxDepth,
            min_samples_split: params.minSamplesSplit,
            criterion: params.criterion,
            subsample_ratio: params.subsampleRatio,
            feature_subset: params.featureSubsetRatio,
          },
          record: recordToSend, // The data point for prediction as a list of floats
        }),
      });

      const data = await response.json();
      console.log("Prediction response:", data);

      if (response.ok) {
        setPrediction(data.majority_vote);
        setIndividualVotes(data.individual_votes);
      } else {
        console.error("Prediction failed:", data.detail || "Unknown error");
        setPrediction(null);
        setIndividualVotes([]);
      }
    } catch (error) {
      console.error("Error during prediction fetch:", error);
      setPrediction(null);
      setIndividualVotes([]);
    }
  };


return (
  <div className="min-h-screen bg-black-900 text-white p-8">
    <h1 className="text-4xl font-bold mb-8 text-center">Random Forest Visualization</h1>

    {/* Trees canvas grid */}
    <div className="grid grid-cols-1 gap-6">
      <div className="space-y-4">
        <h2 className="text-2xl font-semibold">Random Forest Trees</h2>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-3 gap-3">
          {Array.from({ length: params.numberOfTrees }).map((_, index) => (
            <div key={index} className="border border-gray-300 rounded-lg bg-black p-2">
              <h3 className="text-xl font-semibold text-center mb-2">Tree {index + 1}</h3>
              <canvas
                ref={(el) => {
                  if (el) {
                    // Update the ref directly when the element is available
                    canvasRefs.current[index] = el;
                    // Trigger render if forest data exists for this index
                    if (forest[index]) {
                      const ctx = el.getContext("2d");
                      if (ctx) renderDecisionTree(ctx, el.width, el.height, index, forest, classColors);
                    }
                  }
                }}
                width={400}
                height={600}
              />
            </div>
          ))}
        </div>
      </div>

      {/* Sample picker & prediction */}
      {dataset.length > 0 && (
        <div className="mt-6 space-y-4 bg-black-800 p-6 rounded-lg shadow-lg">
          <h2 className="text-2xl font-semibold">Majority Vote Prediction</h2>
          <label className="text-white">
            Pick a sample index (0 to {dataset.length - 1}):
            <input
              type="number"
              min={0}
              max={dataset.length - 1}
              value={selectedIndex}
              onChange={(e) => setSelectedIndex(Number(e.target.value))}
              className="ml-2 p-1 rounded bg-black-700 text-white border border-gray-600"
            />
          </label>
          <button
            onClick={handlePredict}
            className="ml-4 px-4 py-2 bg-[#ff3860] text-white rounded hover:bg-blue-600">
            Predict
          </button>
          {prediction !== null && (
            <div className="text-white mt-4">
              <p>
                <strong>True Label:</strong>{" "}
                {targetNames[parseInt(dataset[selectedIndex]?.label)] || dataset[selectedIndex]?.label || "Unknown"}
              </p>
              <p>
                <strong>Individual Votes:</strong>{" "}
                {individualVotes.map((v) => targetNames[v] || v).join(", ")}
              </p>
              <p>
                <strong>Majority Vote:</strong>{" "}
                {targetNames[prediction] || prediction}
              </p>
            </div>
          )}
        </div>
      )}

      {/* Controls section - The *only* control panel, as requested */}
      <section className="controls-section space-y-6 bg-black-800 p-6 rounded-lg shadow-lg">
        <h3 className="text-xl font-semibold text-white mb-4">Controls</h3>

        {/* Dataset Selector */}
        <div className="control-row flex items-center justify-between">
          <span className="control-label">Dataset</span>
          <select
            value={params.dataset}
            onChange={(e) => setParams({ ...params, dataset: e.target.value as DatasetType })}
            className="styled-select flex-grow mx-4"
          >
            <option value="iris">Iris</option>
            <option value="wine">Wine</option>
            <option value="digits">Digits</option>
          </select>
          <span className="control-value">{params.dataset}</span>
        </div>

        <div className="control-row flex items-center justify-between">
          <span className="control-label">Number of Trees</span>
          <input
            type="range"
            min="1"
            max="20"
            value={params.numberOfTrees}
            onChange={(e) =>
              setParams({ ...params, numberOfTrees: +e.target.value })
            }
            style={
              {
                "--value": `${((params.numberOfTrees - 1) / 19) * 100}%`,
              } as React.CSSProperties
            }
            className="slider flex-grow mx-4"
          />
          <span className="control-value">{params.numberOfTrees}</span>
        </div>

        <div className="control-row flex items-center justify-between">
          <span className="control-label">Subsample Ratio</span>
          <input
            type="range"
            min="0.1"
            max="1.0"
            step="0.05"
            value={params.subsampleRatio}
            onChange={(e) =>
              setParams({ ...params, subsampleRatio: +e.target.value })
            }
            style={
              {
                "--value": `${((params.subsampleRatio - 0.1) / 0.9) * 100}%`,
              } as React.CSSProperties
            }
            className="slider flex-grow mx-4"
          />
          <span className="control-value">
            {(params.subsampleRatio).toFixed(2)}
          </span>
        </div>

        <div className="control-row flex items-center justify-between">
          <span className="control-label">Feature Subset Ratio</span>
          <input
            type="range"
            min="0.1"
            max="1.0"
            step="0.05"
            value={params.featureSubsetRatio}
            onChange={(e) =>
              setParams({ ...params, featureSubsetRatio: +e.target.value })
            }
            style={
              {
                "--value": `${((params.featureSubsetRatio - 0.1) / 0.9) * 100}%`,
              } as React.CSSProperties
            }
            className="slider flex-grow mx-4"
          />
          <span className="control-value">
            {(params.featureSubsetRatio).toFixed(2)}
          </span>
        </div>

        <div className="control-row flex items-center justify-between">
          <span className="control-label">Max Depth</span>
          <input
            type="range"
            min="1"
            max="5"
            value={params.maxDepth}
            onChange={(e) =>
              setParams({ ...params, maxDepth: +e.target.value })
            }
            style={
              {
                "--value": `${((params.maxDepth - 1) / 4) * 100}%`,
              } as React.CSSProperties
            }
            className="slider flex-grow mx-4"
          />
          <span className="control-value">{params.maxDepth}</span>
        </div>

        <div className="control-row flex items-center justify-between">
          <span className="control-label">Min Samples Split</span>
          <input
            type="range"
            min="2"
            max="20"
            value={params.minSamplesSplit}
            onChange={(e) =>
              setParams({ ...params, minSamplesSplit: +e.target.value })
            }
            style={
              {
                "--value": `${((params.minSamplesSplit - 2) / 18) * 100}%`,
              } as React.CSSProperties
            }
            className="slider flex-grow mx-4"
          />
          <span className="control-value">{params.minSamplesSplit}</span>
        </div>

        <div className="control-row flex items-center justify-between">
          <span className="control-label">Criterion</span>
          <select
            value={params.criterion}
            onChange={(e) =>
              setParams({ ...params, criterion: e.target.value })
            }
            className="styled-select flex-grow mx-4"
          >
            <option value="gini">Gini</option>
            <option value="entropy">Entropy</option>

          </select>
          <span className="control-value">{params.criterion}</span>
        </div>

        <button
          className="reset-btn mt-4 flex items-center justify-center gap-2"
          onClick={() =>
            setParams({
              maxDepth: 5,
              minSamplesSplit: 2,
              criterion: "gini",
              numberOfTrees: 10,
              subsampleRatio: 0.8,
              featureSubsetRatio: 0.7,
              dataset: params.dataset
            })
          }
        >
          <FiRefreshCw className="inline-block h-5 w-5" />
          Reset
        </button>
      </section>

      {/* Explanation and info sections */}
      <div className="space-y-6 text-muted-foreground">
        <section>
          <h2 className="text-2xl font-semibold text-white">How It Works</h2>
          <p>
            This visualizer shows multiple decision trees, each trained on a random subset of the selected dataset.
            Each node shows the impurity score (Gini, Entropy, or Gain Ratio), and the predicted class is indicated by color.
            If a split occurs, the feature used and the threshold value are shown above the node.
          </p>
          <p className="mt-2">
            The trees differ because they're trained on different random subsets of the data and may use
            different random feature subsets at each node, similar to how Random Forests work.
          </p>
        </section>

        <section className="text-white">
            <h3 className="text-xl text-white font-semibold">Formulas</h3>
            <p className="text-white font-semibold">Gini Impurity:</p>
            <BlockMath math="Gini = 1 - \sum_{i=1}^{n} p_i^2" />
            <p className="text-white font-semibold">Information Gain:</p>
            <BlockMath math="Gain = Impurity(parent) - \sum_{k} \frac{n_k}{n} Impurity(k)" />
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
      </div>
    </div>

      {/* ——— Embedded CSS ——— */}
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
    </div>
  );
}