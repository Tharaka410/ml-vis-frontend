"use client"

import { useEffect, useMemo, useState } from "react"
import VisualizationCanvas from "@/components/visualization-canvas"

const TOTAL_POINTS = 200
const NOISE_RATIO = 0.1

const generateClusterData = (numClusters: number, clusterPointCount: number) => {
  const clusterCenters = []
  const clusterPoints = []

  for (let i = 0; i < numClusters; i++) {
    const centerX = Math.random() * 1.6 - 0.8
    const centerY = Math.random() * 1.6 - 0.8
    clusterCenters.push({ x: centerX, y: centerY })
  }

  const pointsPerCluster = Math.floor(clusterPointCount / numClusters)

  for (let i = 0; i < numClusters; i++) {
    const center = clusterCenters[i]
    for (let j = 0; j < pointsPerCluster; j++) {
      const angle = Math.random() * Math.PI * 2
      const distance = Math.random() * 0.3

      const x = Math.max(-1, Math.min(1, center.x + Math.cos(angle) * distance))
      const y = Math.max(-1, Math.min(1, center.y + Math.sin(angle) * distance))

      clusterPoints.push({ x, y })
    }
  }

  return clusterPoints
}

export default function DBSCANPage() {
  const [params, setParams] = useState({
    epsilon: 0.2,
    minPoints: 5,
    numClusters: 3,
  })

  const [labeledPoints, setLabeledPoints] = useState<
    { x: number; y: number; cluster: number }[]
  >([])

  // Generate cluster points only when numClusters changes
  const clusterPoints = useMemo(() => {
    const clusterPointCount = Math.floor(TOTAL_POINTS * (1 - NOISE_RATIO))
    return generateClusterData(params.numClusters, clusterPointCount)
  }, [params.numClusters])

  useEffect(() => {
    const fetchLabeledPoints = async () => {
      const noiseCount = Math.floor(TOTAL_POINTS * NOISE_RATIO)
      const noisePoints = Array.from({ length: noiseCount }, () => ({
        x: Math.random() * 2 - 1,
        y: Math.random() * 2 - 1,
      }))

      const allPoints = [...clusterPoints, ...noisePoints]

      try {
        const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/dbscan`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            points: allPoints,
            epsilon: params.epsilon,
            minPoints: params.minPoints, 
          }),
        })

        if (!res.ok) {
          console.error("API returned an error:", res.status)
          return
        }

        const data = await res.json()

        if (!data.labels || data.labels.length !== allPoints.length) {
          console.error("Invalid response from server:", data)
          return
        }

        const labeled = allPoints.map((pt, i) => ({
          ...pt,
          cluster: data.labels[i],
        }))
        setLabeledPoints(labeled)
      } catch (err) {
        console.error("Failed to fetch clusters:", err)
      }
    }

    fetchLabeledPoints()
  }, [params.epsilon, params.minPoints, clusterPoints])

function computeConvexHull(points: { x: number; y: number }[]) {
  if (points.length < 3) return points

  // Sort points by x, then y
  const sorted = points
    .map(p => ({ x: p.x, y: p.y }))
    .sort((a, b) => (a.x === b.x ? a.y - b.y : a.x - b.x))

  const cross = (o: any, a: any, b: any) =>
    (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)

  const lower = []
  for (const p of sorted) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) {
      lower.pop()
    }
    lower.push(p)
  }

  const upper = []
  for (let i = sorted.length - 1; i >= 0; i--) {
    const p = sorted[i]
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) {
      upper.pop()
    }
    upper.push(p)
  }

  upper.pop()
  lower.pop()

  return lower.concat(upper)
}

const renderDBSCAN = (
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number
) => {
  const margin = 40
  const plotWidth = width - 2 * margin
  const plotHeight = height - 2 * margin

  ctx.clearRect(0, 0, width, height)

  // Axes
  ctx.strokeStyle = "#666"
  ctx.lineWidth = 1
  ctx.beginPath()
  ctx.moveTo(margin, height - margin)
  ctx.lineTo(width - margin, height - margin)
  ctx.stroke()
  ctx.beginPath()
  ctx.moveTo(margin, margin)
  ctx.lineTo(margin, height - margin)
  ctx.stroke()

  ctx.fillStyle = "#999"
  ctx.font = "12px sans-serif"
  ctx.textAlign = "center"
  ctx.fillText("-1", margin, height - margin + 20)
  ctx.fillText("0", margin + plotWidth / 2, height - margin + 20)
  ctx.fillText("1", width - margin, height - margin + 20)
  ctx.textAlign = "right"
  ctx.fillText("-1", margin - 10, height - margin)
  ctx.fillText("0", margin - 10, height - margin - plotHeight / 2)
  ctx.fillText("1", margin - 10, margin)

  const clusterColors = [
    "#4ade80",
    "#60a5fa",
    "#f472b6",
    "#fb923c",
    "#a78bfa",
    "#facc15",
    "#22d3ee",
  ]

  const clusters = new Map<number, { x: number; y: number }[]>()

  for (const pt of labeledPoints) {
    if (pt.cluster !== -1) {
      if (!clusters.has(pt.cluster)) clusters.set(pt.cluster, [])
      clusters.get(pt.cluster)!.push(pt)
    }
  }
  ctx.lineWidth = 2
  clusters.forEach((points, clusterId) => {
    const hull = computeConvexHull(points)
    if (hull.length < 3) return

    ctx.strokeStyle = clusterColors[clusterId % clusterColors.length]
    ctx.fillStyle = clusterColors[clusterId % clusterColors.length] + "33"

    ctx.beginPath()
    hull.forEach((p, i) => {
      const canvasX = margin + ((p.x + 1) / 2) * plotWidth
      const canvasY = margin + plotHeight - ((p.y + 1) / 2) * plotHeight
      if (i === 0) ctx.moveTo(canvasX, canvasY)
      else ctx.lineTo(canvasX, canvasY)
    })
    ctx.closePath()
    ctx.fill()
    ctx.stroke()
  })

  for (const point of labeledPoints) {
    const canvasX = margin + ((point.x + 1) / 2) * plotWidth
    const canvasY = margin + plotHeight - ((point.y + 1) / 2) * plotHeight

    let color = "#999"
    if (typeof point.cluster === "number") {
      color =
        point.cluster === -1
          ? "#f87171"
          : clusterColors[point.cluster % clusterColors.length]
    }

    ctx.fillStyle = color
    ctx.beginPath()
    ctx.arc(canvasX, canvasY, 4, 0, Math.PI * 2)
    ctx.fill()
  }

  ctx.fillStyle = "#fff"
  ctx.font = "14px sans-serif"
  ctx.textAlign = "left"
  ctx.fillText(`Epsilon: ${params.epsilon}`, margin + 10, margin + 20)
  ctx.fillText(`Min Points: ${params.minPoints}`, margin + 10, margin + 40)
  ctx.fillText(`Clusters Found: ${clusters.size}`, margin + 10, margin + 60)
  ctx.fillText(
    `Noise Points: ${labeledPoints.filter((p) => p.cluster === -1).length}`,
    margin + 10,
    margin + 80
  )
}



  const paramControls = [
    {
      name: "epsilon",
      label: "Epsilon (Neighborhood Radius)",
      min: 0.05,
      max: 0.5,
      step: 0.05,
      defaultValue: 0.2,
    },
    {
      name: "minPoints",
      label: "Min Points",
      min: 2,
      max: 10,
      step: 1,
      defaultValue: 5,
    },
    {
      name: "numClusters",
      label: "Number of Clusters",
      min: 1,
      max: 5,
      step: 1,
      defaultValue: 3,
    },
  ]


  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-4xl font-bold mb-4">DBSCAN</h1>
        <p className="text-lg text-muted-foreground mb-6">
          Density-Based Spatial Clustering of Applications with Noise - a density-based clustering algorithm that groups
          together points that are closely packed together.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2">
          <VisualizationCanvas
            renderFunction={renderDBSCAN}
            params={params}
            setParams={setParams}
            paramControls={paramControls}
            height={500}
          />
        </div>

        <div className="space-y-6">
          <div>
            <h2 className="text-2xl font-semibold mb-2">How It Works</h2>
            <p className="text-muted-foreground">
              DBSCAN groups together points that are close to each other based on a distance measure (usually Euclidean
              distance) and a minimum number of points. It also marks as outliers (noise) points that are in low-density
              regions.
            </p>
          </div>

          <div>
            <h3 className="text-xl font-semibold mb-2">Key Concepts</h3>
            <ul className="list-disc list-inside space-y-2 text-muted-foreground">
              <li>
                <strong>Core Point:</strong> A point with at least minPoints points within distance epsilon.
              </li>
              <li>
                <strong>Border Point:</strong> A point within distance epsilon of a core point but with fewer than
                minPoints neighbors.
              </li>
              <li>
                <strong>Noise Point:</strong> A point that is neither a core point nor a border point.
              </li>
            </ul>
          </div>

          <div>
            <h3 className="text-xl font-semibold mb-2">Algorithm Steps</h3>
            <ol className="list-decimal list-inside space-y-2 text-muted-foreground">
              <li>Find all core points and their neighborhoods.</li>
              <li>Connect core points that are within epsilon distance of each other to form clusters.</li>
              <li>Assign each border point to the cluster of its closest core point.</li>
              <li>Label any remaining points as noise.</li>
            </ol>
          </div>

          <div>
            <h3 className="text-xl font-semibold mb-2">Interactive Controls</h3>
            <ul className="list-disc list-inside space-y-2 text-muted-foreground">
              <li>
                <strong>Epsilon:</strong> The maximum distance between two points for them to be considered neighbors.
                Larger values create larger clusters.
              </li>
              <li>
                <strong>Min Points:</strong> The minimum number of points required to form a dense region. Higher values
                require denser clusters.
              </li>
              <li>
                <strong>Number of Clusters:</strong> Controls how many clusters are generated in the synthetic data.
              </li>
            </ul>
          </div>

          <div>
            <h3 className="text-xl font-semibold mb-2">Applications</h3>
            <p className="text-muted-foreground">DBSCAN is widely used in:</p>
            <ul className="list-disc list-inside space-y-2 text-muted-foreground">
              <li>Anomaly detection</li>
              <li>Spatial data analysis</li>
              <li>Image segmentation</li>
              <li>Market research (customer segmentation)</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}
