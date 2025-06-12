"use client"

import React, { useEffect, useRef, useState } from "react"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import "katex/dist/katex.min.css"
import { InlineMath } from "react-katex"

const colors = ["red", "green", "blue", "orange", "purple", "cyan"]

export default function KMeansVisualizer() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animRef = useRef<number | null>(null)

  const [config, setConfig] = useState({
    points: 100,
    clusters: 1,
    iterations: 10,
  })
  const [initMode, setInitMode] = useState<"random" | "kmeans++">("random")
  const [version, setVersion] = useState(0)
  const [data, setData] = useState<{ x: number; y: number; cluster?: number }[]>([])
  const [centroids, setCentroids] = useState<{ x: number; y: number }[]>([])

  const generateRandomPoints = () => {
    return Array.from({ length: config.points }, () => ({
      x: Math.random() * 780 + 10,
      y: Math.random() * 580 + 10,
    }))
  }

  const euclideanDistance = (a: { x: number; y: number }, b: { x: number; y: number }) => {
    return (a.x - b.x) ** 2 + (a.y - b.y) ** 2
  }

  const initializeCentroids = (points: typeof data) => {
    if (initMode === "random") {
      return Array.from({ length: config.clusters }, () => ({
        x: Math.random() * 780 + 10,
        y: Math.random() * 580 + 10,
      }))
    } else {
      const centroids = [points[Math.floor(Math.random() * points.length)]]
      while (centroids.length < config.clusters) {
        const distances = points.map(p => Math.min(...centroids.map(c => euclideanDistance(p, c))))
        const sum = distances.reduce((a, b) => a + b, 0)
        const r = Math.random() * sum
        let acc = 0
        for (let i = 0; i < distances.length; i++) {
          acc += distances[i]
          if (acc >= r) {
            centroids.push(points[i])
            break
          }
        }
      }
      return centroids
    }
  }

  useEffect(() => {
    const newData = generateRandomPoints()
    const newCentroids = initializeCentroids(newData)
    setData(newData)
    setCentroids(newCentroids)
  }, [config.points, config.clusters, version])

  useEffect(() => {
    if (!data.length || !centroids.length) return
    let points = [...data]
    let centers = [...centroids]
    let iter = 0

    const step = () => {
      points = points.map(p => ({ ...p, cluster: undefined }))
      points = points.map((p) => {
        let minDist = Infinity
        let cluster = 0
        centers.forEach((c, i) => {
          const dist = euclideanDistance(p, c)
          if (dist < minDist) {
            minDist = dist
            cluster = i
          }
        })
        return { ...p, cluster }
      })

      const sums = Array.from({ length: config.clusters }, () => ({ x: 0, y: 0, count: 0 }))
      points.forEach((p) => {
        const cluster = p.cluster ?? 0
        sums[cluster].x += p.x
        sums[cluster].y += p.y
        sums[cluster].count++
      })

      centers = sums.map((s) => ({
        x: s.count ? s.x / s.count : Math.random() * 780 + 10,
        y: s.count ? s.y / s.count : Math.random() * 580 + 10,
      }))

      draw(points, centers)

      if (++iter < config.iterations) {
        animRef.current = requestAnimationFrame(step)
      } else {
        setData(points)
        setCentroids(centers)
      }
    }

    step()
    return () => cancelAnimationFrame(animRef.current!)
  }, [version, data, centroids])

  const draw = (points: any[], centers: any[]) => {
    const canvas = canvasRef.current
    const ctx = canvas?.getContext("2d")
    if (!canvas || !ctx) return
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Draw Voronoi-like decision boundaries
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)
    const pixels = imageData.data

    for (let y = 0; y < canvas.height; y++) {
      for (let x = 0; x < canvas.width; x++) {
        let minDist = Infinity
        let cluster = 0
        centers.forEach((c, i) => {
          const dist = euclideanDistance({ x, y }, c)
          if (dist < minDist) {
            minDist = dist
            cluster = i
          }
        })
        const idx = (y * canvas.width + x) * 4
        const color = colors[cluster % colors.length]
        const rgb = getColorRGB(color)
        pixels[idx] = rgb[0]
        pixels[idx + 1] = rgb[1]
        pixels[idx + 2] = rgb[2]
        pixels[idx + 3] = 20 // very transparent
      }
    }
    ctx.putImageData(imageData, 0, 0)

    for (const p of points) {
      ctx.beginPath()
      ctx.arc(p.x, p.y, 8, 0, 2 * Math.PI)
      ctx.fillStyle = colors[p.cluster ?? 0]
      ctx.fill()
    }

    for (const [i, c] of centers.entries()) {
      ctx.beginPath()
      ctx.arc(c.x, c.y, 8, 0, 2 * Math.PI)
      ctx.strokeStyle = colors[i % colors.length]
      ctx.lineWidth = 3
      ctx.stroke()

      ctx.fillStyle = "#ffffff"
      ctx.font = "14px sans-serif"
      ctx.fillText(`Cluster ${i + 1}`, c.x + 10, c.y - 10)
    }
  }

  const getColorRGB = (name: string): [number, number, number] => {
    const map: Record<string, [number, number, number]> = {
      red: [255, 0, 0],
      green: [0, 128, 0],
      blue: [0, 0, 255],
      orange: [255, 165, 0],
      purple: [128, 0, 128],
      cyan: [0, 255, 255],
    }
    return map[name] || [255, 255, 255]
  }

  const handleSliderChange = (key: keyof typeof config, value: number) => {
    setConfig((c) => ({ ...c, [key]: value }))
    setVersion((v) => v + 1)
  }

  return (
    <div className="flex flex-col gap-6">
      <div className="text-2xl font-bold">K-Means Visualizer</div>
      <div className="text-muted-foreground text-[16px]">
        Watch the K-Means algorithm cluster random data points into groups.
      </div>

      <div className="flex flex-col lg:flex-row gap-6">
        <div className="flex flex-col gap-6">
          <canvas
            ref={canvasRef}
            width={800}
            height={600}
            className="rounded border shadow"
          />

          <div className="flex flex-col gap-4">
            {[{ label: "Number of points in the dataset", key: "points", min: 10, max: 300 }, { label: "Number of clusters", key: "clusters", min: 1, max: 6 }].map(({ label, key, min, max }) => (
              <div key={key} className="flex flex-col">
                <div className="flex justify-between items-center">
                  <Label className="text-[1.15rem] mb-[5px] mt-[10px] font-semibold">{label}</Label>
                  <span className="text-sm text-muted-foreground">
                    {config[key as keyof typeof config]}
                  </span>
                </div>
                <Slider
                  min={min}
                  max={max}
                  value={[config[key as keyof typeof config] as number]}
                  onValueChange={([val]) => handleSliderChange(key as keyof typeof config, val)}
                />
              </div>
            ))}

            <div className="flex flex-col">
              <Label className="text-[1.15rem] mb-[5px] mt-[10px] font-semibold"> Centroid Initialization Method</Label>
              <select
                className="bg-background border rounded px-3 py-2 text-white"
                value={initMode}
                onChange={(e) => {
                  setInitMode(e.target.value as "random" | "kmeans++")
                  setVersion((v) => v + 1)
                }}
              >
                <option value="random">Random</option>
                <option value="kmeans++">K-Means++</option>
              </select>
            </div>
          </div>
        </div>

        <div className="text-[16px] text-muted-foreground space-y-6 max-w-md">
          <div>
            <div className="text-lg font-semibold text-white mb-1">K-Means Overview</div>
            <ul className="list-disc list-inside space-y-1">
              <li>Initialize centroids using selected method</li>
              <li>Assign points to nearest centroid</li>
              <li>Recalculate centroids</li>
              <li>Repeat until convergence or max iterations</li>
            </ul>
          </div>

          <div>
            <div className="text-lg font-semibold text-white mb-1">Key Formulas</div>
            <div className="mb-2">
              <span className="text-white font-semibold">Distance:</span><br />
              <p className="text-white font-semibold"><InlineMath math="d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}" /></p>
            </div>
            <div>
              <span className="text-white font-semibold">Centroid:</span><br />
              <p className="text-white font-semibold"><InlineMath math="c_i = \frac{1}{n} \sum_{j=1}^{n} x_j" /></p>
            </div>
          </div>

          <div>
            <div className="text-lg font-semibold text-white mb-1">Applications</div>
            <ul className="list-disc list-inside space-y-1">
              <li>Customer segmentation</li>
              <li>Image compression</li>
              <li>Document clustering</li>
              <li>Anomaly detection</li>
              <li>Product grouping</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}

