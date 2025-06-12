"use client"

import { useEffect, useRef, useState, useCallback } from "react"

function useSVMData(params: any) {
    const [points, setPoints] = useState<any[]>([])
    const initialPoints = useRef<any[]>([])
    const prevNoise = useRef<number>(params.noise)

    const generateData = (noise: number, kernel: string) => {
        const numPoints = 25
        const data = []

        for (let i = 0; i < numPoints; i++) {
            const x = Math.random() * 4 - 2 // [-2, 2]
            const y = Math.random() * 4 - 2
            let label = 0

            if (kernel === "linear") {
                label = y > 0.5 * x ? (Math.random() > noise ? 1 : 0) : (Math.random() > noise ? 0 : 1)
            } else {
                const dist = Math.sqrt(x * x + y * y)
                label = dist < 1.2 ? (Math.random() > noise ? 1 : 0) : (Math.random() > noise ? 0 : 1)
            }

            data.push({ x, y, label })
        }

        return data
    }

    useEffect(() => {
        // Initial data generation on mount
        const initial = generateData(params.noise, params.kernel)
        initialPoints.current = initial.map(p => ({ ...p, kernelType: params.kernel })); // Store kernel type with points
        setPoints(initial)
    }, []) // Empty dependency array means this runs once on mount

    useEffect(() => {
        // Regenerate points if noise or kernel changes
        if (prevNoise.current !== params.noise || params.kernel !== (initialPoints.current[0]?.kernelType || null)) {
            prevNoise.current = params.noise;
            const newPoints = generateData(params.noise, params.kernel);
            initialPoints.current = newPoints.map(p => ({ ...p, kernelType: params.kernel })); // Store kernel type with points
            setPoints(newPoints);
        } else {
            // If only C or Gamma changes, use the existing points
            setPoints(initialPoints.current);
        }
    }, [params.noise, params.kernel]); // Depend on noise and kernel

    return { points, setPoints }
}

function renderSVM(ctx: CanvasRenderingContext2D, width: number, height: number, params: any, points: any[]) {
    const { c, kernel, gamma } = params
    const margin = 40
    const plotWidth = width - 2 * margin
    const plotHeight = height - 2 * margin

    const kernelFunction = (x1: number, y1: number, x2: number, y2: number) => {
        if (kernel === "linear") return x1 * x2 + y1 * y2
        const dist2 = Math.pow(x1 - x2, 2) + Math.pow(y1 - y2, 2)
        return Math.exp(-gamma * dist2)
    }

    const predictRawBase = (x: number, y: number) =>
        points.reduce((sum, point) => {
            const k = kernelFunction(x, y, point.x, point.y)
            return sum + (point.label === 1 ? 1 : -1) * k
        }, 0)

    const marginCutoff = 1 / c
    const alphas = points.map(p => {
        const raw = predictRawBase(p.x, p.y)
        return Math.abs(raw) < marginCutoff ? (p.label === 1 ? c : -c) : 0
    })

    const predictRaw = (x: number, y: number) =>
        points.reduce((sum, point, i) => sum + alphas[i] * kernelFunction(x, y, point.x, point.y), 0)

    const predict = (x: number, y: number) => (predictRaw(x, y) > 0 ? 1 : 0)

    // Clear and draw background
    ctx.clearRect(0, 0, width, height)
    const resolution = 200
    const stepX = plotWidth / resolution
    const stepY = plotHeight / resolution

    for (let px = 0; px < resolution; px++) {
        for (let py = 0; py < resolution; py++) {
            const x = (px / resolution) * 10 - 5
            const y = (py / resolution) * 10 - 5
            const raw = predictRaw(x, y)

            let color = "rgba(200,200,200,0.1)"
            if (Math.abs(raw) < 1.0) color = "rgba(255,255,0,0.2)"
            else color = raw > 0 ? "rgba(74, 222, 128, 0.2)" : "rgba(248, 113, 113, 0.2)"

            ctx.fillStyle = color
            ctx.fillRect(margin + px * stepX, margin + py * stepY, stepX, stepY)
        }
    }

    // Draw margins for linear kernel
    if (kernel === "linear") {
        const drawMargin = (level: number, color: string) => {
            ctx.strokeStyle = color
            ctx.beginPath()
            for (let px = 0; px < resolution; px++) {
                const x = (px / resolution) * 10 - 5
                let found = false
                for (let py = 0; py < resolution; py++) {
                    const y = (py / resolution) * 10 - 5
                    const raw = predictRaw(x, y)
                    if (Math.abs(raw - level) < 0.05) {
                        const canvasX = margin + px * stepX
                        const canvasY = margin + py * stepY
                        if (!found) {
                            ctx.moveTo(canvasX, canvasY)
                            found = true
                        } else {
                            ctx.lineTo(canvasX, canvasY)
                            break
                        }
                    }
                }
            }
            ctx.stroke()
        }

        drawMargin(0, "#000")
        drawMargin(1, "#aaa")
        drawMargin(-1, "#aaa")
    }

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

    // Labels
    ctx.fillStyle = "#999"
    ctx.font = "12px sans-serif"
    ctx.textAlign = "center"
    ctx.fillText("-5", margin, height - margin + 20)
    ctx.fillText("0", margin + plotWidth / 2, height - margin + 20)
    ctx.fillText("5", width - margin, height - margin + 20)
    ctx.textAlign = "right"
    ctx.fillText("-5", margin - 10, height - margin)
    ctx.fillText("0", margin - 10, height - margin - plotHeight / 2)
    ctx.fillText("5", margin - 10, margin)


    // Points
    for (const [i, point] of points.entries()) {
        const canvasX = margin + ((point.x + 5) / 10) * plotWidth
        const canvasY = margin + ((point.y + 5) / 10) * plotHeight

        // Highlight support vectors with a border
        const isSupportVector = Math.abs(predictRaw(point.x, point.y)) <= 1.05
        ctx.fillStyle = point.label === 1 ? "#4ade80" : "#f87171"
        ctx.beginPath()
        ctx.arc(canvasX, canvasY, 4, 0, Math.PI * 2)
        ctx.fill()

        if (isSupportVector) {
            ctx.strokeStyle = "#fff"
            ctx.lineWidth = 2
            ctx.beginPath()
            ctx.arc(canvasX, canvasY, 6, 0, Math.PI * 2) // Larger circle for support vectors
            ctx.stroke()
        }
    }

    // Info
    ctx.fillStyle = "#fff"
    ctx.font = "14px sans-serif"
    ctx.textAlign = "left"
    ctx.fillText(`C: ${c.toFixed(1)}`, margin + 10, margin + 20)
    ctx.fillText(`Kernel: ${kernel === "linear" ? "Linear" : "RBF"}`, margin + 10, margin + 40)
    if (kernel === "rbf") ctx.fillText(`Gamma: ${gamma.toFixed(1)}`, margin + 10, margin + 60)

    const errors = points.filter(p => predict(p.x, p.y) !== p.label).length
    ctx.fillText(`Errors: ${errors}`, margin + 10, margin + (kernel === "linear" ? 60 : 80))
}


export default function SVMPage() {
    const [params, setParams] = useState({
        c: 1.0,
        kernel: "rbf",
        gamma: 0.5,
        noise: 0.1,
    })

    const { points } = useSVMData(params)
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const CANVAS_HEIGHT = 500; // Define a constant for canvas height

    const paramControls = [
        { name: "c", label: "C (Regularization)", min: 0.1, max: 2, step: 0.05, defaultValue: 1.0 },
        {
            name: "kernel",
            label: "Kernel Type",
            type: "dropdown",
            options: [
                { value: "rbf", label: "RBF" },
                { value: "linear", label: "Linear" }
            ],
            defaultValue: "rbf"
        },
        { name: "gamma", label: "Gamma (RBF Kernel)", min: 0.1, max: 10, step: 0.1, defaultValue: 0.5 },
        { name: "noise", label: "Data Noise", min: 0, max: 0.3, step: 0.05, defaultValue: 0.1 },
    ];

    const handleParamChange = (name: string, value: string | number) => {
        setParams(prevParams => ({
            ...prevParams,
            [name]: value,
        }));
    };

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();
        canvas.width = rect.width * dpr;
        canvas.height = CANVAS_HEIGHT * dpr;
        ctx.scale(dpr, dpr);

        renderSVM(ctx, rect.width, CANVAS_HEIGHT, params, points);

        // Update range input progress CSS variable
        paramControls.forEach(control => {
            if (!("type" in control) || control.type !== "dropdown") { // Only for range inputs
                const inputElement = document.getElementById(control.name) as HTMLInputElement;
                if (inputElement && inputElement.type === 'range') {
                    const value = (inputElement.valueAsNumber - parseFloat(inputElement.min)) / (parseFloat(inputElement.max) - parseFloat(inputElement.min));
                    inputElement.style.setProperty('--range-progress', `${value * 100}%`);
                }
            }
        });

    }, [params, points]); // Redraw when params or points change

    const sliderStyles = `
        input[type="range"] {
            background: #2d2d2d;
            --range-progress: 0%; /* Default */
            appearance: none;
        }

        /* Webkit (Chrome, Safari, Edge, newer Brave/Opera) */
        input[type="range"].range-slider-pink-track::-webkit-slider-runnable-track {
            background: linear-gradient(to right, #ff3860 var(--range-progress, 0%), #2d2d2d var(--range-progress, 0%));
            border-radius: 9999px;
            height: 8px;
        }

        input[type="range"].range-slider-pink-thumb::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: black;
            border: 2px solid #ff3860;
            cursor: pointer;
            margin-top: -6px;
            box-shadow: 0 0 0 0 rgba(0,0,0,0);
        }

        /* Firefox */
        input[type="range"].range-slider-pink-track::-moz-range-track {
            background: linear-gradient(to right, #ff3860 var(--range-progress, 0%), #2d2d2d var(--range-progress, 0%));
            border-radius: 9999px;
            height: 16px;
        }

        input[type="range"].range-slider-pink-thumb::-moz-range-thumb {
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: black;
            border: 2px solid #ff3860;
            cursor: pointer;
        }

        /* IE/Edge (older versions) - less common, but good for completeness */
        input[type="range"].range-slider-pink-track::-ms-track {
            background: transparent;
            border-color: transparent;
            color: transparent;
            height: 4px;
        }

        input[type="range"].range-slider-pink-track::-ms-fill-lower {
            background: #ff3860;
            border-radius: 9999px;
        }

        input[type="range"].range-slider-pink-track::-ms-fill-upper {
            background: #2d2d2d;
            border-radius: 9999px;
        }

        input[type="range"].range-slider-pink-thumb::-ms-thumb {
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: black;
            border: 2px solid #ff3860;
            cursor: pointer;
            margin-top: 0;
        }
    `;


    return (
        <div className="space-y-8">
            {/* THIS IS THE EMBEDDED STYLE TAG */}
            <style dangerouslySetInnerHTML={{ __html: sliderStyles }} />
            {/* END EMBEDDED STYLE TAG */}

            <header>
                <h1 className="text-4xl font-bold mb-4">Support Vector Machine</h1>
                <p className="text-lg text-muted-foreground mb-6">
                    A supervised learning model that finds the optimal hyperplane separating classes.
                </p>
            </header>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                {/* Main content area: Canvas + Controls */}
                <div className="lg:col-span-2 flex flex-col gap-8">
                    {/* Canvas */}
                    <div className="rounded-lg border bg-card text-card-foreground shadow-sm p-6">
                        <canvas ref={canvasRef} className="w-full" style={{ height: `${CANVAS_HEIGHT}px` }} />
                    </div>

                    {/* Interactive Controls - Now directly below the canvas */}
                        <div className="grid grid-cols-1 gap-4">
                            {paramControls.map((control) => (
                                <div key={control.name} className="flex flex-col">
                                    <label htmlFor={control.name} className="text-medium font-medium mb-1">
                                        {control.label}
                                    </label>
                                    {"type" in control && control.type === "dropdown" ? (
                                        <select
                                            id={control.name}
                                            className="block w-full p-2 border border-input bg-background rounded-md text-sm"
                                            value={params[control.name as keyof typeof params]}
                                            onChange={(e) => handleParamChange(control.name, e.target.value)}
                                        >
                                            {control.options.map(option => (
                                                <option key={option.value} value={option.value}>
                                                    {option.label}
                                                </option>
                                            ))}
                                        </select>
                                    ) : (
                                        <>
                                            <input
                                                id={control.name}
                                                type="range"
                                                min={(control as any).min}
                                                max={(control as any).max}
                                                step={(control as any).step}
                                                value={params[control.name as keyof typeof params]}
                                                onChange={(e) => handleParamChange(control.name, parseFloat(e.target.value))}
                                                // Classes still apply here to pick up the embedded styles
                                                className="w-full h-2 rounded-lg appearance-none cursor-pointer
                                                           range-slider-pink-track range-slider-pink-thumb"
                                            />
                                            <span className="text-right text-xs text-muted-foreground mt-1">
                                                {typeof params[control.name as keyof typeof params] === 'number'
                                                    ? (params[control.name as keyof typeof params] as number).toFixed(2)
                                                    : params[control.name as keyof typeof params]}
                                            </span>
                                        </>
                                    )}
                                </div>
                            ))}
                        </div>
                        <br />
                        
                        <br />
                </div>

                {/* Sidebar content */}
                <aside className="space-y-6">
                    <Section title="How It Works">
                        Support Vector Machines find the hyperplane maximizing the margin between classes, determined by support
                        vectors.
                    </Section>

                    <Section title="Kernel Trick">
                        <ul className="list-disc list-inside space-y-2">
                            <li>Linear: K(x,y) = x·y</li>
                            <li>RBF: K(x,y) = exp(-γ||x−y||²)</li>
                        </ul>
                    </Section>
                    
                    <Section title="Interactive Controls">
                      <ul className="list-disc list-inside space-y-2">
                            <li>C: Regularization trade-off</li>
                            <li>Kernel: Linear or RBF</li>
                            <li>Gamma: Complexity of RBF boundary</li>
                            <li>Noise: Adds classification challenge</li>
                            Note: The points that have a white border are the support vectors.
                        </ul>
                    </Section>
                    <Section title="Applications">
                        <ul className="list-disc list-inside space-y-2">
                            <li>Text/Image classification</li>
                            <li>Handwriting recognition</li>
                            <li>Bioinformatics</li>
                            <li>Face detection</li>
                        </ul>
                    </Section>
                </aside>
            </div>
        </div>
    )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
    return (
        <div>
            <h3 className="text-xl font-semibold mb-2">{title}</h3>
            <div className="text-muted-foreground">{children}</div>
        </div>
    )
}