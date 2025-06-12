// ml-frontend/components/algorithms/som-3d-scene.tsx
import React, { useRef, Suspense } from 'react';
import { Canvas, useFrame, extend } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Html, Line } from '@react-three/drei';
import * as THREE from 'three';

// Extend LineGeometry and LineMaterial for dashed lines
// This might not be strictly necessary if using Line from drei, but it's good practice for custom lines
// extend({ Line_: THREE.LineSegments, LineBasicMaterial_: THREE.LineBasicMaterial }); // This is for Three.js LineSegments, not Drei's Line

interface SOMNode {
  x: number;
  y: number;
  z: number;
  i: number; // Grid row
  j: number; // Grid col
}

interface SOM3DSceneProps {
  dataPoints: Array<{ x: number; y: number; z: number }>;
  somGrid: SOMNode[][]; // 2D array of nodes
  bmuNode: SOMNode | null; // The SOM node closest to the selected input
  bmuInputPoint: { x: number; y: number; z: number } | null; // The actual input data point that triggered BMU
  gridSize: number;
  currentIteration: number;
  totalIterations: number;
}

// Helper to normalize coordinates for the scene (e.g., from [-1, 1] to [-0.5, 0.5])
const normalizeCoord = (val: number) => (val * 0.5);

// Component to render a sphere at a given position and color
const DataPoint = ({ position, color, size = 0.02 }: { position: [number, number, number]; color: string; size?: number }) => {
  return (
    <mesh position={position}>
      <sphereGeometry args={[size, 16, 16]} />
      <meshBasicMaterial color={color} transparent opacity={0.6} />
    </mesh>
  );
};

// Component to render a node (sphere) for the SOM grid
const SOMNodeMesh = ({ position, isBMU }: { position: [number, number, number]; isBMU: boolean }) => {
  const meshRef = useRef<THREE.Mesh>(null);

  useFrame(({ clock }) => {
    if (isBMU && meshRef.current) {
      // Simple pulsing effect for BMU
      meshRef.current.scale.setScalar(1 + Math.sin(clock.getElapsedTime() * 5) * 0.2);
    } else if (meshRef.current) {
      meshRef.current.scale.setScalar(1); // Reset scale if not BMU
    }
  });

  return (
    <mesh ref={meshRef} position={position}>
      <sphereGeometry args={[0.025, 16, 16]} />
      <meshBasicMaterial color={isBMU ? '#22c55e' : '#ff3366'} /> {/* Green for BMU, Pink for others */}
    </mesh>
  );
};

// Component to render lines connecting SOM nodes
const SOMLines = ({ somGrid, gridSize }: { somGrid: SOMNode[][]; gridSize: number }) => {
  const points = [];
  const color = new THREE.Color('rgba(255,255,255,0.4)');

  for (let i = 0; i < gridSize; i++) {
    for (let j = 0; j < gridSize; j++) {
      const node = somGrid[i][j];
      const normalizedNodePos = new THREE.Vector3(normalizeCoord(node.x), normalizeCoord(node.y), normalizeCoord(node.z));

      // Connect to right neighbor
      if (j < gridSize - 1) {
        const rightNode = somGrid[i][j + 1];
        const normalizedRightNodePos = new THREE.Vector3(normalizeCoord(rightNode.x), normalizeCoord(rightNode.y), normalizeCoord(rightNode.z));
        points.push(normalizedNodePos, normalizedRightNodePos);
      }

      // Connect to bottom neighbor
      if (i < gridSize - 1) {
        const downNode = somGrid[i + 1][j];
        const normalizedDownNodePos = new THREE.Vector3(normalizeCoord(downNode.x), normalizeCoord(downNode.y), normalizeCoord(downNode.z));
        points.push(normalizedNodePos, normalizedDownNodePos);
      }
    }
  }

  return (
    <Line
      points={points}
      color={color}
      lineWidth={1}
      transparent
      opacity={0.7}
      dashed={false} // Use solid lines for clarity
    />
  );
};

// Component for XYZ Axes
const Axes = () => {
  const axisLength = 0.6; // Slightly larger than normalized data range of 0.5

  return (
    <group>
      {/* X Axis (Red) */}
      <Line
        points={[
          [0, 0, 0],
          [axisLength, 0, 0],
        ]}
        color="red"
        lineWidth={2}
      />
      <Html position={[axisLength + 0.05, 0, 0]} transform>
        <div style={{ color: 'red', fontSize: '12px' }}>+X</div>
      </Html>

      {/* Y Axis (Green) */}
      <Line
        points={[
          [0, 0, 0],
          [0, axisLength, 0],
        ]}
        color="green"
        lineWidth={2}
      />
      <Html position={[0, axisLength + 0.05, 0]} transform>
        <div style={{ color: 'green', fontSize: '12px' }}>+Y</div>
      </Html>

      {/* Z Axis (Blue) */}
      <Line
        points={[
          [0, 0, 0],
          [0, 0, axisLength],
        ]}
        color="blue"
        lineWidth={2}
      />
      <Html position={[0, 0, axisLength + 0.05]} transform>
        <div style={{ color: 'blue', fontSize: '12px' }}>+Z</div>
      </Html>
    </group>
  );
};

// Bounding Box to define the space
const BoundingBox = ({ size = 1.0, color = 'rgba(255,255,255,0.2)' }) => {
  const halfSize = size / 2;
  const points = [
    // Bottom square
    new THREE.Vector3(-halfSize, -halfSize, -halfSize), new THREE.Vector3(halfSize, -halfSize, -halfSize),
    new THREE.Vector3(halfSize, -halfSize, -halfSize), new THREE.Vector3(halfSize, halfSize, -halfSize),
    new THREE.Vector3(halfSize, halfSize, -halfSize), new THREE.Vector3(-halfSize, halfSize, -halfSize),
    new THREE.Vector3(-halfSize, halfSize, -halfSize), new THREE.Vector3(-halfSize, -halfSize, -halfSize),

    // Top square
    new THREE.Vector3(-halfSize, -halfSize, halfSize), new THREE.Vector3(halfSize, -halfSize, halfSize),
    new THREE.Vector3(halfSize, -halfSize, halfSize), new THREE.Vector3(halfSize, halfSize, halfSize),
    new THREE.Vector3(halfSize, halfSize, halfSize), new THREE.Vector3(-halfSize, halfSize, halfSize),
    new THREE.Vector3(-halfSize, halfSize, halfSize), new THREE.Vector3(-halfSize, -halfSize, halfSize),

    // Vertical lines
    new THREE.Vector3(-halfSize, -halfSize, -halfSize), new THREE.Vector3(-halfSize, -halfSize, halfSize),
    new THREE.Vector3(halfSize, -halfSize, -halfSize), new THREE.Vector3(halfSize, -halfSize, halfSize),
    new THREE.Vector3(halfSize, halfSize, -halfSize), new THREE.Vector3(halfSize, halfSize, halfSize),
    new THREE.Vector3(-halfSize, halfSize, -halfSize), new THREE.Vector3(-halfSize, halfSize, halfSize),
  ];

  return (
    <Line
      points={points}
      color={color}
      lineWidth={1}
      transparent
      opacity={0.3}
    />
  );
};


const SOM3DScene: React.FC<SOM3DSceneProps> = ({
  dataPoints,
  somGrid,
  bmuNode,
  bmuInputPoint,
  gridSize,
  currentIteration,
  totalIterations,
}) => {

  const normalizedDataPoints = dataPoints.map(p => ({
    x: normalizeCoord(p.x), y: normalizeCoord(p.y), z: normalizeCoord(p.z)
  }));

  const normalizedSomGrid: SOMNode[][] = somGrid.map(row =>
    row.map(node => ({
      ...node,
      x: normalizeCoord(node.x), y: normalizeCoord(node.y), z: normalizeCoord(node.z)
    }))
  );

  const normalizedBMUNode = bmuNode ? {
    ...bmuNode,
    x: normalizeCoord(bmuNode.x), y: normalizeCoord(bmuNode.y), z: normalizeCoord(bmuNode.z)
  } : null;

  const normalizedBMUInputPoint = bmuInputPoint ? {
    x: normalizeCoord(bmuInputPoint.x), y: normalizeCoord(bmuInputPoint.y), z: normalizeCoord(bmuInputPoint.z)
  } : null;


  return (
    <Canvas camera={{ position: [0.7, 0.7, 0.7], fov: 75 }}> {/* Adjusted default camera position */}
      <Suspense fallback={null}>
        <ambientLight intensity={0.5} />
        <pointLight position={[10, 10, 10]} />
        <pointLight position={[-10, -10, -10]} />


        {/* Render data points */}
        {normalizedDataPoints.map((p, index) => (
          <DataPoint key={`data-${index}`} position={[p.x, p.y, p.z]} color="rgba(96, 165, 250, 0.5)" size={0.015} />
        ))}

        {/* Render SOM nodes and connections */}
        {normalizedSomGrid.map((row, i) =>
          row.map((node, j) => (
            <SOMNodeMesh
              key={`node-${i}-${j}`}
              position={[node.x, node.y, node.z]}
              isBMU={normalizedBMUNode ? (node.i === normalizedBMUNode.i && node.j === normalizedBMUNode.j) : false}
            />
          ))
        )}
        <SOMLines somGrid={normalizedSomGrid} gridSize={gridSize} />

        {/* Render the current BMU input point */}
        {normalizedBMUInputPoint && (
          <DataPoint
            position={[normalizedBMUInputPoint.x, normalizedBMUInputPoint.y, normalizedBMUInputPoint.z]}
            color="#FFD700" // Gold color for BMU input point
            size={0.03} // Slightly larger to stand out
          />
        )}

        {/* Line connecting BMU Input to BMU Node */}
        {normalizedBMUInputPoint && normalizedBMUNode && (
          <Line
            points={[
              [normalizedBMUInputPoint.x, normalizedBMUInputPoint.y, normalizedBMUInputPoint.z],
              [normalizedBMUNode.x, normalizedBMUNode.y, normalizedBMUNode.z],
            ]}
            color="#FFD700" // Gold color for connection
            lineWidth={1.5}
            transparent
            opacity={0.8}
            dashed={true}
          />
        )}


        {/* Axes and Bounding Box */}
        <Axes />
        <BoundingBox />

        {/* Orbit Controls for interaction */}
        <OrbitControls enableZoom enablePan enableRotate />
      </Suspense>
    </Canvas>
  );
};

export default SOM3DScene;