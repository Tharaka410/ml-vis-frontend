// components/PlotlyGraph.tsx
"use client";

import React, { useEffect, useRef } from "react";
import Plotly from "plotly.js-dist"; // Import Plotly.js for runtime functions
import { Data, Layout } from "plotly.js"; // Explicitly import types from @types/plotly.js

interface PlotlyGraphProps {
  plotData: Data[]; // Use the directly imported 'Data' type
  layout: Partial<Layout>; // Use the directly imported 'Layout' type
  divRef: React.RefObject<HTMLDivElement | null>; // Allow divRef to be null
}

const PlotlyGraph: React.FC<PlotlyGraphProps> = ({ plotData, layout, divRef }) => {
  useEffect(() => {
    // Ensure divRef.current is not null before attempting to plot
    if (divRef.current) {
      Plotly.newPlot(divRef.current, plotData, layout);

      return () => {
        // Ensure divRef.current is not null before purging
        if (divRef.current) {
          Plotly.purge(divRef.current);
        }
      };
    }
  }, [plotData, layout, divRef]); // Include divRef in dependencies

  return <div ref={divRef} className="w-full h-full" />;
};

export default PlotlyGraph;