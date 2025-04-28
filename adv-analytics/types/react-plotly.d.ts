declare module 'react-plotly.js' {
  import * as React from 'react';
  
  interface PlotlyComponentProps {
    data: any[];
    layout?: any;
    config?: any;
    frames?: any[];
    revision?: number;
    style?: React.CSSProperties;
    className?: string;
    onInitialized?: (figure: any, graphDiv: any) => void;
    onUpdate?: (figure: any, graphDiv: any) => void;
    onPurge?: (figure: any, graphDiv: any) => void;
    onError?: (err: Error) => void;
    onClick?: (event: any) => void;
    onClickAnnotation?: (event: any) => void;
    onHover?: (event: any) => void;
    onUnhover?: (event: any) => void;
    onSelected?: (event: any) => void;
    onRelayout?: (event: any) => void;
    onRestyle?: (event: any) => void;
    onRedraw?: (event: any) => void;
    onAnimated?: (event: any) => void;
    onAfterPlot?: (event: any) => void;
    onAnimatingFrame?: (event: any) => void;
    onAnimationInterrupted?: (event: any) => void;
    onAutoSize?: (event: any) => void;
    onBeforeHover?: (event: any) => void;
    onButtonClicked?: (event: any) => void;
    onEvent?: (event: any) => void;
    onLegendClick?: (event: any) => void;
    onLegendDoubleClick?: (event: any) => void;
    onSliderChange?: (event: any) => void;
    onSliderEnd?: (event: any) => void;
    onSliderStart?: (event: any) => void;
    onTransitioning?: (event: any) => void;
    onTransitionInterrupted?: (event: any) => void;
    onDeselect?: (event: any) => void;
    onDoubleClick?: (event: any) => void;
    onFramework?: (event: any) => void;
    onSelecting?: (event: any) => void;
    onWebGlContextLost?: (event: any) => void;
  }

  const PlotComponent: React.ComponentClass<PlotlyComponentProps>;
  export default PlotComponent;
} 