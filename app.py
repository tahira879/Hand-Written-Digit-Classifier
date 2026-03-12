<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Digit Classifier - Dart Theme</title>
    <!-- Load TensorFlow.js -->
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest"></script>
    
    <style>
        :root {
            /* Dart (Dark) Theme Colors */
            --bg-color: #121212;
            --surface-color: #1e1e1e;
            --primary-color: #bb86fc;
            --secondary-color: #03dac6;
            --text-color: #e0e0e0;
            --canvas-bg: #ffffff;
            --danger-color: #cf6679;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-color);
            display: flex;
            flex-direction: column;
            align-items: center;
            min-height: 100vh;
            padding: 20px;
        }

        header {
            text-align: center;
            margin-bottom: 20px;
        }

        h1 {
            color: var(--primary-color);
            margin-bottom: 5px;
        }

        p.subtitle {
            color: #aaa;
            font-size: 0.9rem;
        }

        /* Main Layout */
        .main-container {
            display: flex;
            flex-wrap: wrap;
            gap: 30px;
            justify-content: center;
            width: 100%;
            max-width: 1000px;
        }

        /* Canvas Area */
        .canvas-wrapper {
            background: var(--surface-color);
            padding: 10px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        canvas {
            background-color: var(--canvas-bg);
            cursor: crosshair;
            border-radius: 4px;
            touch-action: none; /* Prevent scrolling on mobile while drawing */
        }

        /* Controls Area */
        .controls-wrapper {
            flex: 1;
            min-width: 300px;
            background: var(--surface-color);
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        /* Toolbar */
        .toolbar {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            padding-bottom: 15px;
            border-bottom: 1px solid #333;
        }

        button {
            padding: 10px 16px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.2s;
            font-size: 0.9rem;
        }

        .btn-tool {
            background-color: #333;
            color: var(--text-color);
        }

        .btn-tool.active {
            background-color: var(--primary-color);
            color: #000;
        }

        .btn-clear {
            background-color: var(--danger-color);
            color: white;
            margin-left: auto;
        }

        .btn-clear:hover {
            opacity: 0.9;
        }

        /* Pencil Options */
        .options-group {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        label {
            font-size: 0.85rem;
            color: #aaa;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        /* Slider */
        input[type="range"] {
            width: 100%;
            height: 6px;
            background: #444;
            border-radius: 5px;
            outline: none;
            -webkit-appearance: none;
        }
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 18px;
            height: 18px;
            background: var(--primary-color);
            border-radius: 50%;
            cursor: pointer;
        }

        /* Color Palette */
        .palette {
            display: flex;
            gap: 10px;
        }

        .color-swatch {
            width: 30px;
            height: 30px;
            border-radius: 50%;
            cursor: pointer;
            border: 2px solid transparent;
            transition: transform 0.2s;
        }

        .color-swatch.selected {
            border-color: white;
            transform: scale(1.1);
        }

        /* Prediction Results */
        .prediction-area {
            margin-top: auto;
        }

        .prediction-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            margin-bottom: 15px;
        }

        .big-score {
            font-size: 3.5rem;
            font-weight: bold;
            color: var(--secondary-color);
            line-height: 1;
        }

        .confidence-label {
            font-size: 0.9rem;
            color: #aaa;
        }

        .bar-chart {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }

        .bar-row {
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 0.9rem;
        }

        .digit-label {
            width: 20px;
            text-align: center;
            font-weight: bold;
        }

        .bar-track {
            flex: 1;
            height: 8px;
            background
