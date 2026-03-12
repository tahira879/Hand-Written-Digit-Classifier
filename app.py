<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Handwritten Digit Classifier - Color Palette</title>
    
    <!-- Load TensorFlow.js -->
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest"></script>

    <style>
        :root {
            --bg-dark: #0e1117; /* Streamlit Dark */
            --bg-sidebar: #262730;
            --primary: #ff4c4c; /* Accent color */
            --text-main: #ffffff;
            --text-muted: #b8b8b8;
            --border-color: #3d3f4b;
            --canvas-bg: #ffffff;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        }

        body {
            background-color: var(--bg-dark);
            color: var(--text-main);
            display: flex;
            height: 100vh;
            overflow: hidden;
        }

        /* --- Sidebar / Controls --- */
        .sidebar {
            width: 300px;
            background-color: var(--bg-sidebar);
            border-right: 1px solid var(--border-color);
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 25px;
            overflow-y: auto;
            flex-shrink: 0;
        }

        .brand {
            font-size: 1.2rem;
            font-weight: bold;
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 10px;
        }

        .brand span {
            color: var(--primary);
        }

        .control-group {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        .label {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-muted);
            font-weight: 600;
        }

        /* Tool Buttons */
        .tools-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }

        .btn {
            background: #3d3f4b;
            color: white;
            border: none;
            padding: 10px;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s;
            font-size: 0.9rem;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }

        .btn:hover {
            background: #4e505f;
        }

        .btn.active {
            background: var(--primary);
            color: white;
            box-shadow: 0 0 10px rgba(255, 76, 76, 0.4);
        }

        .btn-full {
            width: 100%;
            margin-top: 10px;
            font-weight: bold;
        }

        .btn-predict {
            background: #00c853; /* Green for predict */
            margin-top: auto;
        }
        .btn-predict:hover {
            background: #00e676;
            box-shadow: 0 0 15px rgba(0, 200, 83, 0.4);
        }
        .btn-clear {
            background: #d32f2f;
        }

        /* Palette Styles */
        .palette {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            align-items: center;
        }

        .color-swatch {
            width: 32px;
            height: 32px;
            border-radius: 50%;
            cursor: pointer;
            border: 2px solid transparent;
            transition: transform 0.2s, box-shadow 0.2s;
            flex-shrink: 0;
        }
        .color-swatch:hover { transform: scale(1.1); }
        .color-swatch.selected { 
            border-color: white; 
            box-shadow: 0 0 0 2px var(--primary); 
            transform: scale(1.1);
        }

        /* Custom Color Picker Input Styling */
        input[type="color"] {
            -webkit-appearance: none;
            border: none;
            width: 32px;
            height: 32px;
            border-radius: 50%;
            overflow: hidden;
            cursor: pointer;
            padding: 0;
            background: conic-gradient(red, yellow, lime, aqua, blue, magenta, red);
            box-shadow: 0 0 5px rgba(0,0,0,0.5);
            transition: transform 0.2s;
        }
        input[type="color"]:hover { transform: scale(1.1); }
        input[type="color"]::-webkit-color-swatch-wrapper { padding: 0; }
        input[type="color"]::-webkit-color-swatch { border: none; }

        /* Slider */
        input[type=range] {
            width: 100%;
            height: 6px;
            background: #3d3f4b;
            border-radius: 5px;
            outline: none;
            -webkit-appearance: none;
        }
        input[type=range]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 18px;
            height: 18px;
            background: var(--primary);
            border-radius: 50%;
            cursor: pointer;
        }

        /* --- Main Content --- */
        .main-content {
            flex: 1;
            padding: 30px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 30px;
        }

        /* Top Row: Canvas & Result */
        .top-row {
            display: flex;
            gap: 30px;
            height: 400px;
            flex-shrink: 0;
        }

        .canvas-container {
            flex: 1;
            background: #1a1d24;
            border-radius: 12px;
            border: 2px dashed var(--border-color);
            display: flex;
            justify-content: center;
            align-items: center;
            position: relative;
        }

        canvas {
            background: var(--canvas-bg);
            border-radius: 4px;
            cursor: crosshair;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }

        .result-container {
            width: 250px;
            background: #1a1d24;
            border-radius: 12px;
            border: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            padding: 20px;
        }

        .result-label {
            color: var(--text-muted);
            font-size: 0.9rem;
            margin-bottom: 10px;
        }

        .prediction-display {
            font-size: 6rem;
            font-weight: 800;
            color: var(--primary);
            line-height: 1;
            text-shadow: 0 0 20px rgba(255, 76, 76, 0.3);
            margin-bottom: 10px;
        }

        .confidence-score {
            font-size: 1.1rem;
            color: #4caf50;
        }

        /* Bottom Row: Prediction Table */
        .bottom-row {
            background: #1a1d24;
            border-radius: 12px;
            padding: 20px;
            border: 1px solid var(--border-color);
            flex: 1;
        }

        .section-title {
            margin-bottom: 15px;
            font-size: 1.1rem;
            color: var(--text-main);
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 10px;
        }

        .probability-table {
            width: 100%;
            border-collapse: collapse;
        }

        .prob-row {
            height: 40px;
        }

        .prob-cell-num {
            width: 50px;
            font-weight: bold;
            color: var(--text-muted);
        }

        .prob-bar-container {
            width: 100%;
            background: #0e1117;
            border-radius: 4px;
            overflow: hidden;
            height: 100%;
            position: relative;
        }

        .prob-bar {
            height: 100%;
            background: linear-gradient(90deg, #3d3f4b, var(--primary));
            width: 0%;
            transition: width 0.5s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .prob-value {
            width: 60px;
            text-align: right;
            font-family: monospace;
            color: var(--text-muted);
        }

        /* Loading Overlay */
        .loader {
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(14, 17, 23, 0.8);
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            z-index: 100;
            backdrop-filter: blur(2px);
            border-radius: 12px;
        }
        
        .spinner {
            width: 40px;
            height: 40px;
            border: 4px solid rgba(255,255,255,0.1);
            border-left-color: var(--primary);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-bottom: 10px;
        }

        @keyframes spin { 100% { transform: rotate(360deg); } }

        /* Toast */
        .toast {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: #262730;
            color: white;
            padding: 12px 24px;
            border-radius: 6px;
            border-left: 4px solid var(--primary);
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
            transform: translateY(100px);
            transition: transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            z-index: 1000;
        }
        .toast.show { transform: translateY(0); }

        /* Responsive */
        @media (max-width: 900px) {
            body { flex-direction: column; overflow-y: auto; }
            .sidebar { width: 100%; height: auto; border-right: none; border-bottom: 1px solid var(--border-color); }
            .top-row { flex-direction: column; height: auto; }
            .canvas-container { min-height: 300px; }
            .result-container { width: 100%; min-height: 100px; }
        }
    </style>
</head>
<body>

    <!-- Sidebar Controls -->
    <aside class="sidebar">
        <div class="brand">
            <span>◆</span> AI Digit Dashboard
        </div>

        <div class="control-group">
            <div class="label">Tools</div>
            <div class="tools-grid">
                <button class="btn active" id="btnPencil" onclick="setTool('pencil')">
                    ✎ Pencil
                </button>
                <button class="btn" id="btnEraser" onclick="setTool('eraser')">
                    ⌫ Eraser
                </button>
            </div>
        </div>

        <div class="control-group">
            <div class="label">Stroke Width <span id="widthVal">10</span>px</div>
            <input type="range" id="lineWidth" min="2" max="30" value="10" oninput="updateWidth(this.value)">
        </div>

        <div class="control-group">
            <div class="label">Palette</div>
            <div class="palette">
                <!-- Preset Colors -->
                <div class="color-swatch selected" style="background: #000000;" onclick="setColor('#000000', this)"></div>
                <div class="color-swatch" style="background: #2196F3;" onclick="setColor('#2196F3', this)"></div>
                <div class="color-swatch" style="background: #E91E63;" onclick="setColor('#E91E63', this)"></div>
                <div class="color-swatch" style="background: #4CAF50;" onclick="setColor('#4CAF50', this)"></div>
                <div class="color-swatch" style="background: #FFC107;" onclick="setColor('#FFC107', this)"></div>
                
                <!-- Custom Color Picker (Rainbow Circle) -->
                <input type="color" id="customColorPicker" value="#000000" title="Choose Custom Color">
            </div>
            <small style="color: #666; font-size: 0.7rem;">*Black ink recommended for accuracy</small>
        </div>

        <div style="margin-top: auto;">
            <button class="btn btn-clear btn-full" onclick="clearCanvas()">Clear Canvas</button>
            <button class="btn btn-predict btn-full" id="predictBtn" onclick="predictDigit()">PREDICT</button>
        </div>
    </aside>

    <!-- Main Content -->
    <main class="main-content">
        
        <!-- Top Section: Canvas and Result -->
        <div class="top-row">
            <!-- Col 1: Canvas -->
            <div class="canvas-container" id="canvasWrapper">
                <canvas id="drawingCanvas" width="280" height="280"></canvas>
                <div id="modelLoader" class="loader">
                    <div class="spinner"></div>
                    <div>Loading AI Model...</div>
                </div>
            </div>

            <!-- Col 2: Result -->
            <div class="result-container">
                <div class="result-label">PREDICTION</div>
                <div class="prediction-display" id="finalResult">-</div>
                <div class="confidence-score" id="confidenceText">Confidence: 0%</div>
            </div>
        </div>

        <!-- Bottom Section: Prediction Table -->
        <div class="bottom-row">
            <div class="section-title">Prediction Probabilities</div>
            <table class="probability-table">
                <tbody id="probTableBody">
                    <!-- Rows generated by JS -->
                </tbody>
            </table>
        </div>
    </main>

    <!-- Toast Notification -->
    <div id="toast" class="toast">Prediction Complete</div>

    <script>
        // --- 1. Canvas Setup & Logic ---
        const canvas = document.getElementById('drawingCanvas');
        const ctx = canvas.getContext('2d');
        const wrapper = document.getElementById('canvasWrapper');
        
        let isDrawing = false;
        let lastX = 0;
        let lastY = 0;
        
        // Tool State
        let currentTool = 'pencil';
        let currentColor = '#000000';
        let currentWidth = 10;

        // Initialize Canvas Background
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        // Drawing Event Listeners
        canvas.addEventListener('mousedown', startDrawing);
        canvas.addEventListener('mousemove', draw);
        canvas.addEventListener('mouseup', stopDrawing);
        canvas.addEventListener('mouseout', stopDrawing);
        
        // Touch support
        canvas.addEventListener('touchstart', (e) => {
            const touch = e.touches[0];
            const mouseEvent = new MouseEvent('mousedown', {
                clientX: touch.clientX,
                clientY: touch.clientY
            });
            canvas.dispatchEvent(mouseEvent);
        }, {passive: false});

        canvas.addEventListener('touchmove', (e) => {
            e.preventDefault(); // Prevent scrolling
            const touch = e.touches[0];
            const mouseEvent = new MouseEvent('mousemove', {
                clientX: touch.clientX,
                clientY: touch.clientY
            });
            canvas.dispatchEvent(mouseEvent);
        }, {passive: false});

        canvas.addEventListener('touchend', () => {
            const mouseEvent = new MouseEvent('mouseup', {});
            canvas.dispatchEvent(mouseEvent);
        });

        function getMousePos(evt) {
            const rect = canvas.getBoundingClientRect();
            // Scale logic in case canvas is resized via CSS
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;
            return {
                x: (evt.clientX - rect.left) * scaleX,
                y: (evt.clientY - rect.top) * scaleY
            };
        }

        function startDrawing(e) {
            isDrawing = true;
            const pos = getMousePos(e);
            lastX = pos.x;
            lastY = pos.y;
        }

        function draw(e) {
            if (!isDrawing) return;
            const pos = getMousePos(e);
            
            ctx.beginPath();
            ctx.moveTo(lastX, lastY);
            ctx.lineTo(pos.x, pos.y);
            
            ctx.lineWidth = currentWidth;
            if (currentTool === 'eraser') {
                ctx.strokeStyle = '#ffffff';
            } else {
                ctx.strokeStyle = currentColor;
            }
            
            ctx.stroke();
            lastX = pos.x;
            lastY = pos.y;
        }

        function stopDrawing() {
            isDrawing = false;
        }

        // --- 2. UI Interactions ---

        function setTool(tool) {
            currentTool = tool;
            document.querySelectorAll('.tools-grid .btn').forEach(b => b.classList.remove('active'));
            if(tool === 'pencil') document.getElementById('btnPencil').classList.add('active');
            else document.getElementById('btnEraser').classList.add('active');
        }

        // Handle Preset Color Click
        function setColor(color, element) {
            currentColor = color;
            setTool('pencil'); // Switch back to pencil if picking a color
            
            // Visual updates
            document.querySelectorAll('.color-swatch').forEach(el => el.classList.remove('selected'));
            if(element) element.classList.add('selected');
            
            // Sync the custom picker to this color (so if they open it, it starts here)
            document.getElementById('customColorPicker').value = color;
        }

        // Handle Custom Color Picker Change
        const colorPicker = document.getElementById('customColorPicker');
        colorPicker.addEventListener('input', (e) => {
            currentColor = e.target.value;
            setTool('pencil');
            
            // Remove 'selected' from preset swatches because we are now using a custom color
            document.querySelectorAll('.color-swatch').forEach(el => el.classList.remove('selected'));
        });

        function updateWidth(val) {
            currentWidth = val;
            document.getElementById('widthVal').innerText = val;
        }

        function clearCanvas() {
            ctx.fillStyle = '#ffffff';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            // Reset results
            document.getElementById('finalResult').innerText = '-';
            document.getElementById('confidenceText').innerText = 'Confidence: 0%';
            updateProbabilityTable(new Array(10).fill(0));
        }

        function showToast(msg) {
            const toast = document.getElementById('toast');
            toast.innerText = msg;
            toast.classList.add('show');
            setTimeout(() => toast.classList.remove('show'), 3000);
        }

        // Generate Table Structure
        const tableBody = document.getElementById('probTableBody');
        for(let i=0; i<10; i++) {
            const tr = document.createElement('tr');
            tr.className = 'prob-row';
            tr.innerHTML = `
                <td class="prob-cell-num">${i}</td>
                <td>
                    <div class="prob-bar-container">
                        <div class="prob-bar" id="bar-${i}"></div>
                    </div>
                </td>
                <td class="prob-value" id="val-${i}">0%</td>
            `;
            tableBody.appendChild(tr);
        }

        function updateProbabilityTable(probs) {
            for(let i=0; i<10; i++) {
                const pct = (probs[i] * 100).toFixed(1) + '%';
                document.getElementById(`bar-${i}`).style.width = pct;
                document.getElementById(`val-${i}`).innerText = pct;
            }
        }

        // --- 3. Machine Learning (TensorFlow.js) ---
        
        let model;
        const MODEL_URL = 'https://storage.googleapis.com/tfjs-models/tfjs/mnist/model.json';

        async function loadModel() {
            try {
                model = await tf.loadLayersModel(MODEL_URL);
                document.getElementById('modelLoader').style.display = 'none';
                console.log("Model loaded successfully");
            } catch (error) {
                console.error("Failed to load model", error);
                document.getElementById('modelLoader').innerHTML = "<div style='color:red'>Failed to load AI Model.<br>Check internet connection.</div>";
            }
        }

        // Load model on startup
        loadModel();

        async function predictDigit() {
            if (!model) {
                showToast("Model not loaded yet!");
                return;
            }

            const btn = document.getElementById('predictBtn');
            const originalText = btn.innerText;
            btn.innerText = "Processing...";
            btn.disabled = true;

            // Give UI a moment to update
            await new Promise(r => setTimeout(r, 50));

            // 1. Preprocess Image
            // Get image data from canvas
            let tensor = tf.browser.fromPixels(canvas, 1); // Grayscale
            
            // Resize to 28x28 (MNIST standard)
            const resized = tf.image.resizeBilinear(tensor, [28, 28]);
            
            // Normalize to 0-1
            const floatTensor = resized.toFloat().div(255.0);
            
            // Invert colors: MNIST is white digits on black background. 
            // Our canvas is colored digits on white background.
            // So we perform: 1 - pixel (Note: Colors are converted to grayscale first)
            const inverted = floatTensor.sub(1).abs(); 
            
            // Reshape to [1, 28, 28, 1] (batch size of 1)
            const batched = inverted.expandDims(0);

            // 2. Predict
            const prediction = model.predict(batched);
            const data = await prediction.data(); // Get array

            // 3. Find Max
            let maxProb = -1;
            let maxIndex = -1;
            for(let i=0; i<data.length; i++) {
                if(data[i] > maxProb) {
                    maxProb = data[i];
                    maxIndex = i;
                }
            }

            // 4. Update UI
            document.getElementById('finalResult').innerText = maxIndex;
            document.getElementById('confidenceText').innerText = `Confidence: ${(maxProb * 100).toFixed(1)}%`;
            
            // Color code the confidence
            const confText = document.getElementById('confidenceText');
            if(maxProb > 0.8) confText.style.color = "#4caf50"; // Green
            else if(maxProb > 0.5) confText.style.color = "#ff9800"; // Orange
            else confText.style.color = "#f44336"; // Red

            updateProbabilityTable(data);

            // Cleanup tensors
            tensor.dispose();
            resized.dispose();
            floatTensor.dispose();
            inverted.dispose();
            batched.dispose();
            prediction.dispose();

            showToast("Prediction Updated");
            
            btn.innerText = originalText;
            btn.disabled = false;
        }

    </script>
</body>
</html>
