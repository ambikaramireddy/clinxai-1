// UI State Management

const UI = {
    elements: {
        landing: document.getElementById('landing-page'),
        diagnose: document.getElementById('diagnose-page'),
        result: document.getElementById('result-page'),

        startBtn: document.getElementById('btn-start'),
        analyzeBtn: document.getElementById('btn-analyze'),
        resetBtn: document.getElementById('btn-reset'),

        fileInput: document.getElementById('file-input'),
        dropZone: document.getElementById('drop-zone'),
        imagePreview: document.getElementById('image-preview'),

        // Inputs
        inputAge: document.getElementById('inp-age'),
        inputGender: document.getElementById('inp-gender'),
        inputOxygen: document.getElementById('inp-oxygen'),
        inputFever: document.getElementById('inp-fever'),
        inputCough: document.getElementById('inp-cough'),

        // Results
        resDiagnosis: document.getElementById('res-diagnosis'),
        resConfidence: document.getElementById('res-confidence'),
        resRisk: document.getElementById('res-risk'),
        resHeatmap: document.getElementById('res-heatmap'),
        shapBars: document.getElementById('shap-bars'),

        statusIndicator: document.getElementById('system-status')
    },

    init() {
        // Enforce "Reload starts from beginning" policy
        if (!sessionStorage.getItem('from_landing')) {
            window.location.href = 'index.html';
            return;
        }
        // Consume the token so a subsequent reload fails this check
        sessionStorage.removeItem('from_landing');

        this.bindEvents();
        // Force initialization of state (including background)
        this.switchPage('diagnose');
    },

    bindEvents() {
        // Navigation
        if (this.elements.startBtn) {
            this.elements.startBtn.addEventListener('click', () => this.switchPage('diagnose'));
        }
        if (this.elements.resetBtn) {
            this.elements.resetBtn.addEventListener('click', () => this.resetApp());
        }

        // File Upload
        if (this.elements.dropZone && this.elements.fileInput) {
            this.elements.dropZone.addEventListener('click', () => this.elements.fileInput.click());
            this.elements.fileInput.addEventListener('change', (e) => this.handleFile(e.target.files[0]));

            // Drag & Drop
            this.elements.dropZone.addEventListener('dragover', (e) => { e.preventDefault(); this.elements.dropZone.classList.add('drag-over'); });
            this.elements.dropZone.addEventListener('dragleave', (e) => { e.preventDefault(); this.elements.dropZone.classList.remove('drag-over'); });
            this.elements.dropZone.addEventListener('drop', (e) => {
                e.preventDefault();
                this.elements.dropZone.classList.remove('drag-over');
                if (e.dataTransfer.files.length) this.handleFile(e.dataTransfer.files[0]);
            });
        }

        // Analysis
        if (this.elements.analyzeBtn) {
            this.elements.analyzeBtn.addEventListener('click', () => this.runAnalysis());
        }
    },

    switchPage(pageId) {
        // Simple class based switching
        document.querySelectorAll('section').forEach(el => {
            el.classList.remove('active-section');
            el.classList.add('hidden-section');
        });

        const target = document.getElementById(`${pageId}-page`);
        if (target) {
            target.classList.remove('hidden-section');
            target.classList.add('active-section');
        }

        // Hook for Background (Phase 5)
        if (window.setBackgroundState) {
            const stateMap = { 'landing': 'idle', 'diagnose': 'processing', 'result': 'result' };
            // For diagnose, we are technically 'idle' until we click analyze, but let's say 'idle' for now.
            // Wait, "Processing" state is specifically during inference.
            // Diagnosing page interaction is mostly Idle.
            if (pageId === 'diagnose') window.setBackgroundState('idle');
            if (pageId === 'landing') window.setBackgroundState('idle');
        }
    },

    handleFile(file) {
        if (!file) return;
        this.currentFile = file;

        const reader = new FileReader();
        reader.onload = (e) => {
            this.elements.imagePreview.src = e.target.result;
            this.elements.dropZone.classList.add('has-image');
        };
        reader.readAsDataURL(file);
    },

    async runAnalysis() {
        if (!this.currentFile) {
            alert("Please upload an X-ray image.");
            return;
        }

        this.setLoading(true);

        try {
            // Prepare Data
            const params = {
                age: this.elements.inputAge.value,
                gender: this.elements.inputGender.value,
                fever: this.elements.inputFever.checked ? 1 : 0,
                cough: this.elements.inputCough.checked ? 1 : 0,
                oxygen: this.elements.inputOxygen.value
            };

            const formData = new FormData();
            formData.append('file', this.currentFile);

            // 1. Prediction
            const prediction = await API.predictFused(formData, params);

            // 2. Explanations (Parallel)
            const [imgExpl, patExpl] = await Promise.all([
                API.explainImage(formData),
                API.explainPatient({
                    Age: parseInt(params.age),
                    Gender: params.gender,
                    Fever: params.fever,
                    Cough: params.cough,
                    Oxygen: parseInt(params.oxygen)
                })
            ]);

            this.showResults(prediction, imgExpl, patExpl);
            this.switchPage('result');

        } catch (err) {
            console.error(err);
            alert("Analysis failed. See console.");
        } finally {
            this.setLoading(false);
        }
    },

    setLoading(isLoading) {
        const btn = this.elements.analyzeBtn;
        if (isLoading) {
            btn.textContent = "ANALYZING...";
            btn.disabled = true;
            this.elements.statusIndicator.textContent = "Processing Inference...";
            this.elements.statusIndicator.classList.add('blinking');

            if (window.setBackgroundState) window.setBackgroundState('processing');
        } else {
            btn.textContent = "RUN DIAGNOSIS";
            btn.disabled = false;
            this.elements.statusIndicator.textContent = "System Ready";
            this.elements.statusIndicator.classList.remove('blinking');
        }
    },

    showResults(pred, imgExpl, patExpl) {
        // Diagnosis
        this.elements.resDiagnosis.textContent = pred.diagnosis.toUpperCase();
        this.elements.resDiagnosis.className = pred.diagnosis === 'Pneumonia' ? 'text-danger' : 'text-success';

        this.elements.resConfidence.textContent = "Prob. of Pneumonia: " + (pred.confidence * 100).toFixed(1) + "%";

        this.elements.resRisk.textContent = pred.risk_level + " RISK";
        this.elements.resRisk.className = `risk-badge risk-${pred.risk_level.toLowerCase()}`;

        // Heatmap
        this.elements.resHeatmap.src = `data:image/jpeg;base64,${imgExpl.heatmap_base64}`;

        // SHAP Bars
        this.elements.shapBars.innerHTML = '';
        patExpl.features.forEach(f => {
            const row = document.createElement('div');
            row.className = 'shap-row';

            const label = document.createElement('span');
            label.className = 'shap-label';
            label.textContent = `${f.feature} (${f.value})`;

            const barContainer = document.createElement('div');
            barContainer.className = 'shap-bar-container';

            const bar = document.createElement('div');
            bar.className = 'shap-bar';
            // Normalize width (visual guess)
            const width = Math.min(Math.abs(f.shap_value) * 100 * 2, 100);
            bar.style.width = width + '%';
            bar.style.backgroundColor = f.shap_value > 0 ? '#ef4444' : '#22c55e'; // Red adds risk, Green reduces

            barContainer.appendChild(bar);
            row.appendChild(label);
            row.appendChild(barContainer);
            this.elements.shapBars.appendChild(row);
        });

        // Update Background State
        if (window.setBackgroundState) window.setBackgroundState('result');
    },

    resetApp() {
        this.currentFile = null;
        this.elements.imagePreview.src = '';
        this.elements.dropZone.classList.remove('has-image');
        this.helpers_resetForm(); // Helper if needed
        this.switchPage('diagnose');
        if (window.setBackgroundState) window.setBackgroundState('idle');
    },

    helpers_resetForm() {
        this.elements.inputAge.value = 45;
        this.elements.inputOxygen.value = 95;
        this.elements.inputFever.checked = false;
        this.elements.inputCough.checked = false;
    }
};

// Initialize
document.addEventListener('DOMContentLoaded', () => UI.init());
