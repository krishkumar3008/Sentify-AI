// Sentify AI - Application Frontend State & Control Logic

// DOM Element Selectors
const doc = {
    // Top Bar
    statusDot: document.getElementById('status-dot'),
    statusText: document.getElementById('status-text'),
    
    // Training Configuration
    slider: document.getElementById('dataset-size-slider'),
    sliderVal: document.getElementById('dataset-size-val'),
    trainBtn: document.getElementById('train-btn'),
    
    // Progress Logging
    progressContainer: document.getElementById('progress-container'),
    progressBarFill: document.getElementById('progress-bar-fill'),
    progressStatusMsg: document.getElementById('progress-status-msg'),
    progressPercent: document.getElementById('progress-percent'),
    
    // Dashboard Cards & Widgets
    accuracyCard: document.getElementById('accuracy-chart-card'),
    metricsCard: document.getElementById('metrics-card'),
    metricsTbody: document.getElementById('metrics-tbody'),
    cmCard: document.getElementById('confusion-matrix-card'),
    
    // Confusion Matrix Elements
    cmTabs: document.querySelectorAll('.cm-tab'),
    cmTN: document.getElementById('cm-tn'),
    cmFP: document.getElementById('cm-fp'),
    cmFN: document.getElementById('cm-fn'),
    cmTP: document.getElementById('cm-tp'),
    
    // Chatbot Panel
    chatMessages: document.getElementById('chat-messages'),
    chatInput: document.getElementById('chat-input'),
    sendBtn: document.getElementById('send-btn'),
    chatbotAvatar: document.getElementById('chatbot-avatar'),
    botPersonaStatus: document.getElementById('bot-persona-status'),
    activeModelSelect: document.getElementById('active-model-select'),
    
    // Realtime Predictions Overlay
    predOverlay: document.getElementById('realtime-predictions-panel'),
    analyzedSnippet: document.getElementById('analyzed-text-snippet'),
    closePredBtn: document.getElementById('close-predictions-btn'),
    
    // Specific Predictions
    predNBSentiment: document.getElementById('pred-nb-sentiment'),
    predNBBar: document.getElementById('pred-nb-bar'),
    predNBConf: document.getElementById('pred-nb-conf'),
    predNBCard: document.getElementById('pred-nb-card'),
    
    // Logistic Regression
    predLRSentiment: document.getElementById('pred-lr-sentiment'),
    predLRBar: document.getElementById('pred-lr-bar'),
    predLRConf: document.getElementById('pred-lr-conf'),
    predLRCard: document.getElementById('pred-lr-card'),
    
    // LSTM
    predLSTMSentiment: document.getElementById('pred-lstm-sentiment'),
    predLSTMBar: document.getElementById('pred-lstm-bar'),
    predLSTMConf: document.getElementById('pred-lstm-conf'),
    predLSTMCard: document.getElementById('pred-lstm-card'),

    // Header Tabs
    tabBtnChat: document.getElementById('tab-btn-chat'),
    tabBtnBatch: document.getElementById('tab-btn-batch'),

    // Batch Analysis Panel
    batchPanel: document.getElementById('batch-analysis-panel'),
    batchTextInput: document.getElementById('batch-text-input'),
    batchFileInput: document.getElementById('batch-file-input'),
    batchFileStatus: document.getElementById('batch-file-status'),
    batchAnalyzeBtn: document.getElementById('batch-analyze-btn'),
    batchResultsArea: document.getElementById('batch-results-area'),
    batchKpiArea: document.getElementById('batch-kpi-area'),
    batchTableTbody: document.getElementById('batch-table-tbody'),
    kpiTotal: document.getElementById('kpi-total-val'),
    kpiPos: document.getElementById('kpi-pos-val'),
    kpiNeg: document.getElementById('kpi-neg-val'),
    kpiNeu: document.getElementById('kpi-neu-val')
};

// Global App State
let activeMetrics = null;
let pollInterval = null;
let chartInstance = null;
let batchChartInstance = null;
let batchUploadedTexts = [];

// Initialize Event Listeners
document.addEventListener('DOMContentLoaded', () => {
    initApp();
});

function initApp() {
    // Slider Change
    doc.slider.addEventListener('input', (e) => {
        doc.sliderVal.textContent = Number(e.target.value).toLocaleString();
    });

    // Train Models Button
    doc.trainBtn.addEventListener('click', triggerTraining);

    // Chat Actions
    doc.sendBtn.addEventListener('click', handleUserSendMessage);
    doc.chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !doc.chatInput.disabled) {
            handleUserSendMessage();
        }
    });

    // Confusion Matrix Tab Selector
    doc.cmTabs.forEach(tab => {
        tab.addEventListener('click', (e) => {
            doc.cmTabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            updateConfusionMatrixDisplay(tab.dataset.model);
        });
    });

    // Close Prediction Overlay
    doc.closePredBtn.addEventListener('click', () => {
        doc.predOverlay.classList.add('hidden');
    });

    // Tab Switching
    doc.tabBtnChat.addEventListener('click', () => switchTab('chat'));
    doc.tabBtnBatch.addEventListener('click', () => switchTab('batch'));

    // Batch File Upload Change
    doc.batchFileInput.addEventListener('change', handleBatchFileUpload);

    // Batch Analyze Button
    doc.batchAnalyzeBtn.addEventListener('click', triggerBatchAnalysis);

    // Initial check of backend state
    checkSystemStatus();
}

// Check if models are already trained
async function checkSystemStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        
        if (data.status === 'completed') {
            handleTrainingCompleted(data.metrics);
        } else if (['downloading', 'preprocessing', 'training_nb', 'training_lr', 'training_lstm'].includes(data.status)) {
            startPollingStatus();
        } else {
            setSystemReady(false, "Models Not Trained");
        }
    } catch (err) {
        console.error("Connection error checking status:", err);
        setSystemReady(false, "Server Offline");
    }
}

// Set top bar ready status
function setSystemReady(isReady, message) {
    if (isReady) {
        doc.statusDot.className = 'pulse-dot green';
        doc.statusText.textContent = 'Models Ready';
        doc.chatInput.disabled = false;
        doc.sendBtn.disabled = false;
        doc.chatInput.placeholder = 'Write a review/statement to test sentiment...';
        doc.batchAnalyzeBtn.disabled = false;
        doc.batchTextInput.disabled = false;
        doc.batchFileInput.disabled = false;
        doc.batchTextInput.placeholder = 'Paste reviews here. One review per line...\ne.g.\nThis movie was absolutely amazing!\nI hated it, total waste of time.\nIt was average, nothing special.';
    } else {
        doc.statusDot.className = 'pulse-dot red';
        doc.statusText.textContent = message;
        doc.chatInput.disabled = true;
        doc.sendBtn.disabled = true;
        doc.chatInput.placeholder = 'Please train the models on the left to activate chatbot...';
        doc.batchAnalyzeBtn.disabled = true;
        doc.batchTextInput.disabled = true;
        doc.batchFileInput.disabled = true;
        doc.batchTextInput.placeholder = 'Please train the models on the left to activate batch analyzer...';
    }
}

// Trigger background training pipeline
async function triggerTraining() {
    const sampleSize = parseInt(doc.slider.value);
    
    // Update UI to training state
    doc.trainBtn.disabled = true;
    doc.progressContainer.classList.remove('hidden');
    doc.progressBarFill.style.width = '0%';
    doc.progressStatusMsg.textContent = 'Contacting server...';
    doc.progressPercent.textContent = '0%';
    
    // Hide old dashboard data during new train
    doc.accuracyCard.classList.add('hidden');
    doc.metricsCard.classList.add('hidden');
    doc.cmCard.classList.add('hidden');
    
    try {
        const response = await fetch('/api/train', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ sample_size: sampleSize })
        });
        const result = await response.json();
        
        if (result.status === 'started') {
            startPollingStatus();
        } else {
            alert("Error starting training: " + result.message);
            resetTrainingUI();
        }
    } catch (err) {
        alert("Network error starting training. Is the backend running?");
        resetTrainingUI();
    }
}

// Start polling API status endpoint
function startPollingStatus() {
    doc.trainBtn.disabled = true;
    doc.progressContainer.classList.remove('hidden');
    
    if (pollInterval) clearInterval(pollInterval);
    
    pollInterval = setInterval(async () => {
        try {
            const response = await fetch('/api/status');
            const data = await response.json();
            
            // Update progress bar
            doc.progressBarFill.style.width = `${data.progress}%`;
            doc.progressPercent.textContent = `${data.progress}%`;
            doc.progressStatusMsg.textContent = data.message;
            
            if (data.status === 'completed') {
                clearInterval(pollInterval);
                handleTrainingCompleted(data.metrics);
                appendSystemMessage("Success: Models successfully trained! Chat is now active.");
            } else if (data.status === 'failed') {
                clearInterval(pollInterval);
                alert("Model training failed: " + data.error);
                resetTrainingUI();
                appendSystemMessage(`Error: Training failed - ${data.error}`);
            }
        } catch (err) {
            console.error("Polling status error:", err);
        }
    }, 1500);
}

// Reset UI state if training fails or resets
function resetTrainingUI() {
    doc.trainBtn.disabled = false;
    doc.progressContainer.classList.add('hidden');
    setSystemReady(false, "Training Failed");
}

// Action when training completes
function handleTrainingCompleted(metrics) {
    activeMetrics = metrics;
    
    // Stop logs, enable button
    doc.trainBtn.disabled = false;
    doc.progressContainer.classList.add('hidden');
    setSystemReady(true);
    
    // Reveal dashboard cards
    doc.accuracyCard.classList.remove('hidden');
    doc.metricsCard.classList.remove('hidden');
    doc.cmCard.classList.remove('hidden');
    
    // Update performance tables and charts
    renderPerformanceMetricsTable(metrics);
    renderAccuracyChart(metrics);
    
    // Trigger Confusion Matrix update (default Naïve Bayes)
    const activeTab = document.querySelector('.cm-tab.active');
    updateConfusionMatrixDisplay(activeTab ? activeTab.dataset.model : 'naive_bayes');
}

// Render the Accuracy Comparison chart using Chart.js
function renderAccuracyChart(metrics) {
    const ctx = document.getElementById('accuracy-chart').getContext('2d');
    
    const nbAcc = metrics.naive_bayes.accuracy * 100;
    const lrAcc = metrics.logistic_regression.accuracy * 100;
    const lstmAcc = metrics.lstm.accuracy * 100;
    
    const data = {
        labels: ['Naïve Bayes', 'Logistic Regression', 'LSTM (PyTorch)'],
        datasets: [{
            label: 'Accuracy %',
            data: [nbAcc, lrAcc, lstmAcc],
            backgroundColor: [
                'rgba(167, 139, 250, 0.7)', // Purple tint
                'rgba(99, 102, 241, 0.7)',  // Indigo tint
                'rgba(16, 185, 129, 0.7)'   // Emerald tint
            ],
            borderColor: [
                '#a78bfa',
                '#6366f1',
                '#10b981'
            ],
            borderWidth: 1.5,
            borderRadius: 8
        }]
    };
    
    const config = {
        type: 'bar',
        data: data,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: {
                    min: 50,
                    max: 100,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)'
                    },
                    ticks: {
                        color: '#9ca3af',
                        font: { family: 'Inter' }
                    }
                },
                x: {
                    grid: { display: false },
                    ticks: {
                        color: '#9ca3af',
                        font: { family: 'Inter' }
                    }
                }
            }
        }
    };
    
    if (chartInstance) {
        chartInstance.destroy();
    }
    
    chartInstance = new Chart(ctx, config);
}

// Populate the Metrics table
function renderPerformanceMetricsTable(metrics) {
    const models = [
        { name: 'Naïve Bayes', key: 'naive_bayes' },
        { name: 'Logistic Regression', key: 'logistic_regression' },
        { name: 'LSTM (PyTorch)', key: 'lstm' }
    ];
    
    let html = '';
    models.forEach(m => {
        const modelData = metrics[m.key];
        html += `
            <tr>
                <td>${m.name}</td>
                <td><strong>${(modelData.accuracy * 100).toFixed(1)}%</strong></td>
                <td>${(modelData.precision * 100).toFixed(1)}%</td>
                <td>${(modelData.recall * 100).toFixed(1)}%</td>
                <td>${(modelData.f1_score * 100).toFixed(1)}%</td>
            </tr>
        `;
    });
    
    doc.metricsTbody.innerHTML = html;
}

// Update Confusion Matrix Grid
function updateConfusionMatrixDisplay(modelKey) {
    if (!activeMetrics || !activeMetrics[modelKey]) return;
    
    const cm = activeMetrics[modelKey].confusion_matrix;
    // cm is [[TN, FP], [FN, TP]]
    const tn = cm[0][0];
    const fp = cm[0][1];
    const fn = cm[1][0];
    const tp = cm[1][1];
    
    doc.cmTN.textContent = tn.toLocaleString();
    doc.cmFP.textContent = fp.toLocaleString();
    doc.cmFN.textContent = fn.toLocaleString();
    doc.cmTP.textContent = tp.toLocaleString();
}

// User Message Sent Handler
async function handleUserSendMessage() {
    const text = doc.chatInput.value.trim();
    if (!text) return;
    
    // Add user bubble to chat layout
    appendChatMessage('user', text);
    doc.chatInput.value = '';
    
    // Disable inputs during inference
    doc.chatInput.disabled = true;
    doc.sendBtn.disabled = true;
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text })
        });
        
        if (!response.ok) {
            throw new Error("Prediction API error");
        }
        
        const data = await response.json();
        
        // Show predictions overlay panel
        renderRealtimePredictionOverlay(text, data.predictions);
        
        // Formulate dynamic bot reply
        generateChatbotResponse(data.predictions);
        
    } catch (err) {
        appendChatMessage('bot', "Oops, I encountered a glitch trying to process that statement. Please make sure the backend is active!");
    } finally {
        doc.chatInput.disabled = false;
        doc.sendBtn.disabled = false;
        doc.chatInput.focus();
    }
}

// Append Chat Message HTML bubble
function appendChatMessage(sender, text) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${sender}`;
    
    const timestampStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    
    messageDiv.innerHTML = `
        <div class="message-bubble">${text}</div>
        <div class="message-timestamp">${timestampStr}</div>
    `;
    
    doc.chatMessages.appendChild(messageDiv);
    // Smooth scroll to bottom
    doc.chatMessages.scrollTo({
        top: doc.chatMessages.scrollHeight,
        behavior: 'smooth'
    });
}

// Append System Message
function appendSystemMessage(text) {
    const systemDiv = document.createElement('div');
    systemDiv.style.textAlign = 'center';
    systemDiv.style.margin = '10px 0';
    systemDiv.style.fontSize = '12px';
    systemDiv.style.color = 'var(--text-secondary)';
    systemDiv.style.fontStyle = 'italic';
    systemDiv.innerText = text;
    doc.chatMessages.appendChild(systemDiv);
    doc.chatMessages.scrollTop = doc.chatMessages.scrollHeight;
}

// Populates and slides up the realtime prediction analysis widget
function renderRealtimePredictionOverlay(originalText, predictions) {
    // Truncate text snippet
    const maxLen = 45;
    const truncatedText = originalText.length > maxLen ? originalText.substring(0, maxLen) + "..." : originalText;
    doc.analyzedSnippet.textContent = `"${truncatedText}"`;
    
    // NB Data
    populateIndividualPred(predictions.naive_bayes, doc.predNBSentiment, doc.predNBBar, doc.predNBConf, doc.predNBCard);
    
    // LR Data
    populateIndividualPred(predictions.logistic_regression, doc.predLRSentiment, doc.predLRBar, doc.predLRConf, doc.predLRCard);
    
    // LSTM Data
    populateIndividualPred(predictions.lstm, doc.predLSTMSentiment, doc.predLSTMBar, doc.predLSTMConf, doc.predLSTMCard);
    
    doc.predOverlay.classList.remove('hidden');
}

function populateIndividualPred(predObj, sentimentEl, barFillEl, labelConfEl, cardEl) {
    const isPos = predObj.sentiment === 'positive';
    const isNeg = predObj.sentiment === 'negative';
    const isNeu = predObj.sentiment === 'neutral';
    
    sentimentEl.textContent = predObj.sentiment;
    sentimentEl.className = `pred-sentiment ${predObj.sentiment}`;
    
    const confVal = predObj.confidence * 100;
    barFillEl.style.width = `${confVal}%`;
    labelConfEl.textContent = `Confidence: ${confVal.toFixed(1)}%`;
    
    cardEl.className = 'prediction-card';
    if (isPos) {
        cardEl.classList.add('pos-style');
    } else if (isNeg) {
        cardEl.classList.add('neg-style');
    } else {
        cardEl.classList.add('neu-style');
    }
}

// Generate Chatbot Reply and Dynamically change persona avatar
function generateChatbotResponse(predictions) {
    const selectedModel = doc.activeModelSelect.value;
    const modelPrediction = predictions[selectedModel];
    const sentiment = modelPrediction.sentiment;
    const confidence = modelPrediction.confidence * 100;
    
    // List of dynamic bot replies to choose from
    const positiveReplies = [
        `That sounds really positive! 😊 (My ${selectedModel} model detects positive sentiment with ${confidence.toFixed(1)}% confidence). It's great to hear such encouraging words!`,
        `I sense positive vibes from your message! ✨ According to ${selectedModel} (${confidence.toFixed(1)}% confidence), this is definitely positive. Thank you for sharing that upbeat review!`,
        `Awesome! 🌟 That review sounds positive! The ${selectedModel} model classifies this as positive with ${confidence.toFixed(1)}% confidence.`
    ];
    
    const negativeReplies = [
        `I've analyzed your text and detected negative sentiment. 😔 (Classified by ${selectedModel} with ${confidence.toFixed(1)}% confidence). It sounds like you had a frustrating or disappointing experience. I hope things improve!`,
        `That sounds a bit negative. 😢 My ${selectedModel} model is ${confidence.toFixed(1)}% sure of this. I understand how disappointing a bad experience can be.`,
        `Oops, that doesn't sound very good. 💔 My ${selectedModel} model detects negative sentiment with ${confidence.toFixed(1)}% confidence. Thanks for being honest, hopefully it gets better next time!`
    ];

    const neutralReplies = [
        `I've analyzed your text and detected neutral sentiment. 😐 (Classified by ${selectedModel} with ${confidence.toFixed(1)}% confidence). It seems quite balanced or informative!`,
        `Your message appears to be neutral or objective. 🤖 According to ${selectedModel} (${confidence.toFixed(1)}% confidence), this is neutral. Thanks for sharing this objective view!`,
        `Understood. 📝 My ${selectedModel} model detects neutral sentiment with ${confidence.toFixed(1)}% confidence. It's balanced and matter-of-fact.`
    ];
    
    let botReply = '';
    
    if (sentiment === 'positive') {
        // Positive Response style
        doc.chatbotAvatar.textContent = '😊';
        doc.chatbotAvatar.style.filter = 'drop-shadow(0 0 10px rgba(16, 185, 129, 0.5))';
        doc.botPersonaStatus.textContent = 'Persona: Positive & Encouraging';
        botReply = positiveReplies[Math.floor(Math.random() * positiveReplies.length)];
    } else if (sentiment === 'negative') {
        // Negative Response style
        doc.chatbotAvatar.textContent = '😔';
        doc.chatbotAvatar.style.filter = 'drop-shadow(0 0 10px rgba(244, 63, 94, 0.5))';
        doc.botPersonaStatus.textContent = 'Persona: Empathetic & Support';
        botReply = negativeReplies[Math.floor(Math.random() * negativeReplies.length)];
    } else {
        // Neutral Response style
        doc.chatbotAvatar.textContent = '😐';
        doc.chatbotAvatar.style.filter = 'drop-shadow(0 0 10px rgba(245, 158, 11, 0.5))';
        doc.botPersonaStatus.textContent = 'Persona: Balanced & Matter-of-Fact';
        botReply = neutralReplies[Math.floor(Math.random() * neutralReplies.length)];
    }
    
    // Delay bubble slightly for realism
    setTimeout(() => {
        appendChatMessage('bot', botReply);
    }, 600);
}

// Tab Switching Control
function switchTab(tab) {
    if (tab === 'chat') {
        doc.tabBtnChat.classList.add('active');
        doc.tabBtnBatch.classList.remove('active');
        
        doc.chatMessages.classList.remove('hidden');
        // Retrieve parent chat input area
        const inputArea = document.querySelector('.chat-input-area');
        if (inputArea) inputArea.classList.remove('hidden');
        doc.batchPanel.classList.add('hidden');
    } else if (tab === 'batch') {
        doc.tabBtnChat.classList.remove('active');
        doc.tabBtnBatch.classList.add('active');
        
        doc.chatMessages.classList.add('hidden');
        const inputArea = document.querySelector('.chat-input-area');
        if (inputArea) inputArea.classList.add('hidden');
        doc.predOverlay.classList.add('hidden');
        doc.batchPanel.classList.remove('hidden');
    }
}

// Batch File Upload Parser
function handleBatchFileUpload(event) {
    const file = event.target.files[0];
    if (!file) {
        doc.batchFileStatus.textContent = 'No file chosen';
        batchUploadedTexts = [];
        return;
    }
    
    doc.batchFileStatus.textContent = `Selected: ${file.name}`;
    
    const reader = new FileReader();
    reader.onload = function(e) {
        const content = e.target.result;
        if (file.name.endsWith('.csv')) {
            batchUploadedTexts = parseCSVColumn(content);
        } else {
            // text file, split by lines
            batchUploadedTexts = content.split('\n')
                .map(line => line.trim())
                .filter(line => line.length > 0);
        }
        
        doc.batchFileStatus.textContent = `${file.name} (${batchUploadedTexts.length} rows loaded)`;
        
        // Preview content in the textarea
        const preview = batchUploadedTexts.slice(0, 10).join('\n');
        doc.batchTextInput.value = preview + (batchUploadedTexts.length > 10 ? '\n... and ' + (batchUploadedTexts.length - 10) + ' more' : '');
    };
    reader.readAsText(file);
}

// Simple CSV column finder (looks for review, text, tweet, content, comment)
function parseCSVColumn(csvText) {
    const lines = csvText.split('\n');
    if (lines.length === 0) return [];
    
    // Find header index
    const headers = lines[0].split(',').map(h => h.trim().toLowerCase().replace(/"/g, ''));
    let colIdx = 0;
    
    const candidates = ['review', 'text', 'tweet', 'content', 'sentence', 'statement', 'comment'];
    for (let candidate of candidates) {
        const idx = headers.indexOf(candidate);
        if (idx !== -1) {
            colIdx = idx;
            break;
        }
    }
    
    const results = [];
    for (let i = 1; i < lines.length; i++) {
        const line = lines[i].trim();
        if (!line) continue;
        
        // Handle commas inside quotes in CSV
        let fields = [];
        let inQuotes = false;
        let field = '';
        for (let char of line) {
            if (char === '"') {
                inQuotes = !inQuotes;
            } else if (char === ',' && !inQuotes) {
                fields.push(field.trim());
                field = '';
            } else {
                field += char;
            }
        }
        fields.push(field.trim());
        
        if (fields[colIdx]) {
            results.push(fields[colIdx].replace(/^"|"$/g, '').trim());
        }
    }
    return results;
}

// Run batch sentiment analysis via API
async function triggerBatchAnalysis() {
    let texts = [];
    
    // Check if textbox value is valid and not just preview
    const textBoxVal = doc.batchTextInput.value.trim();
    if (textBoxVal && batchUploadedTexts.length === 0) {
        texts = textBoxVal.split('\n').map(t => t.trim()).filter(t => t.length > 0);
    } else {
        texts = batchUploadedTexts;
    }
    
    if (texts.length === 0) {
        alert("Please paste some text lines or upload a valid file first.");
        return;
    }
    
    // Show loading
    doc.batchAnalyzeBtn.disabled = true;
    doc.batchAnalyzeBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Analyzing...';
    
    const selectedModel = doc.activeModelSelect.value;
    
    try {
        const response = await fetch('/api/predict_batch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ texts: texts })
        });
        
        if (!response.ok) {
            throw new Error("Batch prediction request failed");
        }
        
        const data = await response.json();
        const results = data.results;
        
        // Count sentiments
        let pos = 0, neg = 0, neu = 0;
        let tbodyHtml = '';
        
        results.forEach(res => {
            const pred = res.predictions[selectedModel];
            const sentiment = pred.sentiment;
            if (sentiment === 'positive') pos++;
            else if (sentiment === 'negative') neg++;
            else if (sentiment === 'neutral') neu++;
            
            tbodyHtml += `
                <tr>
                    <td title="${res.text.replace(/"/g, '&quot;')}">${res.text}</td>
                    <td><span class="batch-badge ${sentiment}">${sentiment}</span></td>
                </tr>
            `;
        });
        
        const total = results.length;
        doc.kpiTotal.textContent = total.toLocaleString();
        doc.kpiPos.textContent = `${((pos / total) * 100).toFixed(1)}%`;
        doc.kpiNeg.textContent = `${((neg / total) * 100).toFixed(1)}%`;
        doc.kpiNeu.textContent = `${((neu / total) * 100).toFixed(1)}%`;
        
        doc.batchTableTbody.innerHTML = tbodyHtml;
        
        // Render Chart.js Donut Chart
        renderBatchDonutChart(pos, neg, neu);
        
        // Reveal elements
        doc.batchResultsArea.classList.remove('hidden');
        doc.batchKpiArea.classList.remove('hidden');
        
    } catch (err) {
        alert("Error running batch analysis. Check that the backend is active.");
        console.error(err);
    } finally {
        doc.batchAnalyzeBtn.disabled = false;
        doc.batchAnalyzeBtn.innerHTML = '<i class="fa-solid fa-chart-pie"></i> Analyze Batch';
    }
}

// Render Donut Chart using Chart.js
function renderBatchDonutChart(posCount, negCount, neuCount) {
    const ctx = document.getElementById('batch-donut-chart').getContext('2d');
    
    const data = {
        labels: ['Positive %', 'Negative %', 'Neutral %'],
        datasets: [{
            data: [posCount, negCount, neuCount],
            backgroundColor: [
                'rgba(16, 185, 129, 0.75)', // Emerald
                'rgba(244, 63, 94, 0.75)',  // Rose
                'rgba(245, 158, 11, 0.75)'   // Amber
            ],
            borderColor: [
                '#10b981',
                '#f43f5e',
                '#f59e0b'
            ],
            borderWidth: 1.5
        }]
    };
    
    const config = {
        type: 'doughnut',
        data: data,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right',
                    labels: {
                        color: '#9ca3af',
                        font: { family: 'Inter', size: 11 }
                    }
                }
            },
            cutout: '70%'
        }
    };
    
    if (batchChartInstance) {
        batchChartInstance.destroy();
    }
    
    batchChartInstance = new Chart(ctx, config);
}
