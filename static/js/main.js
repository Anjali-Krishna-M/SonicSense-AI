const dropZone = document.querySelector('.drop-zone');
const fileInput = document.getElementById('fileInput');
const uploadForm = document.getElementById('uploadForm');
const analyzeBtn = document.getElementById('analyzeBtn');
const loader = document.getElementById('loader');
const uploadText = document.getElementById('uploadText');

function handleFileSelect(files) {
    if (files.length > 0) {
        // Update text to show filename
        uploadText.innerText = "Selected: " + files[0].name;
        uploadText.style.color = "#00d2ff";
        uploadText.style.fontWeight = "bold";
        
        // Show the Analyze button
        if (analyzeBtn) analyzeBtn.classList.remove('hidden');
    }
}

// Handle Drag & Drop Visuals
if (dropZone) {
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault(); dropZone.style.borderColor = '#00d2ff'; dropZone.style.background = 'rgba(0, 210, 255, 0.05)';
    });
    dropZone.addEventListener('dragleave', () => {
        dropZone.style.borderColor = 'rgba(255, 255, 255, 0.2)'; dropZone.style.background = 'transparent';
    });
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = 'rgba(255, 255, 255, 0.2)'; dropZone.style.background = 'transparent';
        if (e.dataTransfer.files.length) {
            fileInput.files = e.dataTransfer.files;
            handleFileSelect(e.dataTransfer.files);
        }
    });
    // Handle Click Selection
    fileInput.addEventListener('change', () => handleFileSelect(fileInput.files));
}

// Handle Manual Form Submission
if (uploadForm) {
    uploadForm.addEventListener('submit', () => {
        // Hide button, show loader
        if (analyzeBtn) analyzeBtn.classList.add('hidden');
        if (loader) loader.classList.remove('hidden');
    });
}