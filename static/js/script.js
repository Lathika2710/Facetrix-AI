document.addEventListener('DOMContentLoaded', () => {
    // -------------------------------------------------------------
    // Drag and Drop Zone Handling
    // -------------------------------------------------------------
    const dropzones = document.querySelectorAll('.dropzone');

    dropzones.forEach(dropzone => {
        const fileInput = dropzone.querySelector('.file-input');
        const previewGrid = dropzone.parentElement.querySelector('.preview-grid');

        if (!fileInput) return;

        // Click to open file dialog
        dropzone.addEventListener('click', (e) => {
            if (e.target !== fileInput) {
                fileInput.click();
            }
        });

        // Drag events
        ['dragenter', 'dragover'].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.add('dragover');
            }, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.remove('dragover');
            }, false);
        });

        // Handle drop
        dropzone.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files.length > 0) {
                fileInput.files = files;
                updatePreviews(files, previewGrid);
            }
        });

        // Handle manual file input change
        fileInput.addEventListener('change', () => {
            if (fileInput.files.length > 0) {
                updatePreviews(fileInput.files, previewGrid);
            }
        });
    });

    function updatePreviews(files, previewGrid) {
        if (!previewGrid) return;
        previewGrid.innerHTML = '';

        Array.from(files).forEach(file => {
            if (!file.type.startsWith('image/')) return;

            const reader = new FileReader();
            reader.onload = (e) => {
                const item = document.createElement('div');
                item.className = 'preview-item';
                item.innerHTML = `<img src="${e.target.result}" alt="${file.name}" title="${file.name}">`;
                previewGrid.appendChild(item);
            };
            reader.readAsDataURL(file);
        });
    }

    // -------------------------------------------------------------
    // Auto-dismiss Alerts after 6 seconds
    // -------------------------------------------------------------
    const alerts = document.querySelectorAll('.alert');
    alerts.forEach(alert => {
        setTimeout(() => {
            alert.style.transition = 'opacity 0.5s ease';
            alert.style.opacity = '0';
            setTimeout(() => alert.remove(), 500);
        }, 6000);
    });
});
