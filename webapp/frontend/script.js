document.addEventListener('DOMContentLoaded', () => {
    const uploadButton = document.getElementById('upload-button');
    const fileInput = document.getElementById('audio-upload');
    const statusMessage = document.getElementById('status-message');
    const resultArea = document.getElementById('result-area');
    const resultAudio = document.getElementById('result-audio');

    uploadButton.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', async (event) => {
        const file = event.target.files[0];
        if (!file) {
            return;
        }

        statusMessage.textContent = `Đang tải lên và dịch file ${file.name}...`;
        statusMessage.parentElement.style.borderColor = '#ff9800'; // Orange for processing
        statusMessage.classList.add('processing');
        resultArea.classList.add('hidden');

        const formData = new FormData();
        formData.append('audio_file', file);

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData,
            });

            const data = await response.json();

            if (response.ok && data.success) {
                statusMessage.textContent = 'Dịch thành công! Đang phát âm thanh...';
                statusMessage.parentElement.style.borderColor = '#4caf50'; // Green for success
                statusMessage.classList.remove('processing');
                
                resultAudio.src = data.output_audio_url;
                resultArea.classList.remove('hidden');
                resultAudio.play();
            } else {
                throw new Error(data.error || 'Đã xảy ra lỗi không xác định.');
            }
        } catch (error) {
            statusMessage.textContent = `Lỗi: ${error.message}`;
            statusMessage.parentElement.style.borderColor = '#f44336'; // Red for error
            statusMessage.classList.remove('processing');
            console.error('Upload failed:', error);
        } finally {
            // Reset file input to allow re-uploading the same file
            fileInput.value = '';
        }
    });
});