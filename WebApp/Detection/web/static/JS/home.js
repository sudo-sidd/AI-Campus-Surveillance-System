window.onload = function() {
    const cameraGrid = document.getElementById('cameraGrid');
    let totalCameras = 3;  
    // let camerasPerPage = 9; 
    let camerasDisplayed = 0;

    function loadCameras() {
        let camerasToLoad = Math.min(totalCameras - camerasDisplayed); // Adjust to load cameras in defined chunks
        for (let i = 0; i < camerasToLoad; i++) {
            const cameraId = camerasDisplayed + i;
            const cameraDiv = document.createElement('div');
            cameraDiv.classList.add('col-md-4', 'mb-4'); // Bootstrap grid for 3 cameras per row
            cameraDiv.setAttribute('id', 'camera' + cameraId);
            cameraDiv.setAttribute('onclick', `openCamera(${cameraId})`);

            // Dynamically load video stream from backend for each camera
            cameraDiv.innerHTML = `
                <div class="card" style='width: 350px; height: 180px;'>
                    <img id="video${cameraId}" class="card-img-top" src="/live_stream/${cameraId}" alt="Camera ${cameraId} Live Feed" style="height: 180px;">
                    <div class="card-body text-center">
                        <p class="card-text">Camera ${cameraId}</p>
                    </div>
                </div>
            `;
            cameraGrid.appendChild(cameraDiv);
        }
        camerasDisplayed += camerasToLoad;
    }

    // Initial load of 9 cameras
    loadCameras();

    // Load more cameras when the user scrolls to the bottom of the grid
    cameraGrid.addEventListener('scroll', function() {
        if (cameraGrid.scrollTop + cameraGrid.clientHeight >= cameraGrid.scrollHeight) {
            loadCameras();
        }
    });
};

// Function to open the camera in full screen and play the video stream using Bootstrap modal
function openCamera(cameraId) {
    const modalContent = document.getElementById('modalCameraContent');

    // Use image tag to display full-screen live stream in the modal
    modalContent.innerHTML = `
        <img id="video${cameraId}-fullscreen" class="img-fluid" src="/live_stream/${cameraId}" alt="Camera ${cameraId} Live Feed">
    `;

    // Show the Bootstrap modal
    $('#cameraModal').modal('show');
}

// Function to close the full-screen camera view
function closeCamera() {
    $('#cameraModal').modal('hide');  // Use Bootstrap modal hide method
}
