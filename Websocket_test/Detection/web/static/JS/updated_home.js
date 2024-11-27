window.onload = function() {
    const cameraGrid = document.getElementById('cameraGrid');
    let cameras = [];

    // Fetch camera data from the JSON file
    fetch('/static/data.json')
        .then(response => response.json())
        .then(data => {
            cameras = data;
            loadCameras(); // Start loading cameras dynamically
        })
        .catch(error => {
            console.error("Error loading camera data:", error);
        });

    function loadCameras() {
        cameras.forEach((camera, index) => {
            const cameraDiv = document.createElement('div');
            cameraDiv.classList.add('col-md-4', 'mb-4'); // Bootstrap grid for 3 cameras per row
            cameraDiv.setAttribute('id', 'camera' + index);
            cameraDiv.setAttribute('onclick', `openCamera(${index})`);

            // Dynamically load video stream from backend for each camera
            cameraDiv.innerHTML = `
                <div class="card" style='width: 350px; height: 180px;'>
                    <img id="video${index}" class="card-img-top" src="" alt="${camera.camera_location} Live Feed" style="height: 180px;">
                    <div class="card-body text-center">
                        <p class="card-text">${camera.camera_location}</p>
                    </div>
                </div>
            `;
            cameraGrid.appendChild(cameraDiv);

            // Open WebSocket connection for each camera feed
            openCameraFeed(camera, index);
        });
    }

    // Function to open the camera in full screen and play the video stream using Bootstrap modal
    function openCamera(index) {
        const modalContent = document.getElementById('modalCameraContent');
        
        // Set the source of the fullscreen image dynamically
        const videoElement = document.getElementById(`video${index}`);
        if (videoElement) {
            const videoSrc = videoElement.src; // Get the current video stream (base64 or URL)
            modalContent.src = videoSrc; // Set the modal image source to the current video
        }

        // Show the Bootstrap modal
        $('#cameraModal').modal('show');
    }

    // Open WebSocket connection for camera and update the feed
    function openCameraFeed(camera, index) {
        const socket = new WebSocket(`ws://192.168.137.91:7000/ws/video/${index}/`);  // FastAPI WebSocket endpoint

        socket.onopen = function() {
            console.log(`Connected to WebSocket for Camera ${index}`);
        };

        socket.onmessage = function(event) {
            const data = JSON.parse(event.data);
            const frame = data.frame;

            // Update the camera feed image dynamically (using the base64-encoded JPEG image)
            const videoElement = document.getElementById(`video${index}`);
            if (videoElement) {
                videoElement.src = `data:image/jpeg;base64,${frame}`;  // Set the base64-encoded image
            }

            // Also update the fullscreen image if the modal is opened
            const fullscreenVideoElement = document.getElementById(`video${index}-fullscreen`);
            if (fullscreenVideoElement) {
                fullscreenVideoElement.src = `data:image/jpeg;base64,${frame}`;  // Set the base64-encoded image for the modal
            }
        };

        socket.onerror = function(error) {
            console.error(`WebSocket error for Camera ${index}:`, error);
        };

        socket.onclose = function() {
            console.log(`Disconnected from WebSocket for Camera ${index}`);
        };
    }

    // Function to close the full-screen camera view
    function closeCamera() {
        $('#cameraModal').modal('hide');  // Use Bootstrap modal hide method
    }
};
