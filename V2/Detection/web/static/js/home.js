// script.js

window.onload = function () {
    const cameraGrid = document.getElementById('cameraGrid');
    let cameras = [];

    // Fetch camera data from the backend
    fetch('/api/cameras/')  // Assuming your Django endpoint to serve the camera data
        .then(response => response.json())
        .then(data => {
            cameras = data;
            loadCameras(); // Start loading cameras dynamically
        })
        .catch(error => {
            console.error("Error loading camera data:", error);
        });

    // Function to load cameras dynamically
    function loadCameras() {
        const cameraFragments = document.createDocumentFragment(); // Use a fragment to avoid frequent DOM updates

        cameras.forEach((camera, index) => {
            const cameraDiv = document.createElement('div');
            cameraDiv.classList.add('col-md-4', 'mb-4');
            cameraDiv.setAttribute('id', 'camera' + index);
            cameraDiv.setAttribute('onclick', `openCamera(${index})`); // Attach event handler

            cameraDiv.innerHTML = `
                <div class="card" style="width: 100%; max-width: 350px; height: auto; border: 1px solid #ddd; border-radius: 8px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                    <img id="video${index}" class="card-img-top" src="" alt="${camera.camera_location} Live Feed" style="height: 180px; object-fit: cover; border-radius: 8px 8px 0 0;">
                    <div class="card-body d-flex flex-column justify-content-end" style="padding: 16px;">
                        <p class="card-text mb-0" style="font-size: 16px; font-weight: 600; color: #333; text-align: center;">${camera.camera_location}</p>
                    </div>
                </div>
            `;

            cameraFragments.appendChild(cameraDiv);

            // Open WebSocket connection for each camera feed
            openCameraFeed(camera, index);
        });

        // Append all camera elements at once to improve performance
        cameraGrid.appendChild(cameraFragments);
    }

    // Function to open the camera in full screen
    window.openCamera = function(cameraId) {
        // Redirect to fullscreen page with camera ID
        window.location.href = `/fullscreen/?camera=${cameraId}`;
    }

    // Function to open WebSocket connection for each camera and update the feed
    function openCameraFeed(camera, index) {
        const socketUrl = `ws://192.168.28.86:7000/ws/video/${index}/`;  // WebSocket URL
        const socket = new WebSocket(socketUrl);

        socket.onopen = function () {
            console.log(`Connected to WebSocket for Camera ${index}`);
        };

        socket.onmessage = function (event) {
            const data = JSON.parse(event.data);
            const frame = data.frame;

            // Update the camera feed image dynamically
            const videoElement = document.getElementById(`video${index}`);
            if (true) {
                videoElement.src = `data:image/jpeg;base64,${frame}`;  // Set the base64-encoded image
            }
        };

        socket.onerror = function (error) {
            console.error(`WebSocket error for Camera ${index}:`, error);
        };

        socket.onclose = function () {
            console.log(`Disconnected from WebSocket for Camera ${index}`);
        };
    }
};

