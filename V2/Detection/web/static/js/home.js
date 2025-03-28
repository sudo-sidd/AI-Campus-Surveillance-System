// script.js
window.onload = function () {
    const cameraGrid = document.getElementById('cameraGrid');
    let cameras = [];
    // let IP = '127.0.0.1';
    let IP = '192.168.8.86';

    // fetch('/api/get_env')
    // .then(response => response.json())
    // .then((env) => {
    //     IP = env.IP;
    // })

    fetch('/api/cameras/')
        .then(response => {
            if (!response.ok) throw new Error(`Cameras fetch failed: ${response.status}`);
            return response.json();
        })
        .then(data => {
            cameras = data;
            console.log('Cameras loaded:', cameras);
            loadCameras();
        })
        .catch(error => {
            console.error("Initialization error:", error);
            alert("Failed to load cameras. Check console.");
        });

    function loadCameras() {
        const cameraFragments = document.createDocumentFragment();
        cameras.forEach((camera, index) => {
            const cameraDiv = document.createElement('div');
            cameraDiv.classList.add('col-md-4', 'mb-4');
            cameraDiv.setAttribute('id', 'camera' + index);
            cameraDiv.setAttribute('onclick', `openCamera(${index})`);

            cameraDiv.innerHTML = `
                <div class="card" style="width: fit-content; max-width: 350px; height: auto; border: 1px solid #ddd; border-radius: 8px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                    <img id="video${index}" class="card-img-top" src="" alt="${camera.camera_location} Live Feed" style="height: 180px; object-fit: contain; border-radius: 8px 8px 0 0;">
                    <div class="card-body d-flex flex-column justify-content-end" style="padding: 1px">
                        <p class="card-text mb-0" style="font-size: 16px; font-weight: 600; color: #333; text-align: center;">${camera.camera_location}</p>
                    </div>
                </div>
            `;
            cameraFragments.appendChild(cameraDiv);
        });
        cameraGrid.appendChild(cameraFragments);
        
        // Initialize camera feeds after all elements are in the DOM
        cameras.forEach((_, index) => {
            openCameraFeed(index);
        });
    }

    window.openCamera = function(cameraId) {
        window.location.href = `/fullscreen/?camera=${cameraId}`;
    };

    function openCameraFeed(index) {
        console.log(`video${index}`)
        const videoElement = document.getElementById(`video${index}`);
        console.log(videoElement)
        if (videoElement) {
            // Set the image source to the video feed endpoint with the camera index
            videoElement.src = `http://${IP}:7000/video/${index}`;
            console.log("Video Element :",videoElement)
            // Add error handling for the stream
            videoElement.onerror = () => {
                setTimeout(() => {
                    console.log(`Attempting to reconnect Camera ${index}...`);
                    videoElement.src = `http://${IP}:7000/video/${index}`;
                }, 5000);
                setTimeout(() => {
                    console.log(`Attempting to reconnect Camera ${index}...`);
                    videoElement.src = `http://${IP}/video/${index}`;
                }, 5000);
            };
            
            console.log(`Started video stream for Camera ${index}`);
        } else {
            console.error(`Video element not found for Camera ${index}`);
        }
    }
};
