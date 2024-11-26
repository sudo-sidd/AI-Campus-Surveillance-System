// WebSocket connection to the backend server
const ws = new WebSocket("ws://localhost:8765");

// Camera grid element
const cameraGrid = document.getElementById("cameraGrid");

// Function to render detections
function renderDetections(detections) {
    // Clear the camera grid
    cameraGrid.innerHTML = "";

    // Iterate over detections and render cards
    detections.forEach((det, index) => {
        const { box, status, id_card_type } = det;

        // Create a card for each detection
        const detectionCard = document.createElement("div");
        detectionCard.className = "col-md-4 mb-4";

        detectionCard.innerHTML = `
            <div class="card shadow-sm">
                <div class="card-body">
                    <h5 class="card-title">Detection ${index + 1}</h5>
                    <p class="card-text"><strong>Box:</strong> [${box.join(", ")}]</p>
                    <p class="card-text"><strong>Status:</strong> ${status}</p>
                    <p class="card-text"><strong>ID Card:</strong> ${id_card_type || "Not Detected"}</p>
                </div>
            </div>
        `;

        // Append the detection card to the camera grid
        cameraGrid.appendChild(detectionCard);
    });
}

// WebSocket open event
ws.onopen = () => {
    console.log("Connected to WebSocket server.");
    // Optionally, send a handshake or initial message
    ws.send(JSON.stringify({ message: "Frontend connected" }));
};

// WebSocket message event
ws.onmessage = (event) => {
    try {
        const data = JSON.parse(event.data);

        if (data.error) {
            cameraGrid.innerHTML = `<div class="col-12 text-center text-danger"><p>Error: ${data.error}</p></div>`;
        } else if (data.detections && Array.isArray(data.detections)) {
            // Call the render function with detections
            renderDetections(data.detections);
        }
    } catch (error) {
        console.error("Error parsing WebSocket message:", error);
        cameraGrid.innerHTML = `<div class="col-12 text-center text-danger"><p>Error processing server data.</p></div>`;
    }
};

// WebSocket close event
ws.onclose = () => {
    console.log("WebSocket connection closed.");
    cameraGrid.innerHTML = "<div class='col-12 text-center text-muted'><p>Disconnected from server.</p></div>";
};

// WebSocket error event
ws.onerror = (error) => {
    console.error("WebSocket error:", error);
    cameraGrid.innerHTML = "<div class='col-12 text-center text-danger'><p>An error occurred. Check the console for details.</p></div>";
};
