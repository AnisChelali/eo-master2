
// Creating map options
var mapOptions = {
    center: [36.077962, 4.768066],
    zoom: 10
}

// Creating a map object
var map = new L.map('map', mapOptions);

// Add OpenStreetMap layer to the map
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors'
}).addTo(map);

// Function to fetch data from the server
function fetchData() {
    fetch('/load_image')
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            // Convert base64 string back to array
            var decodedData = atob(data.array);
            var byteArray = new Uint8Array(decodedData.length);
            for (var i = 0; i < decodedData.length; i++) {
                byteArray[i] = decodedData.charCodeAt(i);
            }
            var blob = new Blob([byteArray], { type: 'image/png' });

            // Convert blob to URL
            var imageUrl = URL.createObjectURL(blob);

            // Add image overlay to the map


            var bounds = [[data.miny, data.minx], [data.maxy, data.maxx]];
            console.log(bounds);
            // Add the image overlay to the map
            addImageOverlay(imageUrl, bounds);
        })
        .catch(error => console.error('Error:', error));
}

// Function to add image overlay to the map
function addImageOverlay(imageUrl, bounds) {
    L.imageOverlay(imageUrl, bounds).addTo(map);
}

// Call the function to fetch data when the page loads
document.addEventListener('DOMContentLoaded', function () {
    fetchData();
});

// Adjust map size after all elements are loaded
map.invalidateSize(true);
