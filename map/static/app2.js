// Global variable to store the image overlay
var imagesOverlay = {};
var map = L.map('map').setView([36.385913, 4.482422], 5);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(map);

var popup = L.popup();


// Function to initialize the map with the given geographic transformation
function zoom(map, latitude, longitude) {
    // Convert latitude and longitude to Leaflet LatLng object
    var latLng = L.latLng(latitude, longitude);

    // Set map view to the specified coordinates with a certain zoom level
    map.setView(latLng, 12);

    // Add OpenStreetMap tile layer
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(map);

    return map;
}

// Function to overlay the image on the map
function overlayImage(map, city, imageType, imagePath, tl_latlong, tr_latlong, bl_latlong, br_latlong) {
    // var latLngBounds = L.latLngBounds([top_left_lat, bottom_right_long]);
    console.log("latLngBounds", tl_latlong, tr_latlong, bl_latlong, br_latlong);

    // Remove existing image overlay if it exists
    if (imagesOverlay[city]) {
        map.removeLayer(imagesOverlay[city]);
    }

    // opacity = imageType === 'landcover'? 0.5 : 0.7 ;
    // console.log('opacity : ', opacity , opa)
    // L.rectangle(latLngBounds, { color: "#ff7800", weight: 1 }).addTo(map);
    var ImageOverlay = L.imageOverlay.rotated(imagePath, tl_latlong, tr_latlong, bl_latlong, {
        opacity: 1,
        interactive: true,
    }).addTo(map);

    imagesOverlay[city] = ImageOverlay;
    document.getElementById("varianceRange").style.display = "block"
}

function updateValue(val) {
    if(document.getElementsByClassName('leaflet-image-layer')[1]){
        document.getElementsByClassName('leaflet-image-layer')[1].style.opacity = val
    }else {
        console.log('there is no Overlayimage')
    }
}


// Function to fetch geographic information from the server
function zoom_over_city(map, city) {
    $.ajax({
        url: '/zoom/' + city,
        method: 'GET',
        success: function (response) {
            if (!response.error) {
                zoom(map, response.lattitude, response.longitude);
                // overlayImage(map, response.image, response.lattitude, response.longitude);
            } else {
                console.error(response.error);
            }
        },
        error: function (_xhr, status, error) {
            console.error('Error:', error);
        }
    });
}


function get_image(map, city, imageType, tl_latlong, tr_latlong, bl_latlong, br_latlong) {
    $.ajax({
        url: '/send_image/' + city + '/' + imageType,
        method: 'GET',
        xhrFields: {
            responseType: 'blob' // Set the response type to blob
        },
        success: function (response) {
            if (!response.error) {
                var reader = new FileReader(); // Create a FileReader object
                reader.onload = function () {
                    var imageUrl = reader.result; // Get the base64 encoded image URL
                    overlayImage(map, city, imageType, imageUrl, tl_latlong, tr_latlong, bl_latlong, br_latlong);
                };
                reader.readAsDataURL(response);
            } else {
                console.log("Response error 2");
                console.error(response.error);
            }
        },
        error: function (xhr, status, error) {
            console.log("Response error 1");
            console.error('Error:', error);
        }
    });
}


function overlay_image(map, city, imageType) {
    $.ajax({
        url: '/overlay_image/' + city + '/' + imageType,
        method: 'GET',
        success: function (response) {
            if (!response.error) {
                get_image(map, city, imageType, response.tl_latlong, response.tr_latlong, response.bl_latlong, response.br_latlong);
            } else {
                console.log("Response error 2");
                console.error(response.error);
            }
        },
        error: function (xhr, status, error) {
            console.log("Response error 1");

            console.error('Error:', error);
        }
    });
}

// Function to fetch geographic information from the server
// function overlay_landcover_city(map, city, ) {
//     $.ajax({
//         url: '/overlay_image/' + city,
//         method: 'GET',
//         success: function (response) {
//             if (!response.error) {
//                 get_image(map, city, response.tl_latlong, response.tr_latlong, response.bl_latlong, response.br_latlong);
//             } else {
//                 console.log("Response error 2");
//                 console.error(response.error);
//             }
//         },
//         error: function (xhr, status, error) {
//             console.log("Response error 1");

//             console.error('Error:', error);
//         }
//     });
// }




function onMapClick(e) {
    popup
        .setLatLng(e.latlng)
        .setContent("You clicked the map at " + e.latlng.toString())
        .openOn(map);
}

map.on('click', onMapClick);



// Example: Fetching geographic information and overlaying the image
// city = "BordjBouArreridj";
// zoom_over_city(map, city);


// Function to get the name of the selected city
function getSelectedCityName() {
    // Get all radio buttons with class ".cities input[type="radio"]"
    var radioButtons = document.querySelectorAll('.cities input[type="radio"]');

    // Loop through each radio button
    for (var i = 0; i < radioButtons.length; i++) {
        // Check if the radio button is checked
        if (radioButtons[i].checked) {
            // Get the label text associated with the checked radio button
            var label = radioButtons[i].nextElementSibling.textContent.trim();
            return label; // Return the city name
        }
    }
    return null; // Return null if no radio button is checked
}

// Function to get the name of the checked radio button in the "cities" div
function setListnerCheckedCityName() {
    // Get all radio buttons within the "cities" div
    var radioButtons = document.querySelectorAll('.cities input[type="radio"]');

    // Loop through each radio button
    for (var i = 0; i < radioButtons.length; i++) {
        // Add event listener to each radio button
        radioButtons[i].addEventListener('click', function () {
            // Check if the radio button is checked
            if (this.checked) {
                // Output the name of the checked city radio button
                console.log(this.getAttribute('id'));
                city = this.getAttribute('id');
                zoom_over_city(map, city);
            }
        });
    }
}

// Function to get the name of the checked radio button in the "cities" div
function getCheckedOverlayData(map, city) {
    // Get all radio buttons within the "cities" div
    var radioButtons = document.querySelectorAll('.overlay_data input[type="checkbox"]');

    // Loop through each radio button
    for (var i = 0; i < radioButtons.length; i++) {
        // Add event listener to each radio button
        radioButtons[i].addEventListener('click', function () {

            var city = getSelectedCityName();

            // Check if the radio button is checked
            if (this.checked) {
                // Output the name of the checked city radio button
                console.log("checked", this.getAttribute('id'));
                overlay_data = this.getAttribute('id');
                console.log("city ", city);
                overlay_image(map, city, overlay_data);
            }
            else {
                console.log("Unchecked", this.getAttribute('id'));
                // Check if the overlayed image exists, then remove it
                console.log("Removing : ", imagesOverlay[city]);

                if (imagesOverlay[city]) {
                    console.log("Removing...");
                    map.removeLayer(imagesOverlay[city]);
                    delete imagesOverlay[city];
                }

                document.getElementById("varianceRange").style.display = "none"
            }
        });
    }
}



// Example usage
setListnerCheckedCityName();
getCheckedOverlayData(map);