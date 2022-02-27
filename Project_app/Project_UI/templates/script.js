Webcam.set({
    width: 500,
    height: 400,
    image_format: "jpeg",
    jpeg_quality: 90,
    force_flash: false,
    flip_horiz: true,
    fps: 45
});
const canvas = document.getElementById("canvas");
const snap = document.getElementById("snap");
const errorMsgElement = document.querySelector('span#errorMsg');
Webcam.set("constraints", {
    optional: [{ minWidth: 600 }]
});
var video = document.querySelector("#videoElement");

function detect(){
    if (navigator.mediaDevices.getUserMedia) {
  navigator.mediaDevices.getUserMedia({ video: {width:1280, height:720} })
    .then(function (stream) {
      video.srcObject = stream;
    })
    .catch(function (err0r) {
      console.log("Something went wrong!");
    });
}
}
var context = canvas.getContext('2d');
snap.addEventListener("click", function() {
    context.drawImage(video, 0, 0, 640, 480);
});
function stop_cam(){
    if (navigator.mediaDevices.getUserMedia) {
  navigator.mediaDevices.getUserMedia({ video: true })
    .then(function (stream) {
      video.srcObject = null;
    })
    .catch(function (err0r) {
      console.log("Something went wrong!");
    });
}
}
function take_snapshot()
{
        Webcam.snap( function(data_uri) {
            $(".image-tag").val(data_uri);
            document.getElementById('results').innerHTML = '<img src="'+data_uri+'"/>';
        } );
    }