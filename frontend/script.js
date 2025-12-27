function detect() {
  fetch("http://127.0.0.1:5000/predict", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({features: Array(20).fill(Math.random())})
  })
  .then(res => res.json())
  .then(data => {
    document.getElementById("output").innerHTML =
      `Prediction: ${data.prediction}<br>Distance: ${data.distance}`;
  });
}
