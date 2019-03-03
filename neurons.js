// JavaScript implementation of Multi-Layer Perceptron (neural network).
// author: Cédric "tuzepoito" Chartron

var requestAnimation = window.requestAnimationFrame ||
  function (callback, el) { setTimeout(callback, 1000/60.0); };

var width = 0;
var height = 0;

// figure
var figSize = 1.5; // max. coordinate in the figure
var tickLength = 0.05;

var pointsCtx;

// perceptron
var pointDimensions = 2;
var numPoints = 200;
var points = [];
var randomTheta = 0.25;

var classes = ["#ff5555", "#5555ff", "#ffff55", "#55ff55", "#ff55ff", "#55ffff"]; // class colors
var numClasses = 3;
var pointClasses = [];

// hyperparameters
var stepSize = 1;
var reg = 0.001; // regularization strength

function convertCoords (x, y) {
  return [width/2 + (x/figSize)*width/2, height/2 - (y/figSize)*height/2]
}

function drawLine (ctx, x1, y1, x2, y2) {
  ctx.beginPath();
  var newCoords1 = convertCoords(x1, y1);
  var newCoords2 = convertCoords(x2, y2);
  ctx.moveTo(newCoords1[0], newCoords1[1]);
  ctx.lineTo(newCoords2[0], newCoords2[1]);
  ctx.stroke();
}

function drawPoint (ctx, x, y) {
  ctx.beginPath();
  var newCoords = convertCoords(x, y);
  ctx.arc(newCoords[0], newCoords[1], 2, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
}

function classToColor (clas) {
  var r = parseInt(clas.substr(1, 2), 16);
  var g = parseInt(clas.substr(3, 2), 16);
  var b = parseInt(clas.substr(5, 2), 16);

  return [r, g, b];
}

function initPoints () {
  points = [];
  pointClasses = [];

  var pointsPerClass = numPoints / numClasses;

  for (var i = 0; i < numPoints; i++) {
    // compute the class
    var j = Math.floor(i / pointsPerClass);

    // generate randomized spiral
    var radius = (i % pointsPerClass) / pointsPerClass;
    // var angle = j * 2 * Math.PI / numClasses + (i / numPoints) * Math.PI + randomTheta * Math.random();
    var angle = (j + radius) * 4 + randomTheta * Math.random();
    var angle = j * 2 * Math.PI / numClasses + radius * 4 + randomTheta * Math.random();

    var newPoint = [radius * Math.cos(angle), radius * Math.sin(angle)];
    points.push(newPoint);
    pointClasses.push(j);
  }

  drawPoints();
}

function drawPoints () {
  // erase graph
  pointsCtx.clearRect(0, 0, width, height);

  // draw points on graph
  pointsCtx.fillStyle = "#FF0000";

  // draw axes
  drawLine(pointsCtx, -2, 0, 2, 0);
  drawLine(pointsCtx, 0, -2, 0, 2);

  // draw ticks
  drawLine(pointsCtx, 1.0, -tickLength, 1.0, tickLength);
  drawLine(pointsCtx, -1.0, -tickLength, -1.0, tickLength);
  drawLine(pointsCtx, -tickLength, 1.0, tickLength, 1.0);
  drawLine(pointsCtx, -tickLength, -1.0, tickLength, -1.0);

  for (var i = 0; i < numPoints; i++) {
    pointsCtx.fillStyle = classes[pointClasses[i]];
    drawPoint(pointsCtx, points[i][0], points[i][1]);
  }
}

var Perceptron = function (hiddenSize) {
  // build a multilayer perceptron.
  // here we have only one hidden layer, of size <hiddenSize>.

  var w1, b1, w2, b2;

  var self = this;

  this.init = function () {
    // initialization
    // first layer
    w1 = new Array(pointDimensions); // weights
    b1 = new Array(hiddenSize); // bias

    for (var i = 0; i < pointDimensions; i++) {
      w1[i] = new Array(hiddenSize);
      for (var j = 0; j < hiddenSize; j++) {
        w1[i][j] = 0.01 * Math.random();
      }
    }
    for (var i = 0; i < hiddenSize; i++) {
      b1[i] = 0;
    }

    // second layer
    w2 = new Array(hiddenSize); // weights
    b2 = new Array(numClasses); // bias

    for (var i = 0; i < hiddenSize; i++) {
      w2[i] = new Array(numClasses);
      for (var j = 0; j < numClasses; j++) {
        w2[i][j] = 0.01 * Math.random();
      }
    }
    for (var i = 0; i < numClasses; i++) {
      b2[i] = 0;
    }
  }

  this.computeClasses = function (points) {
    // execute the network on a set of 2 dimensional points.

    // hidden layer
    var hiddenLayer = new Array(points.length);
    for (var i = 0; i < points.length; i++) {
      hiddenLayer[i] = new Array(hiddenSize);
    }

    for (var i = 0; i < points.length; i++) {
      for (var j = 0; j < hiddenSize; j++) {
        hiddenLayer[i][j] = 0;
        for (var k = 0; k < pointDimensions; k++) {
          hiddenLayer[i][j] += w1[k][j] * points[i][k];
        }
        hiddenLayer[i][j] += b1[j];

        // reLU activation
        if (hiddenLayer[i][j] < 0) {
          hiddenLayer[i][j] = 0;
        }
      }
    }

    // final layer
    var scores = new Array(points.length);
    for (var i = 0; i < points.length; i++) {
      scores[i] = new Array(numClasses);
    }
    var expScores = new Array(points.length);
    for (var i = 0; i < points.length; i++) {
      expScores[i] = new Array(numClasses);
    }
    var probs = new Array(points.length);
    for (var i = 0; i < points.length; i++) {
      probs[i] = new Array(numClasses);
    }

    // compute scores
    for (var i = 0; i < points.length; i++) {
      var sumScores = 0;
      for (var j = 0; j < numClasses; j++) {
        scores[i][j] = 0;
        for (var k = 0; k < hiddenSize; k++) {
          scores[i][j] += w2[k][j] * hiddenLayer[i][k];
        }
        scores[i][j] += b2[j];

        expScores[i][j] = Math.exp(scores[i][j]);
        sumScores += expScores[i][j];
      }

      for (var j = 0; j < numClasses; j++) {
        probs[i][j] = expScores[i][j] / sumScores;
      }
    }

    return [scores, probs, hiddenLayer];
  }

  this.step = function (points, pointClasses) {
    var computed = self.computeClasses(points);
    var scores = computed[0], probs = computed[1], hiddenLayer = computed[2];

    var accurate = 0;

    // compute the loss and accuracy
    var loss = 0;
    // data loss
    for (var i = 0; i < points.length; i++) {
      loss -= Math.log(probs[i][pointClasses[i]]);

      // get max. Score (computed class)
      var maxScore = -Infinity;
      var maxClass = -1;

      for (var j = 0; j < numClasses; j++) {
        if (scores[i][j] > maxScore) {
          maxScore = scores[i][j];
          maxClass = j;
        }
      }

      if (maxClass == pointClasses[i]) {
        accurate += 1;
      }
    }

    // regularization loss
    for (var i = 0; i < pointDimensions; i++) {
      for (var j = 0; j < hiddenSize; j++) {
        loss += 0.5 * reg * w1[i][j] * w1[i][j];
      }
    }
    for (var i = 0; i < hiddenSize; i++) {
      for (var j = 0; j < numClasses; j++) {
        loss += 0.5 * reg * w2[i][j] * w2[i][j];
      }
    }

    // compute the gradient
    var dScores = new Array(points.length);
    for (var i = 0; i < points.length; i++) {
      dScores[i] = new Array(numClasses);
      for (var j = 0; j < numClasses; j++) {
        dScores[i][j] = probs[i][j];
        if (j == pointClasses[i]) {
          dScores[i][j] -= 1;
        }
        dScores[i][j] /= points.length;
      }
    }

    // backpropagation into final layer
    var dw2 = new Array(hiddenSize);
    var db2 = new Array(numClasses);
    for (var i = 0; i < hiddenSize; i++) {
      dw2[i] = new Array(numClasses);

      for (var j = 0; j < numClasses; j++) {
        dw2[i][j] = 0;
        for (var k = 0; k < points.length; k++) {
          dw2[i][j] += hiddenLayer[k][i] * dScores[k][j];
        }

        // regularization
        dw2[i][j] += reg * w2[i][j];
      }
    }

    for (var j = 0; j < numClasses; j++) {
      db2[j] = 0;
      for (var i = 0; i < points.length; i++) {
        db2[j] += dScores[i][j];
      }

      // update
      b2[j] -= stepSize * db2[j];
    }

    // backpropagation into hidden layer
    var dHidden = new Array(points.length);
    for (var i = 0; i < points.length; i++) {
      dHidden[i] = new Array(hiddenSize);

      for (var j = 0; j < hiddenSize; j++) {
        dHidden[i][j] = 0;
        if (hiddenLayer[i][j] > 0) { // backpropagation of reLU
          for (var k = 0; k < numClasses; k++) {
            dHidden[i][j] += dScores[i][k] * w2[j][k];
          }
        }
      }
    }

    var dw1 = new Array(pointDimensions);
    var db1 = new Array(hiddenSize);
    for (var i = 0; i < pointDimensions; i++) {
      dw1[i] = new Array(hiddenSize);

      for (var j = 0; j < hiddenSize; j++) {
        dw1[i][j] = 0;
        for (var k = 0; k < points.length; k++) {
          dw1[i][j] += dHidden[k][j] * points[k][i];
        }

        // regularization
        dw1[i][j] += reg * w1[i][j];
      }
    }

    for (var j = 0; j < hiddenSize; j++) {
      db1[j] = 0;
      for (var k = 0; k < points.length; k++) {
        db1[j] += dHidden[k][j];
      }

      // update
      b1[j] -= stepSize * db1[j];
    }

    // update weights
    for (var i = 0; i < pointDimensions; i++) {
      for (var j = 0; j < hiddenSize; j++) {
        w1[i][j] -= stepSize * dw1[i][j];
      }
    }
    for (var i = 0; i < hiddenSize; i++) {
      for (var j = 0; j < numClasses; j++) {
        w2[i][j] -= stepSize * dw2[i][j];
      }
    }

    return [loss, accurate / points.length];
  }
}

function load () {
  var backgroundCanvas = document.getElementById("backgroundCanvas");
  var pointsCanvas = document.getElementById("pointsCanvas");
  width = pointsCanvas.width;
  height = pointsCanvas.height;

  if (!pointsCanvas.getContext) {
    return;
  }

  pointsCtx = pointsCanvas.getContext("2d");

  var backgroundCtx = backgroundCanvas.getContext("2d");

  var imageData = backgroundCtx.createImageData(backgroundCanvas.width, backgroundCanvas.height);
  var data = imageData.data;

  var backgroundPoints = [];

  for (var i = 0; i < backgroundCanvas.height; i++) {
    for (var j = 0; j < backgroundCanvas.width; j++) {
      var backgroundPoint = [
        -figSize + 2 * figSize * j / backgroundCanvas.width,
        figSize - 2 * figSize * i / backgroundCanvas.height
      ];
      backgroundPoints.push(backgroundPoint);
    }
  }

  var perc;
  var iterationCount = 0;
  var stop = false;
  var loss = 0, accuracy = 0;

  var hiddenRange = document.getElementById("hiddenSize");
  var numPointsRange = document.getElementById("numPoints");
  var numClassRange = document.getElementById("numClasses");

  document.getElementById("pauseButton").addEventListener("click", function () {
    stop = !stop;
  });

  function render() {
    if (!stop) {
      for (var i = 0; i < 10; i++) {
        var stepResult = perc.step(points, pointClasses);
        loss = stepResult[0];
        accuracy = stepResult[1];
        iterationCount++;
      }

      // make background picture
      var computedBackground = perc.computeClasses(backgroundPoints);
      var backgroundScores = computedBackground[0];

      var offset = 0;
      var dataOffset = 0;

      for (var i = 0; i < backgroundCanvas.height; i++) {
        for (var j = 0; j < backgroundCanvas.width; j++) {
          // get max. Score (computed class)
          var maxScore = -Infinity;
          var maxClass = 0;

          for (var k = 0; k < numClasses; k++) {
            if (backgroundScores[offset][k] > maxScore) {
              maxScore = backgroundScores[offset][k];
              maxClass = k;
            }
          }

          var color = classToColor(classes[maxClass]);
          // fade the color a bit to white
          color[0] = Math.round(255 - (255 - color[0]) / 1.5);
          color[1] = Math.round(255 - (255 - color[1]) / 1.5);
          color[2] = Math.round(255 - (255 - color[2]) / 1.5);

          data[dataOffset] = color[0]; // red
          data[dataOffset + 1] = color[1]; // green
          data[dataOffset + 2] = color[2]; // blue
          data[dataOffset + 3] = 255; // alpha
    
          dataOffset += 4;
          offset += 1;
        }
      }

      backgroundCtx.putImageData(imageData, 0, 0);

      if (iterationCount == 10 || iterationCount % 50 == 0) {
        document.getElementById("iterations").innerText = "Iterations: " + iterationCount;
        document.getElementById("loss").innerText = "Loss: " + loss.toFixed(5);
        var percentAccuracy = 100 * accuracy;
        document.getElementById("accuracy").innerText = "Accuracy: " + percentAccuracy.toFixed(2) + "%";
      }
    }
    requestAnimation(render);
  }

  function displayHiddenSize () {
    var newHiddenSize = parseInt(hiddenRange.value, 10);
    document.getElementById("hiddenSizeText").innerText = newHiddenSize;

    return newHiddenSize;
  }

  function displayNumPoints () {
    var newNumPoints = parseInt(numPointsRange.value, 10);
    document.getElementById("numPointsText").innerText = newNumPoints;

    return newNumPoints;
  }

  function displayNumClasses () {
    var newNumClasses = parseInt(numClassRange.value, 10);
    document.getElementById("numClassesText").innerText = newNumClasses;

    return newNumClasses;
  }

  function reset() {
    var newHiddenSize = displayHiddenSize();
    numPoints = displayNumPoints();
    numClasses = displayNumClasses();

    perc = new Perceptron(newHiddenSize);
    perc.init();

    initPoints();

    iterationCount = 0;
    accuracy = 0;
    loss = 0;
  }

  hiddenRange.addEventListener("input", displayHiddenSize);
  hiddenRange.addEventListener("change", reset);
  document.getElementById("resetButton").addEventListener("click", reset);

  numPointsRange.addEventListener("input", displayNumPoints);
  numPointsRange.addEventListener("change", reset);

  numClassRange.addEventListener("input", displayNumClasses);
  numClassRange.addEventListener("change", reset);

  reset();

  requestAnimation(render);
}

if (window.addEventListener) {
  window.addEventListener('load', load);
} else if (window.attachEvent) {
  window.attachEvent('onload', load);
} else {
  window.onload = load;
}
