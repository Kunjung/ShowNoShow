var target;
var mutationRate;
var population;
var maxPopulation;
var bestPhrase;

function setup() {
  noCanvas();
  bestPhrase = select("#movieHeading");
  //var canvas = createCanvas(600, 400);
  //canvas.parent("#movieHeading");
  target = "Show or No Show";
  mutationRate = 0.002;
  maxPopulation = 670;
  population = new Population(target, mutationRate, maxPopulation);
}

function draw() {
  
  background(100);
  
  population.calcFitness();
  
  population.createMatingPool();
  
  population.reproduction();
  
  displayInfo();
  
  if (population.getBest() == target) {
    noLoop();
  }  
  
}

function displayInfo() {
  var best = population.getBest();
  bestPhrase.html(best);
  // textAlign(LEFT);
  // fill(240, 50, 240);
  // textSize(24);
  // text(best, width/2 - width/3, height/2)
}

