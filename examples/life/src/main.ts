import "./style.css";
import { setup, vivarium } from "@wonderyard/vivarium";

/* Create */

const vi = vivarium();

const space = vi.element("space", "#04153b");
const alien = vi.element("alien", "#34d399");

// An alien is born if there's a family of 3 in the area.
space.to(alien).count(alien, 3);

// The alien stays if the area is neither too empty nor too crowded...
alien.to(alien).count(alien, 2, 3);

// ...otherwise the alien will leave the area forever.
alien.to(space);

const life = vi.create();

/* Run */

// Get an existing canvas (or you could create one)
const canvas = document.getElementById("life-canvas") as HTMLCanvasElement;
canvas.style.imageRendering = "pixelated";

// Set the canvas size. This will be the automaton size as well.
const size = 128;
canvas.width = size;
canvas.height = size;

// Pass the canvas and the automaton you created to the setup function:
const { evolve } = setup({ canvas, automaton: life });

// Create a simple loop that evolves the canvas:
const loop = async () => {
  await evolve();
  requestAnimationFrame(loop);
};
requestAnimationFrame(loop);
