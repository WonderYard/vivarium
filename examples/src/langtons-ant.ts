import { ElementBlueprint, setup, vivarium } from "@wonderyard/vivarium";
import "/src/style.css";

/* Create */

const vi = vivarium("cross");

const neighbors = [vi.neighbor.RIGHT, vi.neighbor.BOTTOM, vi.neighbor.LEFT, vi.neighbor.TOP];

// Clockwise and counter-clockwise helper functions
const cw = (i: number) => (i + 1) % neighbors.length;
const ccw = (i: number) => (i - 1 + neighbors.length) % neighbors.length;

// Ground states
const white = vi.element("white", "#99bbee");
const black = vi.element("black", "#000000");

// Kinds to generalize ant departure rules no matter the ant's orientation
const antOnWhite = vi.kind("antOnWhite");
const antOnBlack = vi.kind("antOnBlack");

const antsOnWhite: ElementBlueprint[] = [];
const antsOnBlack: ElementBlueprint[] = [];

for (let i = 0; i < neighbors.length; i++) {
  antsOnWhite.push(vi.element(`antOnWhite${i}`, "#ff000" + i, [antOnWhite]));
  antsOnBlack.push(vi.element(`antOnBlack${i}`, "#ff000" + (neighbors.length + i), [antOnBlack]));
}

// --- Ant departure rules ---
// On white: turn clockwise (90 degrees), flip to black, move forward
antOnWhite.to(black);
// On black: turn counter-clockwise (90 degrees), flip to white, move forward
antOnBlack.to(white);

// --- Ant arrival rules ---
// A cell becomes an ant if a neighbor ant is about to move into it.
for (let i = 0; i < neighbors.length; i++) {
  const from = cw(cw(i)); // origin is opposite (180 degrees) of destination
  const fromWhiteTurn = ccw(i); // inverse turn for white (-90 degrees)
  const fromBlackTurn = cw(i); // inverse turn for black (+90 degrees)

  // Ant enters white/black facing i if it comes from i-180 and it was on white facing i-90 or black facing i+90
  white.to(antsOnWhite[i]).is(neighbors[from], antsOnWhite[fromWhiteTurn]);
  white.to(antsOnWhite[i]).is(neighbors[from], antsOnBlack[fromBlackTurn]);

  black.to(antsOnBlack[i]).is(neighbors[from], antsOnWhite[fromWhiteTurn]);
  black.to(antsOnBlack[i]).is(neighbors[from], antsOnBlack[fromBlackTurn]);
}

const langtonsAnt = vi.create();

/* Initialize grid — place ant facing left on a white cell at the center */

const size = 128;
const initialGrid: number[] = new Array(size * size).fill(white.index);
const leftIndex = neighbors.indexOf(vi.neighbor.LEFT);
initialGrid[Math.floor(size / 2) * size + Math.floor(size / 2)] = antsOnWhite[leftIndex].index; // start facing left

/* Run */

const canvas = document.getElementById("example-canvas") as HTMLCanvasElement;
canvas.style.imageRendering = "pixelated";

canvas.width = size;
canvas.height = size;

const { update, draw } = setup({ canvas, automaton: langtonsAnt, initialGrid });

let before = performance.now();

const TIME_FRAME_MS = 1000 / (60 * 10); // 600 updates per second

const loop = async () => {
  const now = performance.now();
  let i = 0;
  while (now - before >= TIME_FRAME_MS && i < 10 /* to avoid freezes */) {
    update();
    before += TIME_FRAME_MS;
    i++;
  }
  await draw();
  requestAnimationFrame(loop);
};
requestAnimationFrame(loop);
