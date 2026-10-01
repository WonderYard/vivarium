import "/src/style.css";
import { setup, vivarium } from "@wonderyard/vivarium";

/* Create */

// Rule 110 is a one-dimensional automaton: each row of the grid is one generation,
// computed from the three cells right above it (top-left, top, top-right).
const vi = vivarium("square", { wrapping: true });

const bg = vi.element("bg", "#0b1020");
const off = vi.element("off", "#1e293b");
const on = vi.element("on", "#fbbf24");

// Patterns 110 and 010 turn on.
bg.to(on).is(vi.neighbor.TOP, on).is(vi.neighbor.TOP_RIGHT, off);

// Patterns 101 and 001 turn on.
bg.to(on).is(vi.neighbor.TOP, off).is(vi.neighbor.TOP_RIGHT, on);

// Pattern 011 turns on.
bg.to(on).is(vi.neighbor.TOP_LEFT, off).is(vi.neighbor.TOP, on).is(vi.neighbor.TOP_RIGHT, on);

// Patterns 111, 100 and 000 turn off, once the generation above has been computed.
bg.to(off).is(vi.neighbor.TOP, bg).accept("none");

const rule110 = vi.create();

/* Initialize grid — a single "on" cell at the top right, everything else background */

// The pattern grows one cell to the left per generation, so starting from the
// right edge it fills the whole grid without wrapping around.
const size = 256;
const initialGrid: number[] = Array.from({ length: size * size }, (_, i) =>
  i < size ? off.index : bg.index,
);
initialGrid[size - 1] = on.index;

/* Run */

const canvas = document.getElementById("example-canvas") as HTMLCanvasElement;
canvas.style.imageRendering = "pixelated";

canvas.width = size;
canvas.height = size;

const { update, draw } = setup({ canvas, automaton: rule110, initialGrid });

const loop = async () => {
  update();
  await draw();
  requestAnimationFrame(loop);
};
requestAnimationFrame(loop);
