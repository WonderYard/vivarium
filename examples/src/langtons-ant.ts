import "/src/style.css";
import { setup, vivarium } from "@wonderyard/vivarium";

/* Create */

const vi = vivarium("cross");

// Ground states
const white = vi.element("white", "#ffffff");
const black = vi.element("black", "#1a1a2e");

// Ant states: encode direction + underlying cell color (all red)
const antUpWhite = vi.element("ant-up-white", "#ef4444");
const antRightWhite = vi.element("ant-right-white", "#ef4444");
const antDownWhite = vi.element("ant-down-white", "#ef4444");
const antLeftWhite = vi.element("ant-left-white", "#ef4444");
const antUpBlack = vi.element("ant-up-black", "#ef4444");
const antRightBlack = vi.element("ant-right-black", "#ef4444");
const antDownBlack = vi.element("ant-down-black", "#ef4444");
const antLeftBlack = vi.element("ant-left-black", "#ef4444");

// --- Ant departure rules ---
// On white: turn clockwise, flip to black, move forward
antUpWhite.to(black);
antRightWhite.to(black);
antDownWhite.to(black);
antLeftWhite.to(black);

// On black: turn counter-clockwise, flip to white, move forward
antUpBlack.to(white);
antRightBlack.to(white);
antDownBlack.to(white);
antLeftBlack.to(white);

// --- Ant arrival rules ---
// A white cell becomes an ant if a neighbor ant is about to move into it.
//
// On white, the ant turns clockwise:
//   ant-up-white    → faces right, moves right (arrives from LEFT)
//   ant-right-white → faces down, moves down   (arrives from TOP)
//   ant-down-white  → faces left, moves left   (arrives from RIGHT)
//   ant-left-white  → faces up, moves up        (arrives from BOTTOM)
//
// On black, the ant turns counter-clockwise:
//   ant-up-black    → faces left, moves left   (arrives from RIGHT)
//   ant-right-black → faces up, moves up        (arrives from BOTTOM)
//   ant-down-black  → faces right, moves right (arrives from LEFT)
//   ant-left-black  → faces down, moves down   (arrives from TOP)

// White cell receives ant (keeps white color encoding)
white.to(antRightWhite).is(vi.neighbor.LEFT, antUpWhite);
white.to(antRightWhite).is(vi.neighbor.LEFT, antDownBlack);
white.to(antDownWhite).is(vi.neighbor.TOP, antRightWhite);
white.to(antDownWhite).is(vi.neighbor.TOP, antLeftBlack);
white.to(antLeftWhite).is(vi.neighbor.RIGHT, antDownWhite);
white.to(antLeftWhite).is(vi.neighbor.RIGHT, antUpBlack);
white.to(antUpWhite).is(vi.neighbor.BOTTOM, antLeftWhite);
white.to(antUpWhite).is(vi.neighbor.BOTTOM, antRightBlack);

// Black cell receives ant (keeps black color encoding)
black.to(antRightBlack).is(vi.neighbor.LEFT, antUpWhite);
black.to(antRightBlack).is(vi.neighbor.LEFT, antDownBlack);
black.to(antDownBlack).is(vi.neighbor.TOP, antRightWhite);
black.to(antDownBlack).is(vi.neighbor.TOP, antLeftBlack);
black.to(antLeftBlack).is(vi.neighbor.RIGHT, antDownWhite);
black.to(antLeftBlack).is(vi.neighbor.RIGHT, antUpBlack);
black.to(antUpBlack).is(vi.neighbor.BOTTOM, antLeftWhite);
black.to(antUpBlack).is(vi.neighbor.BOTTOM, antRightBlack);

const ant = vi.create();

/* Initialize grid — place ant facing up on a white cell at the center */

const size = 128;
const initialGrid: number[] = new Array(size * size).fill(white.index);
initialGrid[Math.floor(size / 2) * size + Math.floor(size / 2)] = antUpWhite.index;

/* Run */

const canvas = document.getElementById("example-canvas") as HTMLCanvasElement;
canvas.style.imageRendering = "pixelated";

canvas.width = size;
canvas.height = size;

const { update, draw } = setup({ canvas, automaton: ant, initialGrid });

const loop = async () => {
  update();
  await draw();
  requestAnimationFrame(loop);
};
requestAnimationFrame(loop);
