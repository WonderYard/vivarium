import "/src/style.css";
import { setup, vivarium } from "@wonderyard/vivarium";

/* Create */

const vi = vivarium();

const empty = vi.element("empty", "#1a1a2e");
const wire = vi.element("wire", "#facc15");
const head = vi.element("head", "#3b82f6");
const tail = vi.element("tail", "#ef4444");

// Wire becomes head if exactly 1 or 2 neighbors are heads.
wire.to(head).count(head, 1, 2);

// Electron head becomes tail.
head.to(tail);

// Electron tail becomes wire.
tail.to(wire);

const wireworld = vi.create();

/* Initialize grid */

// Element indices
const E = empty.index;
const W = wire.index;
const H = head.index;
const T = tail.index;

const size = 50;
const grid: number[][] = [];

for (let y = 0; y < size; y++) {
  grid[y] = [];
  for (let x = 0; x < size; x++) {
    grid[y].push(E);
  }
}

const set = (x: number, y: number, value: number) => {
  if (x >= 0 && x < size && y >= 0 && y < size) {
    grid[y][x] = value;
  }
};

// Draw a horizontal wire
const hWire = (x: number, y: number, length: number) => {
  for (let i = 0; i < length; i++) set(x + i, y, W);
};

// Draw a vertical wire
const vWire = (x: number, y: number, length: number) => {
  for (let i = 0; i < length; i++) set(x, y + i, W);
};

// Clock generator (a loop that produces periodic signals)
// Small loop at top-left
const clockX = 4;
const clockY = 10;
hWire(clockX, clockY, 8);
hWire(clockX, clockY + 2, 8);
set(clockX, clockY + 1, W);
set(clockX + 7, clockY + 1, W);
// Place electron head and tail in the loop
set(clockX + 1, clockY, H);
set(clockX + 2, clockY, T);

// Wire leading from clock to the right
hWire(clockX + 7, clockY + 1, 20);

// A diode (one-way gate) built from the wire
const diodeX = clockX + 27;
const diodeY = clockY + 1;
set(diodeX, diodeY - 1, W);
set(diodeX - 1, diodeY - 1, W);
set(diodeX, diodeY + 1, W);
set(diodeX - 1, diodeY + 1, W);
set(diodeX + 1, diodeY, W);
hWire(diodeX + 1, diodeY, 15);

// Another diode (reversed)
const diode2X = clockX + 36;
const diode2Y = clockY + 1;
set(diode2X, diode2Y - 1, W);
set(diode2X + 1, diode2Y - 1, W);
set(diode2X, diode2Y + 1, W);
set(diode2X, diode2Y, E);
set(diode2X + 1, diode2Y + 1, W);
set(diode2X + 1, diode2Y, W);

// Second clock at bottom
const clock2X = 4;
const clock2Y = 30;
hWire(clock2X, clock2Y, 6);
hWire(clock2X, clock2Y + 2, 6);
set(clock2X, clock2Y + 1, W);
set(clock2X + 5, clock2Y + 1, W);
set(clock2X + 1, clock2Y, H);
set(clock2X + 2, clock2Y, T);

// Wire from second clock
hWire(clock2X + 5, clock2Y + 1, 12);

// OR gate: two wires merging into one
const orX = 20;
const orY = clock2Y + 1;
// Both wires meet
vWire(orX, clockY + 2, orY - clockY - 2);
set(orX, clockY + 1, W);

/* Run */

const canvas = document.getElementById("example-canvas") as HTMLCanvasElement;
canvas.style.imageRendering = "pixelated";

canvas.width = size;
canvas.height = size;

const { evolve } = setup({ canvas, automaton: wireworld, initialGrid: grid });

let skip = false;

const loop = async () => {
  if (!skip) await evolve();
  skip = !skip;

  requestAnimationFrame(loop);
};
requestAnimationFrame(loop);
