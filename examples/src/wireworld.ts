import "/src/style.css";
import { setup, vivarium } from "@wonderyard/vivarium";

/* Create */

const vi = vivarium();

const empty = vi.element("empty", "#1a1a2e");
const head = vi.element("head", "#3b82f6");
const tail = vi.element("tail", "#ef4444");
const conductor = vi.element("conductor", "#facc15");

// Empty stays empty.
empty.to(empty);

// Electron head becomes tail.
head.to(tail);

// Electron tail becomes conductor.
tail.to(conductor);

// Conductor becomes head if exactly 1 or 2 neighbors are heads.
conductor.to(head).count(head, 1, 2);

// Otherwise conductor stays conductor.
conductor.to(conductor);

const wireworld = vi.create();

/* Initialize grid */

// Element indices: 0=empty, 1=head, 2=tail, 3=conductor
const E = 0;
const H = 1;
const T = 2;
const C = 3;

const size = 64;
const grid = new Array<number>(size * size).fill(E);

const set = (x: number, y: number, value: number) => {
  if (x >= 0 && x < size && y >= 0 && y < size) {
    grid[y * size + x] = value;
  }
};

// Draw a horizontal wire
const hWire = (x: number, y: number, length: number) => {
  for (let i = 0; i < length; i++) set(x + i, y, C);
};

// Draw a vertical wire
const vWire = (x: number, y: number, length: number) => {
  for (let i = 0; i < length; i++) set(x, y + i, C);
};

// Clock generator (a loop that produces periodic signals)
// Small loop at top-left
const clockX = 4;
const clockY = 10;
hWire(clockX, clockY, 8);
hWire(clockX, clockY + 2, 8);
set(clockX, clockY + 1, C);
set(clockX + 7, clockY + 1, C);
// Place electron head and tail in the loop
set(clockX + 1, clockY, H);
set(clockX + 2, clockY, T);

// Wire leading from clock to the right
hWire(clockX + 7, clockY + 1, 20);

// A diode (one-way gate) built from the wire
// The diode is a small structure: conductor narrows to a point
const diodeX = clockX + 27;
const diodeY = clockY + 1;
set(diodeX, diodeY - 1, C);
set(diodeX, diodeY + 1, C);
set(diodeX + 1, diodeY, C);
hWire(diodeX + 1, diodeY, 15);

// Second clock at bottom
const clock2X = 4;
const clock2Y = 30;
hWire(clock2X, clock2Y, 6);
hWire(clock2X, clock2Y + 2, 6);
set(clock2X, clock2Y + 1, C);
set(clock2X + 5, clock2Y + 1, C);
set(clock2X + 1, clock2Y, H);
set(clock2X + 2, clock2Y, T);

// Wire from second clock
hWire(clock2X + 5, clock2Y + 1, 15);

// OR gate: two wires merging into one
const orX = 20;
const orY = clock2Y + 1;
// Both wires meet
vWire(orX, clockY + 2, orY - clockY - 2);
set(orX, clockY + 1, C);

// Output wire from merge point
hWire(orX, orY, 20);

// Another small loop pattern (signal generator)
const gen2X = 4;
const gen2Y = 45;
hWire(gen2X, gen2Y, 10);
hWire(gen2X, gen2Y + 2, 10);
set(gen2X, gen2Y + 1, C);
set(gen2X + 9, gen2Y + 1, C);
set(gen2X + 3, gen2Y, H);
set(gen2X + 4, gen2Y, T);

// Long output wire
hWire(gen2X + 9, gen2Y + 1, 40);

/* Run */

const canvas = document.getElementById("example-canvas") as HTMLCanvasElement;
canvas.style.imageRendering = "pixelated";

canvas.width = size;
canvas.height = size;

const { evolve } = setup({ canvas, automaton: wireworld, initialGrid: grid });

const loop = async () => {
  await evolve();
  requestAnimationFrame(loop);
};
requestAnimationFrame(loop);
