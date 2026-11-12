import "/src/style.css";
import { setup, vivarium } from "@wonderyard/vivarium";

/* Create */

const vi = vivarium("cross");
const { between, not } = vi.helpers;
const { TOP, RIGHT, BOTTOM, LEFT } = vi.neighbor;

const empty = vi.element("empty", "#1a1a2e");

// Every part idles until enough pulses point at it, then pulses for a step along its outputs.
// Only the pulsing state has outputs, so `pointed` counts incoming pulses, never idle wires.
// Arrows only point one way, so pulses never flow back and wires can run side by side.
type Outputs = Parameters<typeof vi.element>[3];

const part = (name: string, color: string, outputs: Outputs, count: number[]) => {
  const idle = vi.element(name, color);
  const on = vi.element(`${name} on`, "#facc15", [], outputs);
  idle.to(on).pointed(count);
  on.to(idle).pointed(not(count));
  return { idle, on };
};

const any = between(1, 4);

const parts = {
  ">": part("right", "#334155", [RIGHT], any),
  "<": part("left", "#334155", [LEFT], any),
  "^": part("up", "#334155", [TOP], any),
  v: part("down", "#334155", [BOTTOM], any),
  // Copies a pulse to two places: it closes a clock loop and feeds a gate.
  S: part("splitter", "#64748b", [BOTTOM, RIGHT], any),
  // A gate is one cell: its count is its truth table.
  O: part("or", "#22c55e", [RIGHT], any),
  A: part("and", "#3b82f6", [RIGHT], [2]),
  X: part("xor", "#a855f7", [RIGHT], [1]),
};

const logicGates = vi.create();

/* Initialize grid */

const size = 32;

// Two clocks, with periods 6 and 8, feed a gate. Their pulses meet at the gate every 24 steps.
// s is a splitter that starts with a pulse.
const circuit = (gate: "O" | "A" | "X") => [
  `>>s>>>>>>>>${gate}`.padEnd(size, ">"),
  "^<<........^",
  "...........^",
  ">>>s>>>>>>>^",
  "^<<<",
];

const rows = [
  "",
  "",
  ...circuit("O"),
  "",
  "",
  "",
  "",
  "",
  ...circuit("A"),
  "",
  "",
  "",
  "",
  "",
  ...circuit("X"),
];

const initialGrid = Array.from({ length: size }, (_, y) =>
  Array.from({ length: size }, (_, x) => {
    const char = rows[y]?.[x] ?? ".";
    if (char === "s") return parts.S.on.index;
    return char in parts ? parts[char as keyof typeof parts].idle.index : empty.index;
  }),
);

/* Run */

const canvas = document.getElementById("example-canvas") as HTMLCanvasElement;
canvas.style.imageRendering = "pixelated";

canvas.width = size;
canvas.height = size;

const { update, draw } = await setup({ canvas, automaton: logicGates, initialGrid });

let before = performance.now();

const TIME_FRAME_MS = 1000 / 5; // 5 updates per second

const loop = () => {
  const now = performance.now();
  let i = 0;
  while (now - before >= TIME_FRAME_MS && i < 10 /* to avoid freezes */) {
    update();
    before += TIME_FRAME_MS;
    i++;
  }
  draw();
  requestAnimationFrame(loop);
};
requestAnimationFrame(loop);
