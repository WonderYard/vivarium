import "/src/style.css";
import { setup, vivarium } from "@wonderyard/vivarium";
import { inspectOnClick } from "./inspect";

/* Create */

const vi = vivarium("square", { wrapping: true });

const off = vi.element("off", "black");
const on = vi.element("on", "white");
const dying = vi.element("dying", "blue");

// A cell turns on if exactly 2 neighbors are on.
off.to(on).count(on, 2);

// An on cell always starts dying.
on.to(dying);

// A dying cell turns off.
dying.to(off);

const brain = vi.create();

inspectOnClick(brain, "Brian's Brain");

/* Run */

const canvas = document.getElementById("example-canvas") as HTMLCanvasElement;
canvas.style.imageRendering = "pixelated";

const size = 256;
canvas.width = size;
canvas.height = size;

const { update, draw } = await setup({ canvas, automaton: brain });

const loop = () => {
  update();
  draw();
  requestAnimationFrame(loop);
};
requestAnimationFrame(loop);
