import "/src/style.css";
import { setup, vivarium } from "@wonderyard/vivarium";

/* Create */

const vi = vivarium();

const empty = vi.element("empty", "#1a1a2e");
const tree = vi.element("tree", "#16a34a");
const fire = vi.element("burning", "#ef4444");

// Empty ground may grow a tree.
empty.to(tree).chance(1, 1000);

// A tree catches fire by lightning (chance) or if any neighbor is burning.
tree.to(fire).chance(1, 100000).count(fire).accept("any");

// A burning tree becomes empty.
fire.to(empty);

const forest = vi.create();

/* Run */

const canvas = document.getElementById("example-canvas") as HTMLCanvasElement;
canvas.style.imageRendering = "pixelated";

const size = 256;
canvas.width = size;
canvas.height = size;

const { update, draw } = setup({ canvas, automaton: forest });

const loop = async () => {
  update();
  await draw();
  requestAnimationFrame(loop);
};
requestAnimationFrame(loop);
