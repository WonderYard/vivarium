import "/src/style.css";
import { setup, vivarium } from "@wonderyard/vivarium";

/* Create */

const vi = vivarium();

const empty = vi.element("empty", "#1a1a2e");
const tree = vi.element("tree", "#16a34a");
const burning = vi.element("burning", "#ef4444");

// A burning tree becomes empty.
burning.to(empty);

// A tree catches fire if any neighbor is burning.
tree.to(burning).count(burning, vi.helpers.between(1, 8));

// A tree may catch fire by lightning.
tree.to(burning).chance(1, 10000);

// Empty ground may grow a tree.
empty.to(tree).chance(1, 100);

// Otherwise, the tree and empty stay as they are.
tree.to(tree);
empty.to(empty);

const fire = vi.create();

/* Run */

const canvas = document.getElementById("example-canvas") as HTMLCanvasElement;
canvas.style.imageRendering = "pixelated";

const size = 256;
canvas.width = size;
canvas.height = size;

const { evolve } = setup({ canvas, automaton: fire });

const loop = async () => {
  await evolve();
  requestAnimationFrame(loop);
};
requestAnimationFrame(loop);
