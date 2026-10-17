import { setup, vivarium } from "@wonderyard/vivarium";
import { inspectOnClick } from "./inspect";
import "/src/style.css";

/* Create */

const vi = vivarium("cross");

const sea = vi.element("sea", "darkblue");
const coral = vi.element("coral", "coral");
const growth = vi.element("growth", "#ffff00");

const bubble = vi.element("bubble", "#0000ff");
// Looks the same as bubble (colors must be unique): a bubble that has decided to rise on the next step.
const rising = vi.element("rising", "#0000fe");

// The sea above a rising bubble always takes it, so the bubble is never lost or duplicated.
sea.to(bubble).is(vi.neighbor.BOTTOM, rising);

sea.to(growth).is(vi.neighbor.BOTTOM, growth).chance(50, 100).accept("all");
sea.to(growth).count(growth, 1).chance(42, 100).accept("all");

// Bubbles appear at random.
sea.to(bubble).chance(1, 5000);

// A bubble may pop on its way up. Safe, since only a rising bubble is moved by the sea above.
bubble.to(sea).chance(1, 200);

// The only coin flip: the bubble decides to rise, and the next step both cells act on it.
bubble.to(rising).chance(4, 7);

// A rising bubble leaves only if the cell above is free.
rising.to(sea).is(vi.neighbor.TOP, sea);
// It pops when it hits a coral.
rising.to(sea).is(vi.neighbor.TOP, coral);
// It also pops when it hits another bubble, resting or rising.
rising.to(sea).is(vi.neighbor.TOP, bubble).is(vi.neighbor.TOP, rising).accept("any");
// At the surface it pops: the cell above is outside the grid, so it can't even equal itself.
rising.to(sea).is(vi.neighbor.TOP, vi.neighbor.TOP).accept("none");
// Otherwise it stays put.
rising.to(bubble);

growth.to(coral);

// An isolated coral (no coral neighbors) may start growing again.
coral.to(growth).count(coral, 0).chance(1, 500);

const coralGrowth = vi.create();

inspectOnClick(coralGrowth, "Coral Growth");

/* Run */

const size = 128;

// Every world starts from the same grid: an empty sea with a single growth in the middle.
const initialGrid = Array.from({ length: size * size }, () => sea.index);
initialGrid[Math.floor(size * size - (size / 4) * 1)] = growth.index;
initialGrid[Math.floor(size * size - (size / 4) * 2)] = growth.index;
initialGrid[Math.floor(size * size - (size / 4) * 3)] = growth.index;

const cards = document.querySelectorAll<HTMLElement>("[data-world]");

for (const card of cards) {
  const canvas = card.querySelector("canvas") as HTMLCanvasElement;
  const button = card.querySelector("button") as HTMLButtonElement;
  canvas.style.imageRendering = "pixelated";
  canvas.width = size;
  canvas.height = size;

  // Each setup call gets its own buffers, step counter and random seed.
  const { update, draw } = await setup({ canvas, automaton: coralGrowth, initialGrid });

  let paused = false;
  button.onclick = () => {
    paused = !paused;
    button.textContent = paused ? "Play" : "Pause";
  };

  const loop = () => {
    if (!paused) {
      update();
      draw();
    }
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}
