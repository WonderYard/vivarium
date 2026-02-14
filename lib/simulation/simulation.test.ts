import { describe, expect, test } from "vitest";
import { Square } from "@/common/constants";
import { vivarium } from "@/vivarium/vivarium";
import { compileGpuAutomaton } from "@/webgpu/compiler";
import { type Grid, createGrid, evolve } from "./simulation";

// ── Helpers ─────────────────────────────────────────────────────────

/**
 * Build an automaton with the vivarium API, compile it, and evolve a grid
 * one step. Returns the new grid.
 */
const step = (
  build: (vi: ReturnType<typeof vivarium>) => void,
  grid: Grid,
  neighborhood?: "square" | "cross"
): Grid => {
  const vi = vivarium(neighborhood);
  build(vi);
  const gpu = compileGpuAutomaton(vi.create());
  return evolve(grid, gpu);
};

/** Shorthand: create a grid from a 2D array of element indices. */
const grid = (rows: number[][]): Grid => {
  const height = rows.length;
  const width = rows[0].length;
  const ids = rows.flat();
  return createGrid(width, height, ids);
};

/** Convert a grid to a 2D array for readable assertions. */
const toRows = (g: Grid): number[][] => {
  const rows: number[][] = [];
  for (let y = 0; y < g.height; y++) {
    rows.push(g.ids.slice(y * g.width, (y + 1) * g.width));
  }
  return rows;
};

// ── Unconditional rules ─────────────────────────────────────────────

describe("unconditional rules", () => {
  test("oscillator: every cell toggles between two elements", () => {
    const before = grid([
      [0, 1],
      [1, 0],
    ]);

    const after = step(
      (vi) => {
        const a = vi.element("a", "#ff0000");
        const b = vi.element("b", "#00ff00");
        a.to(b);
        b.to(a);
      },
      before
    );

    expect(toRows(after)).toEqual([
      [1, 0],
      [0, 1],
    ]);
  });

  test("oscillator returns to original state after two steps", () => {
    const original = grid([
      [0, 1, 0],
      [1, 0, 1],
    ]);

    const vi = vivarium();
    const a = vi.element("a", "#ff0000");
    const b = vi.element("b", "#00ff00");
    a.to(b);
    b.to(a);
    const gpu = compileGpuAutomaton(vi.create());

    const step1 = evolve(original, gpu);
    const step2 = evolve(step1, gpu);

    expect(toRows(step2)).toEqual(toRows(original));
  });

  test("unconditional rule to self keeps grid unchanged", () => {
    const before = grid([
      [0, 1],
      [1, 0],
    ]);

    const after = step(
      (vi) => {
        const a = vi.element("a", "#ff0000");
        const b = vi.element("b", "#00ff00");
        a.to(a);
        b.to(b);
      },
      before
    );

    expect(toRows(after)).toEqual(toRows(before));
  });
});

// ── No matching rule ────────────────────────────────────────────────

describe("no matching rule", () => {
  test("cells with no rules remain unchanged", () => {
    const before = grid([
      [0, 0],
      [0, 0],
    ]);

    const after = step(
      (vi) => {
        vi.element("a", "#ff0000");
      },
      before
    );

    expect(toRows(after)).toEqual(toRows(before));
  });

  test("cells stay when no rule condition matches", () => {
    // alive needs exactly 2 or 3 alive neighbors to survive.
    // In a 2x2 grid of all alive, each cell wraps and sees itself
    // through wrapping. Let's check a specific case.
    const before = grid([
      [0, 0, 0],
      [0, 1, 0],
      [0, 0, 0],
    ]);

    // Rule: element 0 transitions to 1 if it has exactly 5 neighbors of type 1.
    // Since there's only 1 cell of type 1, no cell of type 0 can match.
    const after = step(
      (vi) => {
        const a = vi.element("a", "#ff0000");
        const b = vi.element("b", "#00ff00");
        a.to(b).count(b, 5);
      },
      before
    );

    expect(toRows(after)).toEqual(toRows(before));
  });
});

// ── COUNT_ELEMENT condition ─────────────────────────────────────────

describe("COUNT_ELEMENT condition", () => {
  test("Game of Life: birth rule — dead cell with exactly 3 alive neighbors becomes alive", () => {
    // 3x3 grid, center is dead, corners and edges set so center has exactly 3 alive neighbors
    const before = grid([
      [1, 1, 0],
      [1, 0, 0],
      [0, 0, 0],
    ]);

    const after = step(
      (vi) => {
        const dead = vi.element("dead", "#000000");
        const alive = vi.element("alive", "#ffffff");
        dead.to(alive).count(alive, 3);
        alive.to(alive).count(alive, 2, 3);
        alive.to(dead);
      },
      before
    );

    // Center (1,1) has 3 alive neighbors: (0,0), (1,0), (0,1) → becomes alive
    expect(after.ids[4]).toBe(1); // center cell index = 1*3+1 = 4
  });

  test("Game of Life: survival rule — alive cell with 2 or 3 alive neighbors survives", () => {
    // Center alive with exactly 2 alive neighbors
    const before = grid([
      [0, 1, 0],
      [1, 1, 0],
      [0, 0, 0],
    ]);

    const after = step(
      (vi) => {
        const dead = vi.element("dead", "#000000");
        const alive = vi.element("alive", "#ffffff");
        dead.to(alive).count(alive, 3);
        alive.to(alive).count(alive, 2, 3);
        alive.to(dead);
      },
      before
    );

    // Center (1,1) has 2 alive neighbors: (1,0), (0,1) → survives
    expect(after.ids[4]).toBe(1);
  });

  test("Game of Life: death rule — alive cell with fewer than 2 alive neighbors dies", () => {
    // Isolated alive cell
    const before = grid([
      [0, 0, 0],
      [0, 1, 0],
      [0, 0, 0],
    ]);

    const after = step(
      (vi) => {
        const dead = vi.element("dead", "#000000");
        const alive = vi.element("alive", "#ffffff");
        dead.to(alive).count(alive, 3);
        alive.to(alive).count(alive, 2, 3);
        alive.to(dead);
      },
      before
    );

    // Center alive cell has 0 alive neighbors → dies
    expect(after.ids[4]).toBe(0);
  });

  test("Game of Life: death by overcrowding — alive cell with 4+ neighbors dies", () => {
    // Center surrounded by 4 alive cells
    const before = grid([
      [0, 1, 0],
      [1, 1, 1],
      [0, 1, 0],
    ]);

    const after = step(
      (vi) => {
        const dead = vi.element("dead", "#000000");
        const alive = vi.element("alive", "#ffffff");
        dead.to(alive).count(alive, 3);
        alive.to(alive).count(alive, 2, 3);
        alive.to(dead);
      },
      before
    );

    // Center (1,1) has 4 alive neighbors → dies
    expect(after.ids[4]).toBe(0);
  });

  test("count of 0 matches when no neighbors of that type exist", () => {
    const before = grid([
      [0, 0],
      [0, 0],
    ]);

    // Rule: a → b when count(b) == 0 (no b neighbors)
    const after = step(
      (vi) => {
        const a = vi.element("a", "#ff0000");
        const b = vi.element("b", "#00ff00");
        a.to(b).count(b, 0);
      },
      before
    );

    // All cells have 0 neighbors of type b → all transition
    expect(toRows(after)).toEqual([
      [1, 1],
      [1, 1],
    ]);
  });
});

// ── COUNT_KIND condition ────────────────────────────────────────────

describe("COUNT_KIND condition", () => {
  test("Wireworld: wire becomes head when exactly 1 or 2 head neighbors", () => {
    // 3x3: center is wire(1), one neighbor is head(2)
    const before = grid([
      [0, 2, 0],
      [0, 1, 0],
      [0, 0, 0],
    ]);

    const after = step(
      (vi) => {
        vi.element("empty", "#000000");
        const wire = vi.element("wire", "#ff8800");
        const head = vi.element("head", "#0088ff");
        vi.element("tail", "#ffffff");

        head.to(head).count(head, 8); // dummy rule to prevent fallthrough
        wire.to(head).count(head, 1, 2);
      },
      before
    );

    // Center wire(1) has 1 head neighbor → becomes head(2)
    expect(after.ids[4]).toBe(2);
  });

  test("kind count matches across multiple elements in the kind", () => {
    // Elements: empty(0), wire(1), head(2), tail(3)
    // Kind "conductor" includes wire(1), head(2), tail(3)
    // Rule: empty → wire when it has exactly 2 conductor neighbors
    const before = grid([
      [1, 0, 2],
      [0, 0, 0],
      [0, 0, 0],
    ]);

    const after = step(
      (vi) => {
        const conductor = vi.kind("conductor");
        const empty = vi.element("empty", "#000000");
        const wire = vi.element("wire", "#ff8800", [conductor]);
        vi.element("head", "#0088ff", [conductor]);
        vi.element("tail", "#ffffff", [conductor]);

        empty.to(wire).count(conductor, 2);
      },
      before
    );

    // Cell (1,0) has 2 conductor neighbors: wire(1) at (0,0) and head(2) at (2,0)
    expect(after.ids[1]).toBe(1); // becomes wire
  });
});

// ── IS_ELEMENT condition ────────────────────────────────────────────

describe("IS_ELEMENT condition", () => {
  test("cell transitions when specific neighbor matches element", () => {
    const before = grid([
      [0, 1],
      [0, 0],
    ]);

    // Rule: a → b when TOP neighbor is b
    const after = step(
      (vi) => {
        const a = vi.element("a", "#ff0000");
        const b = vi.element("b", "#00ff00");
        a.to(b).is(Square.TOP, b);
      },
      before
    );

    // Cell (1,1) has TOP=(1,0)=b → transitions to b
    expect(after.ids[3]).toBe(1);
    // Cell (0,1) has TOP=(0,0)=a → stays a
    expect(after.ids[2]).toBe(0);
  });

  test("is condition does not match when neighbor is different", () => {
    const before = grid([
      [0, 0],
      [0, 0],
    ]);

    const after = step(
      (vi) => {
        const a = vi.element("a", "#ff0000");
        const b = vi.element("b", "#00ff00");
        a.to(b).is(Square.TOP, b);
      },
      before
    );

    // No cell has b as its TOP neighbor → no transition
    expect(toRows(after)).toEqual(toRows(before));
  });
});

// ── IS_POINT condition ──────────────────────────────────────────────

describe("IS_POINT condition", () => {
  test("cell transitions when two neighbor positions have the same element", () => {
    const before = grid([
      [1, 0, 1],
      [0, 0, 0],
      [0, 0, 0],
    ]);

    // Rule: a → b when LEFT and RIGHT have the same element
    const after = step(
      (vi) => {
        const a = vi.element("a", "#ff0000");
        const b = vi.element("b", "#00ff00");
        a.to(b).is(Square.LEFT, Square.RIGHT);
      },
      before
    );

    // Cell (1,0): LEFT=(0,0)=b, RIGHT=(2,0)=b → same → transitions
    expect(after.ids[1]).toBe(1);
    // Cell (1,1): LEFT=(0,1)=a, RIGHT=(2,1)=a → same → transitions
    expect(after.ids[4]).toBe(1);
  });

  test("is point does not match when positions have different elements", () => {
    const before = grid([
      [1, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
    ]);

    // Rule: a → b when LEFT and RIGHT have the same element
    const after = step(
      (vi) => {
        const a = vi.element("a", "#ff0000");
        const b = vi.element("b", "#00ff00");
        a.to(b).is(Square.LEFT, Square.RIGHT);
      },
      before
    );

    // Cell (1,0): LEFT=(0,0)=b, RIGHT=(2,0)=a → different → no transition
    expect(after.ids[1]).toBe(0);
  });
});

// ── Accept strategies ───────────────────────────────────────────────

describe("accept strategies", () => {
  test("accept ALL: all conditions must pass", () => {
    // Two conditions: count(b, 1) AND is(TOP, b)
    const before = grid([
      [1, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
    ]);

    const after = step(
      (vi) => {
        const a = vi.element("a", "#ff0000");
        const b = vi.element("b", "#00ff00");
        // ALL (default): both conditions must pass
        a.to(b).count(b, 1).is(Square.TOP, b);
      },
      before
    );

    // Cell (0,1): count(b)=1 ✓, TOP=(0,0)=b ✓ → transitions
    expect(after.ids[3]).toBe(1);
    // Cell (1,0): count(b)=1 ✓, TOP=(1,2)(wrapping)=a ✗ → no transition
    expect(after.ids[1]).toBe(0);
  });

  test("accept ANY: at least one condition must pass", () => {
    // count(b, 3) unlikely, but is(TOP, b) likely
    const before = grid([
      [1, 0],
      [0, 0],
    ]);

    const after = step(
      (vi) => {
        const a = vi.element("a", "#ff0000");
        const b = vi.element("b", "#00ff00");
        a.to(b).count(b, 3).is(Square.TOP, b).accept("any");
      },
      before
    );

    // Cell (0,1): count(b)=3? With wrapping in 2x2 grid:
    // Neighbors of (0,1): (-1,0)→(1,0)=a, (0,0)=b, (1,0)=a, (-1,1)→(1,1)=a,
    // (1,1)=a, (-1,2)→(1,0)=a, (0,2)→(0,0)=b, (1,2)→(1,0)=a
    // b count = 2 → count(b,3) fails
    // TOP=(0,0)=b ✓ → ANY passes → transitions
    expect(after.ids[2]).toBe(1);
  });

  test("accept ONE: exactly one condition must pass", () => {
    const before = grid([
      [1, 0],
      [0, 0],
    ]);

    const after = step(
      (vi) => {
        const a = vi.element("a", "#ff0000");
        const b = vi.element("b", "#00ff00");
        // ONE: exactly one condition must pass
        a.to(b).count(b, 3).is(Square.TOP, b).accept("one");
      },
      before
    );

    // Cell (0,1): count(b,3) fails, is(TOP,b) passes → ONE(1 of 2) ✓
    expect(after.ids[2]).toBe(1);
  });

  test("accept ONE: fails when both conditions pass", () => {
    const before = grid([
      [1, 0],
      [0, 0],
    ]);

    const after = step(
      (vi) => {
        const a = vi.element("a", "#ff0000");
        const b = vi.element("b", "#00ff00");
        // ONE: fails when both pass
        // In 2x2 with wrapping: (0,1) has TOP=(0,0)=b
        // Also checking is(RIGHT, a) — RIGHT=(1,1)=a ✓
        // Both pass → ONE fails
        a.to(b).is(Square.TOP, b).is(Square.RIGHT, a).accept("one");
      },
      before
    );

    // Cell (0,1): TOP=b ✓, RIGHT(1,1)=a ✓ → both pass → ONE fails
    expect(after.ids[2]).toBe(0);
  });

  test("accept NONE: transitions when no conditions pass", () => {
    const before = grid([
      [0, 0],
      [0, 0],
    ]);

    const after = step(
      (vi) => {
        const a = vi.element("a", "#ff0000");
        const b = vi.element("b", "#00ff00");
        // NONE: transition when count(b, 1) fails
        a.to(b).count(b, 1).accept("none");
      },
      before
    );

    // All cells are a, no b neighbors → count(b,1) fails → NONE passes → transition
    expect(toRows(after)).toEqual([
      [1, 1],
      [1, 1],
    ]);
  });

  test("accept NONE: does not transition when a condition passes", () => {
    const before = grid([
      [1, 0],
      [0, 0],
    ]);

    const after = step(
      (vi) => {
        const a = vi.element("a", "#ff0000");
        const b = vi.element("b", "#00ff00");
        // In 2x2 wrapping: cell (1,0) has b neighbor at (0,0) → count passes
        a.to(b).count(b, 2).accept("none");
      },
      before
    );

    // Cell (1,0) neighbors: wrapping means b at (0,0) is a neighbor.
    // In 2x2, (1,0) has 8 "neighbors" (with wrapping): count of b.
    // Let's check: all neighbors of (1,0) in 2x2 with wrapping:
    //   (0,-1)→(0,1)=a, (1,-1)→(1,1)=a, (2,-1)→(0,1)=a
    //   (0,0)=b, (2,0)→(0,0)=b
    //   (0,1)=a, (1,1)=a, (2,1)→(0,1)=a
    // count(b) = 2 → count(b,2) passes → NONE fails → no transition
    expect(after.ids[1]).toBe(0);
  });
});

// ── Wrapping (toroidal grid) ────────────────────────────────────────

describe("wrapping (toroidal grid)", () => {
  test("top edge wraps to bottom", () => {
    const before = grid([
      [0, 0, 0],
      [0, 0, 0],
      [0, 1, 0],
    ]);

    // Rule: a → b when TOP neighbor is b
    const after = step(
      (vi) => {
        const a = vi.element("a", "#ff0000");
        const b = vi.element("b", "#00ff00");
        a.to(b).is(Square.BOTTOM, b);
      },
      before
    );

    // Cell (1,2): BOTTOM=(1,0) via wrapping = a → no transition
    expect(after.ids[7]).toBe(1); // stays b (no rule for b)
    // Cell (1,1): BOTTOM=(1,2) = b → transitions
    expect(after.ids[4]).toBe(1);
  });

  test("left edge wraps to right", () => {
    const before = grid([
      [0, 0, 1],
      [0, 0, 0],
      [0, 0, 0],
    ]);

    // Rule: a → b when RIGHT neighbor is b
    const after = step(
      (vi) => {
        const a = vi.element("a", "#ff0000");
        const b = vi.element("b", "#00ff00");
        a.to(b).is(Square.RIGHT, b);
      },
      before
    );

    // Cell (2,0) is b → stays b (no rule for b)
    // Cell (1,0) has RIGHT=(2,0)=b → transitions
    expect(after.ids[1]).toBe(1);
  });

  test("corner wraps diagonally", () => {
    const before = grid([
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 1],
    ]);

    // Rule: a → b when BOTTOM_RIGHT neighbor is b
    const after = step(
      (vi) => {
        const a = vi.element("a", "#ff0000");
        const b = vi.element("b", "#00ff00");
        a.to(b).is(Square.BOTTOM_RIGHT, b);
      },
      before
    );

    // Cell (1,1) has BOTTOM_RIGHT=(2,2)=b → transitions
    expect(after.ids[4]).toBe(1);
    // Cell (2,2) is b → stays b
    // Cell (0,0) has BOTTOM_RIGHT=(1,1)=a → no transition
    expect(after.ids[0]).toBe(0);
  });
});

// ── Edge cases ──────────────────────────────────────────────────────

describe("edge cases", () => {
  test("single cell grid with unconditional rule transitions", () => {
    const before = grid([[0]]);

    const after = step(
      (vi) => {
        const a = vi.element("a", "#ff0000");
        const b = vi.element("b", "#00ff00");
        a.to(b);
      },
      before
    );

    expect(toRows(after)).toEqual([[1]]);
  });

  test("single cell grid: all neighbors wrap to self", () => {
    // In a 1x1 grid, all 8 neighbors wrap back to the single cell
    const before = grid([[0]]);

    // count(a, 8): the single cell sees itself as all 8 neighbors
    const after = step(
      (vi) => {
        const a = vi.element("a", "#ff0000");
        const b = vi.element("b", "#00ff00");
        a.to(b).count(a, 8);
      },
      before
    );

    expect(toRows(after)).toEqual([[1]]);
  });

  test("single cell grid: count excludes self", () => {
    // Self is not included in neighbor count (it's not one of the 8 surrounding positions…
    // but in 1x1 grid, all surrounding positions wrap back to self!)
    // So count IS 8 for a 1x1 grid
    const before = grid([[0]]);

    // count(a, 0) should NOT match in 1x1 grid since all neighbors are self
    const after = step(
      (vi) => {
        const a = vi.element("a", "#ff0000");
        const b = vi.element("b", "#00ff00");
        a.to(b).count(a, 0);
      },
      before
    );

    // count(a) = 8 (all neighbors are self) → count(a,0) fails → stays a
    expect(toRows(after)).toEqual([[0]]);
  });

  test("uniform grid: all same element with no matching rule stays unchanged", () => {
    const before = grid([
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
    ]);

    const after = step(
      (vi) => {
        const a = vi.element("a", "#ff0000");
        const b = vi.element("b", "#00ff00");
        // Rule needs exactly 1 b neighbor — impossible in uniform grid
        a.to(b).count(b, 1);
      },
      before
    );

    expect(toRows(after)).toEqual(toRows(before));
  });

  test("uniform grid: all cells transition when unconditional", () => {
    const before = grid([
      [0, 0],
      [0, 0],
    ]);

    const after = step(
      (vi) => {
        const a = vi.element("a", "#ff0000");
        const b = vi.element("b", "#00ff00");
        a.to(b);
      },
      before
    );

    expect(toRows(after)).toEqual([
      [1, 1],
      [1, 1],
    ]);
  });

  test("to point (SELF) keeps element as itself", () => {
    const before = grid([
      [0, 1],
      [1, 0],
    ]);

    const after = step(
      (vi) => {
        const a = vi.element("a", "#ff0000");
        const b = vi.element("b", "#00ff00");
        a.to(Square.SELF);
        b.to(Square.SELF);
      },
      before
    );

    expect(toRows(after)).toEqual(toRows(before));
  });

  test("to point copies neighbor's element", () => {
    const before = grid([
      [0, 1],
      [0, 0],
    ]);

    // Rule: a → TOP (take whatever element is at TOP position)
    const after = step(
      (vi) => {
        const a = vi.element("a", "#ff0000");
        vi.element("b", "#00ff00");
        a.to(Square.TOP);
      },
      before
    );

    // Cell (0,0): TOP=(0,-1) wraps to (0,1)=a → stays a
    expect(after.ids[0]).toBe(0);
    // Cell (1,0): TOP=(1,-1) wraps to (1,1)=a → stays a
    expect(after.ids[1]).toBe(1); // b has no rule, stays b
    // Cell (0,1): TOP=(0,0)=a → becomes a (already a)
    expect(after.ids[2]).toBe(0);
    // Cell (1,1): TOP=(1,0)=b → becomes b
    expect(after.ids[3]).toBe(1);
  });

  test("first matching rule wins (rules are ordered)", () => {
    const before = grid([[0]]);

    const after = step(
      (vi) => {
        const a = vi.element("a", "#ff0000");
        const b = vi.element("b", "#00ff00");
        const c = vi.element("c", "#0000ff");
        // First rule: a → b (unconditional)
        a.to(b);
        // Second rule: a → c (unconditional, but never reached)
        a.to(c);
      },
      before
    );

    expect(toRows(after)).toEqual([[1]]); // becomes b, not c
  });
});

// ── Game of Life characteristic patterns ────────────────────────────

describe("Game of Life patterns", () => {
  /** Standard Game of Life rules builder */
  const buildLife = (vi: ReturnType<typeof vivarium>) => {
    const dead = vi.element("dead", "#000000");
    const alive = vi.element("alive", "#ffffff");
    dead.to(alive).count(alive, 3);
    alive.to(alive).count(alive, 2, 3);
    alive.to(dead);
  };

  const D = 0;
  const A = 1;

  test("block (still life) remains stable", () => {
    // 4x4 grid with a 2x2 block in the center
    const before = grid([
      [D, D, D, D],
      [D, A, A, D],
      [D, A, A, D],
      [D, D, D, D],
    ]);

    const after = step(buildLife, before);

    expect(toRows(after)).toEqual(toRows(before));
  });

  test("blinker oscillates (period 2)", () => {
    // 5x5 grid with horizontal blinker
    const before = grid([
      [D, D, D, D, D],
      [D, D, D, D, D],
      [D, A, A, A, D],
      [D, D, D, D, D],
      [D, D, D, D, D],
    ]);

    const vi = vivarium();
    buildLife(vi);
    const gpu = compileGpuAutomaton(vi.create());

    const step1 = evolve(before, gpu);

    // Should become vertical blinker
    expect(toRows(step1)).toEqual([
      [D, D, D, D, D],
      [D, D, A, D, D],
      [D, D, A, D, D],
      [D, D, A, D, D],
      [D, D, D, D, D],
    ]);

    const step2 = evolve(step1, gpu);

    // Should return to horizontal blinker
    expect(toRows(step2)).toEqual(toRows(before));
  });

  test("single alive cell dies (underpopulation)", () => {
    const before = grid([
      [D, D, D, D, D],
      [D, D, D, D, D],
      [D, D, A, D, D],
      [D, D, D, D, D],
      [D, D, D, D, D],
    ]);

    const after = step(buildLife, before);

    // Lone cell has 0 neighbors → dies
    expect(after.ids[12]).toBe(D); // center = 2*5+2 = 12
  });

  test("two adjacent alive cells both die", () => {
    const before = grid([
      [D, D, D, D, D],
      [D, D, D, D, D],
      [D, D, A, A, D],
      [D, D, D, D, D],
      [D, D, D, D, D],
    ]);

    const after = step(buildLife, before);

    // Each has only 1 alive neighbor → both die
    expect(after.ids[12]).toBe(D);
    expect(after.ids[13]).toBe(D);
  });

  test("L-shape (3 cells) evolves correctly", () => {
    const before = grid([
      [D, D, D, D, D],
      [D, D, A, D, D],
      [D, D, A, D, D],
      [D, D, A, A, D],
      [D, D, D, D, D],
    ]);

    const after = step(buildLife, before);

    // Cell (2,1): 1 alive neighbor → dies
    expect(after.ids[1 * 5 + 2]).toBe(D);
    // Cell (2,2): 3 alive neighbors → survives
    expect(after.ids[2 * 5 + 2]).toBe(A);
    // Cell (2,3): 2 alive neighbors → survives
    expect(after.ids[3 * 5 + 2]).toBe(A);
    // Cell (3,3): 2 alive neighbors → survives
    expect(after.ids[3 * 5 + 3]).toBe(A);
    // Cell (1,2) was dead: 3 alive neighbors → birth
    expect(after.ids[2 * 5 + 1]).toBe(A);
  });

  test("full board dies (overcrowding except edges, but wrapping makes all corners overcrowded too)", () => {
    // 3x3 all alive — every cell has 8 alive neighbors → all die
    const before = grid([
      [A, A, A],
      [A, A, A],
      [A, A, A],
    ]);

    const after = step(buildLife, before);

    // Every cell has 8 alive neighbors → dies
    expect(toRows(after)).toEqual([
      [D, D, D],
      [D, D, D],
      [D, D, D],
    ]);
  });
});

// ── Multi-step evolution ────────────────────────────────────────────

describe("multi-step evolution", () => {
  test("three-element cycle: a→b→c→a", () => {
    const before = grid([
      [0, 1, 2],
    ]);

    const vi = vivarium();
    const a = vi.element("a", "#ff0000");
    const b = vi.element("b", "#00ff00");
    const c = vi.element("c", "#0000ff");
    a.to(b);
    b.to(c);
    c.to(a);
    const gpu = compileGpuAutomaton(vi.create());

    const step1 = evolve(before, gpu);
    expect(toRows(step1)).toEqual([[1, 2, 0]]);

    const step2 = evolve(step1, gpu);
    expect(toRows(step2)).toEqual([[2, 0, 1]]);

    const step3 = evolve(step2, gpu);
    expect(toRows(step3)).toEqual([[0, 1, 2]]); // back to original
  });

  test("Game of Life glider moves after 4 steps on large enough grid", () => {
    const D = 0;
    const A = 1;

    const buildLife = (vi: ReturnType<typeof vivarium>) => {
      const dead = vi.element("dead", "#000000");
      const alive = vi.element("alive", "#ffffff");
      dead.to(alive).count(alive, 3);
      alive.to(alive).count(alive, 2, 3);
      alive.to(dead);
    };

    // Glider on a 6x6 grid (big enough to avoid wrapping interference for 4 steps)
    const before = grid([
      [D, A, D, D, D, D],
      [D, D, A, D, D, D],
      [A, A, A, D, D, D],
      [D, D, D, D, D, D],
      [D, D, D, D, D, D],
      [D, D, D, D, D, D],
    ]);

    const vi = vivarium();
    buildLife(vi);
    const gpu = compileGpuAutomaton(vi.create());

    let current = before;
    for (let i = 0; i < 4; i++) {
      current = evolve(current, gpu);
    }

    // After 4 steps, glider moves 1 right and 1 down
    expect(toRows(current)).toEqual([
      [D, D, D, D, D, D],
      [D, D, A, D, D, D],
      [D, D, D, A, D, D],
      [D, A, A, A, D, D],
      [D, D, D, D, D, D],
      [D, D, D, D, D, D],
    ]);
  });
});
