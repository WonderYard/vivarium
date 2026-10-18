/// <reference types="vite/client" />

import type { Automaton } from "@/automaton/types";
import type { ElementBlueprint, KindBlueprint } from "@/blueprint/ref-blueprint";
import { GpuCondition, GpuElement, GpuRule, Square, WORKGROUP_SIZE } from "@/common/constants";
import { vivarium } from "@/vivarium/vivarium";
import { compileGpuAutomaton } from "@/webgpu/compiler";
import { automatonLayout, gridLayout, compute, setup } from "@/webgpu/setup";
import tgpu, { type TgpuRoot } from "typegpu";
import * as d from "typegpu/data";
import { beforeAll, describe, expect, test } from "vitest";

// ── Debug ───────────────────────────────────────────────────────────

const DEBUG = import.meta.env.VITE_DEBUG === "1";

// ── Test utility ────────────────────────────────────────────────────

type Grid = {
  width: number;
  height: number;
  ids: number[];
};

const grid = (rows: number[][]): Grid => {
  const height = rows.length;
  const width = rows[0].length;
  return { width, height, ids: rows.flat() };
};

const toRows = (g: Grid): number[][] => {
  const rows: number[][] = [];
  for (let y = 0; y < g.height; y++) {
    rows.push(g.ids.slice(y * g.width, (y + 1) * g.width));
  }
  return rows;
};

// ── ASCII grid visualization ────────────────────────────────────────

const formatGrid = (g: Grid, symbols: string[]): string[] => {
  const rows: string[] = [];
  for (let y = 0; y < g.height; y++) {
    let row = "";
    for (let x = 0; x < g.width; x++) {
      row += symbols[g.ids[y * g.width + x]] ?? "?";
    }
    rows.push(row);
  }
  return rows;
};

const printEvolution = (before: Grid, after: Grid, automaton: Automaton, label?: string): void => {
  if (!DEBUG) return;

  const symbols = automaton.elements.map((el) => el.name[0].toUpperCase());
  const beforeRows = formatGrid(before, symbols);
  const afterRows = formatGrid(after, symbols);
  const height = Math.max(beforeRows.length, afterRows.length);

  const lines: string[] = [];
  if (label) lines.push(`  ${label}`);
  for (let y = 0; y < height; y++) {
    const left = (beforeRows[y] ?? "").padEnd(before.width);
    const arrow = y === Math.floor(height / 2) ? " → " : "   ";
    const right = afterRows[y] ?? "";
    lines.push(`  ${left}${arrow}${right}`);
  }
  console.log(lines.join("\n"));
};

/**
 * Run one GPU evolution step: write initial IDs, dispatch compute, read back IDs.
 */
async function gpuEvolve(
  root: ReturnType<typeof tgpu.initFromDevice>,
  automaton: Automaton,
  inputGrid: Grid,
): Promise<Grid> {
  const { width, height, ids } = inputGrid;
  const pipeline = root.createComputePipeline({ compute });

  const WORKGROUP_COUNT_W = Math.ceil(width / WORKGROUP_SIZE[0]);
  const WORKGROUP_COUNT_H = Math.ceil(height / WORKGROUP_SIZE[1]);

  const { gpuNeighborhood, gpuElements, gpuRules, gpuConditions, gpuKindWords, gpuKindBits } =
    compileGpuAutomaton(automaton);

  const dimensions = root.createBuffer(d.vec2u, d.vec2u(width, height)).$usage("uniform");

  const wrappingBuffer = root.createBuffer(d.u32, automaton.wrapping ? 1 : 0).$usage("uniform");

  const ids0 = root.createBuffer(d.arrayOf(d.u32, width * height)).$usage("storage");

  const ids1 = root.createBuffer(d.arrayOf(d.u32, width * height)).$usage("storage");

  // Initialize with controlled data
  ids0.write([...ids]);

  const gridGroup = root.createBindGroup(gridLayout, {
    dimensions,
    wrapping: wrappingBuffer,
    seed: root.createUniform(d.f32, 0),
    ids: ids0,
    newIds: ids1,
  });

  const rules = gpuRules.length > 0 ? gpuRules : [GpuRule()];
  const conditions = gpuConditions.length > 0 ? gpuConditions : [GpuCondition()];

  const automatonGroup = root.createBindGroup(automatonLayout, {
    neighborhood: root.createBuffer(d.u32, gpuNeighborhood).$usage("uniform"),
    elements: root
      .createBuffer(d.arrayOf(GpuElement, Math.max(gpuElements.length, 1)), gpuElements)
      .$usage("storage"),
    rules: root.createBuffer(d.arrayOf(GpuRule, rules.length), rules).$usage("storage"),
    conditions: root
      .createBuffer(d.arrayOf(GpuCondition, conditions.length), conditions)
      .$usage("storage"),
    kindWords: root.createBuffer(d.u32, gpuKindWords).$usage("uniform"),
    kindBits: root
      .createBuffer(d.arrayOf(d.u32, Math.max(gpuKindBits.length, 1)), gpuKindBits)
      .$usage("storage"),
  });

  // Dispatch compute
  pipeline
    .with(automatonGroup)
    .with(gridGroup)
    .dispatchWorkgroups(WORKGROUP_COUNT_W, WORKGROUP_COUNT_H);

  // Read back IDs from the output buffer (ids1)
  const resultIds = await ids1.read();
  const outputGrid = { width, height, ids: Array.from(resultIds) };

  printEvolution(inputGrid, outputGrid, automaton);

  return outputGrid;
}

/**
 * Run multiple GPU evolution steps sequentially.
 */
async function gpuEvolveMulti(
  root: ReturnType<typeof tgpu.initFromDevice>,
  automaton: Automaton,
  inputGrid: Grid,
  steps: number,
): Promise<Grid> {
  let current = inputGrid;
  for (let i = 0; i < steps; i++) {
    current = await gpuEvolve(root, automaton, current);
  }
  return current;
}

// ── Build helpers ───────────────────────────────────────────────────

const step = async (
  root: ReturnType<typeof tgpu.initFromDevice>,
  build: (vi: ReturnType<typeof vivarium>) => void,
  inputGrid: Grid,
  neighborhood?: "square" | "cross",
  options?: { wrapping?: boolean },
): Promise<Grid> => {
  const vi = vivarium(neighborhood, options);
  build(vi);
  return gpuEvolve(root, vi.create(), inputGrid);
};

// ── Tests ───────────────────────────────────────────────────────────

describe("GPU simulation", () => {
  let root: TgpuRoot;

  beforeAll(async () => {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      throw new Error("WebGPU adapter not available");
    }
    const device = await adapter.requestDevice();
    root = tgpu.initFromDevice({ device });
  });

  // ── Unconditional rules ─────────────────────────────────────────

  describe("unconditional rules", () => {
    test("oscillator: every cell toggles between two elements", async () => {
      const before = grid([[0, 1]]);

      const after = await step(
        root,
        (vi) => {
          const a = vi.element("a", "#ff0000");
          const b = vi.element("b", "#00ff00");
          a.to(b);
          b.to(a);
        },
        before,
      );

      expect(toRows(after)).toEqual([[1, 0]]);
    });

    test("oscillator returns to original state after two steps", async () => {
      const original = grid([[0, 1]]);

      const vi = vivarium();
      const a = vi.element("a", "#ff0000");
      const b = vi.element("b", "#00ff00");
      a.to(b);
      b.to(a);
      const automaton = vi.create();

      const step2 = await gpuEvolveMulti(root, automaton, original, 2);

      expect(toRows(step2)).toEqual(toRows(original));
    });

    test("unconditional rule to self keeps grid unchanged", async () => {
      const before = grid([[0, 1]]);

      const after = await step(
        root,
        (vi) => {
          const a = vi.element("a", "#ff0000");
          const b = vi.element("b", "#00ff00");
          a.to(a);
          b.to(b);
        },
        before,
      );

      expect(toRows(after)).toEqual(toRows(before));
    });
  });

  // ── No matching rule ──────────────────────────────────────────

  describe("no matching rule", () => {
    test("cells with no rules remain unchanged", async () => {
      const before = grid([[0]]);

      const after = await step(
        root,
        (vi) => {
          vi.element("a", "#ff0000");
        },
        before,
      );

      expect(toRows(after)).toEqual(toRows(before));
    });

    test("cells stay when no rule condition matches", async () => {
      const before = grid([[0]]);

      const after = await step(
        root,
        (vi) => {
          const a = vi.element("a", "#ff0000");
          const b = vi.element("b", "#00ff00");
          a.to(b).count(b, 5);
        },
        before,
      );

      expect(toRows(after)).toEqual(toRows(before));
    });
  });

  // ── COUNT_ELEMENT condition ──────────────────────────────────

  describe("COUNT_ELEMENT condition", () => {
    const buildLife = (vi: ReturnType<typeof vivarium>) => {
      const dead = vi.element(".", "#000000");
      const alive = vi.element("#", "#ffffff");
      dead.to(alive).count(alive, 3);
      alive.to(alive).count(alive, 2, 3);
      alive.to(dead);
    };

    test("Game of Life: birth rule — dead cell with exactly 3 alive neighbors becomes alive", async () => {
      const before = grid([
        [1, 1],
        [1, 0],
      ]);

      const after = await step(root, buildLife, before);

      expect(toRows(after)).toEqual([
        [1, 1],
        [1, 1],
      ]);
    });

    test("Game of Life: survival rule — alive cell with 2 or 3 alive neighbors survives", async () => {
      // The middle cell has 2 alive neighbors, the ends only 1.
      const before = grid([[1, 1, 1]]);

      const after = await step(root, buildLife, before);

      expect(toRows(after)).toEqual([[0, 1, 0]]);
    });

    test("Game of Life: death rule — alive cell with fewer than 2 alive neighbors dies", async () => {
      const before = grid([[1]]);

      const after = await step(root, buildLife, before);

      expect(toRows(after)).toEqual([[0]]);
    });

    test("Game of Life: death by overcrowding — alive cell with 4+ neighbors dies", async () => {
      // Both middle cells have 4 alive neighbors.
      const before = grid([
        [1, 1, 1],
        [1, 1, 0],
      ]);

      const after = await step(root, buildLife, before);

      expect(toRows(after)).toEqual([
        [1, 0, 1],
        [1, 0, 1],
      ]);
    });

    test("count of 0 matches when no neighbors of that type exist", async () => {
      const before = grid([[0]]);

      const after = await step(
        root,
        (vi) => {
          const a = vi.element("a", "#ff0000");
          const b = vi.element("b", "#00ff00");
          a.to(b).count(b, 0);
        },
        before,
      );

      expect(toRows(after)).toEqual([[1]]);
    });
  });

  // ── COUNT_KIND condition ─────────────────────────────────────

  describe("COUNT_KIND condition", () => {
    test("Wireworld: wire becomes head when exactly 1 or 2 electron-head neighbors", async () => {
      const before = grid([[2, 1]]);

      const after = await step(
        root,
        (vi) => {
          // Wireworld rules:
          // head → tail, tail → wire, wire → head if 1 or 2 signal neighbors
          const signal = vi.kind("signal");
          vi.element("empty", "#000000");
          const wire = vi.element("wire", "#ff8800");
          const head = vi.element("head", "#0088ff", [signal]);
          const tail = vi.element("tail", "#ffffff");

          head.to(tail);
          tail.to(wire);
          wire.to(head).count(signal, 1, 2);
        },
        before,
      );

      // The head becomes a tail, and the wire next to it becomes a head.
      expect(toRows(after)).toEqual([[3, 2]]);
    });

    test("kind count matches across multiple elements in the kind", async () => {
      const before = grid([[1, 0, 2]]);

      const after = await step(
        root,
        (vi) => {
          const warm = vi.kind("warm");
          const cold = vi.element("cold", "#000000");
          const fire = vi.element("fire", "#ff8800", [warm]);
          vi.element("lava", "#0088ff", [warm]);

          // cold becomes fire when exactly 2 warm neighbors (fire or lava)
          cold.to(fire).count(warm, 2);
        },
        before,
      );

      // The middle cell has 2 warm neighbors: fire on its left and lava on its right → becomes fire
      expect(toRows(after)).toEqual([[1, 1, 2]]);
    });
  });

  // ── Kind membership ───────────────────────────────────────────

  describe("kind membership", () => {
    // Elements 8 and 33 extend kind k, element 40 doesn't. When kind conditions packed
    // element indices into 32 bits, 40 wrapped around to 8 and falsely matched.
    // 41 elements and 41 kinds, so k is kind 40 and lives in the second word of each row.
    const build =
      (condition: (cell: ElementBlueprint, hit: ElementBlueprint, k: KindBlueprint) => void) =>
      (vi: ReturnType<typeof vivarium>) => {
        for (let i = 0; i < 40; i++) {
          vi.kind(`filler${i}`);
        }
        const k = vi.kind("k");
        const cell = vi.element("cell", "#ff0000");
        const hit = vi.element("hit", "#00ff00");
        for (let i = 2; i <= 40; i++) {
          vi.element(
            `e${i}`,
            `#${i.toString(16).padStart(6, "0")}`,
            i === 8 || i === 33 ? [k] : [],
          );
        }
        condition(cell, hit, k);
      };

    test("count kind matches members past index 31 and never aliases", async () => {
      // Each cell sits between one member and 40: an aliased 40 makes the count 2.
      const after = await step(
        root,
        build((cell, hit, k) => cell.to(hit).count(k, 1)),
        grid([[8, 0, 40, 0, 33]]),
      );

      expect(toRows(after)).toEqual([[8, 1, 40, 1, 33]]);
    });

    test("is kind matches members past index 31 and never aliases", async () => {
      const after = await step(
        root,
        build((cell, hit, k) => cell.to(hit).is(Square.LEFT, k)),
        grid([[8, 0, 33, 0, 40, 0]]),
      );

      expect(toRows(after)).toEqual([[8, 1, 33, 1, 40, 0]]);
    });
  });

  // ── IS_ELEMENT condition ──────────────────────────────────────

  describe("IS_ELEMENT condition", () => {
    test("cell transitions when specific neighbor matches element", async () => {
      const before = grid([[1], [0]]);

      const after = await step(
        root,
        (vi) => {
          const a = vi.element("a", "#ff0000");
          const b = vi.element("b", "#00ff00");
          a.to(b).is(Square.TOP, b);
        },
        before,
      );

      expect(toRows(after)).toEqual([[1], [1]]);
    });

    test("is condition does not match when neighbor is different", async () => {
      const before = grid([[0], [0]]);

      const after = await step(
        root,
        (vi) => {
          const a = vi.element("a", "#ff0000");
          const b = vi.element("b", "#00ff00");
          a.to(b).is(Square.TOP, b);
        },
        before,
      );

      expect(toRows(after)).toEqual(toRows(before));
    });
  });

  // ── IS_POINT condition ────────────────────────────────────────

  describe("IS_POINT condition", () => {
    test("cell transitions when two neighbor positions have the same element", async () => {
      const before = grid([[1, 0, 1]]);

      const after = await step(
        root,
        (vi) => {
          const a = vi.element("a", "#ff0000");
          const b = vi.element("b", "#00ff00");
          a.to(b).is(Square.LEFT, Square.RIGHT);
        },
        before,
      );

      expect(toRows(after)).toEqual([[1, 1, 1]]);
    });

    test("is point does not match when positions have different elements", async () => {
      const before = grid([[1, 0, 0]]);

      const after = await step(
        root,
        (vi) => {
          const a = vi.element("a", "#ff0000");
          const b = vi.element("b", "#00ff00");
          a.to(b).is(Square.LEFT, Square.RIGHT);
        },
        before,
      );

      expect(toRows(after)).toEqual(toRows(before));
    });
  });

  // ── Accept strategies ─────────────────────────────────────────

  describe("accept strategies", () => {
    test("accept ALL: all conditions must pass", async () => {
      // Every 'a' has 1 'b' neighbor, but only the bottom-left one has it on TOP.
      const before = grid([
        [1, 0],
        [0, 0],
      ]);

      const after = await step(
        root,
        (vi) => {
          const a = vi.element("a", "#ff0000");
          const b = vi.element("b", "#00ff00");
          a.to(b).count(b, 1).is(Square.TOP, b);
        },
        before,
      );

      expect(toRows(after)).toEqual([
        [1, 0],
        [1, 0],
      ]);
    });

    test("accept ANY: at least one condition must pass", async () => {
      const before = grid([[1], [0]]);

      const after = await step(
        root,
        (vi) => {
          const a = vi.element("a", "#ff0000");
          const b = vi.element("b", "#00ff00");
          a.to(b).count(b, 3).is(Square.TOP, b).accept("any");
        },
        before,
      );

      expect(toRows(after)).toEqual([[1], [1]]);
    });

    test("accept ONE: exactly one condition must pass", async () => {
      const before = grid([[1], [0]]);

      const after = await step(
        root,
        (vi) => {
          const a = vi.element("a", "#ff0000");
          const b = vi.element("b", "#00ff00");
          a.to(b).count(b, 3).is(Square.TOP, b).accept("one");
        },
        before,
      );

      expect(toRows(after)).toEqual([[1], [1]]);
    });

    test("accept ONE: fails when both conditions pass", async () => {
      // Both conditions pass for the bottom-left cell, neither does for the others.
      const before = grid([
        [1, 0],
        [0, 0],
      ]);

      const after = await step(
        root,
        (vi) => {
          const a = vi.element("a", "#ff0000");
          const b = vi.element("b", "#00ff00");
          a.to(b).is(Square.TOP, b).is(Square.RIGHT, a).accept("one");
        },
        before,
      );

      expect(toRows(after)).toEqual(toRows(before));
    });

    test("accept NONE: transitions when no conditions pass", async () => {
      const before = grid([[0]]);

      const after = await step(
        root,
        (vi) => {
          const a = vi.element("a", "#ff0000");
          const b = vi.element("b", "#00ff00");
          a.to(b).count(b, 1).accept("none");
        },
        before,
      );

      expect(toRows(after)).toEqual([[1]]);
    });

    test("accept NONE: does not transition when a condition passes", async () => {
      const before = grid([[1, 0]]);

      const after = await step(
        root,
        (vi) => {
          const a = vi.element("a", "#ff0000");
          const b = vi.element("b", "#00ff00");
          a.to(b).count(b, 1).accept("none");
        },
        before,
      );

      expect(toRows(after)).toEqual(toRows(before));
    });
  });

  // ── Wrapping (toroidal grid) ──────────────────────────────────

  describe("wrapping (toroidal grid)", () => {
    test("bottom edge wraps to top", async () => {
      // The bottom cell's BOTTOM wraps to the top cell, which is 'b'.
      const before = grid([[1], [0]]);

      const after = await step(
        root,
        (vi) => {
          const a = vi.element("a", "#ff0000");
          const b = vi.element("b", "#00ff00");
          a.to(b).is(Square.BOTTOM, b);
        },
        before,
        undefined,
        { wrapping: true },
      );

      expect(toRows(after)).toEqual([[1], [1]]);
    });

    test("right edge wraps to left", async () => {
      // The right cell's RIGHT wraps to the left cell, which is 'b'.
      const before = grid([[1, 0]]);

      const after = await step(
        root,
        (vi) => {
          const a = vi.element("a", "#ff0000");
          const b = vi.element("b", "#00ff00");
          a.to(b).is(Square.RIGHT, b);
        },
        before,
        undefined,
        { wrapping: true },
      );

      expect(toRows(after)).toEqual([[1, 1]]);
    });

    // 0 - 1 underflows to 2^32 - 1, which is already 1 modulo 2, so the top and left edges
    // need 3 cells to tell a correct wrap from a broken one.
    test("top edge wraps to bottom", async () => {
      // The top cell's TOP wraps to the bottom cell, which is 'b'.
      const before = grid([[0], [0], [1]]);

      const after = await step(
        root,
        (vi) => {
          const a = vi.element("a", "#ff0000");
          const b = vi.element("b", "#00ff00");
          a.to(b).is(Square.TOP, b);
        },
        before,
        undefined,
        { wrapping: true },
      );

      expect(toRows(after)).toEqual([[1], [0], [1]]);
    });

    test("left edge wraps to right", async () => {
      // The left cell's LEFT wraps to the right cell, which is 'b'.
      const before = grid([[0, 0, 1]]);

      const after = await step(
        root,
        (vi) => {
          const a = vi.element("a", "#ff0000");
          const b = vi.element("b", "#00ff00");
          a.to(b).is(Square.LEFT, b);
        },
        before,
        undefined,
        { wrapping: true },
      );

      expect(toRows(after)).toEqual([[1, 0, 1]]);
    });

    test("corner wraps diagonally", async () => {
      // The bottom-right cell's BOTTOM_RIGHT wraps to the top-left cell, which is 'b'.
      const before = grid([
        [1, 0],
        [0, 0],
      ]);

      const after = await step(
        root,
        (vi) => {
          const a = vi.element("a", "#ff0000");
          const b = vi.element("b", "#00ff00");
          a.to(b).is(Square.BOTTOM_RIGHT, b);
        },
        before,
        undefined,
        { wrapping: true },
      );

      expect(toRows(after)).toEqual([
        [1, 0],
        [0, 1],
      ]);
    });

    test("count of 8 matches all cells in wrapping mode", async () => {
      // On a single cell, all 8 neighbors wrap to the cell itself.
      const before = grid([[0]]);

      const after = await step(
        root,
        (vi) => {
          const a = vi.element("a", "#ff0000");
          const b = vi.element("b", "#00ff00");
          a.to(b).count(a, 8);
        },
        before,
        undefined,
        { wrapping: true },
      );

      expect(toRows(after)).toEqual([[1]]);
    });
  });

  // ── Non-wrapping (bounded grid) ──────────────────────────────

  describe("non-wrapping (bounded grid)", () => {
    test("edge cell does not see wrapped neighbors", async () => {
      // The right cell's RIGHT would wrap to 'b', but it is OOB.
      const before = grid([[1, 0]]);

      const after = await step(
        root,
        (vi) => {
          const a = vi.element("a", "#ff0000");
          const b = vi.element("b", "#00ff00");
          a.to(b).is(Square.RIGHT, b);
        },
        before,
      );

      expect(toRows(after)).toEqual(toRows(before));
    });

    test("corner cells have fewer effective neighbors", async () => {
      const before = grid([
        [0, 0, 0],
        [0, 0, 0],
      ]);

      const after = await step(
        root,
        (vi) => {
          const a = vi.element("a", "#ff0000");
          const b = vi.element("b", "#00ff00");
          // Corner cells have 3 neighbors, so count=3 should match corners only
          a.to(b).count(a, 3);
        },
        before,
      );

      // Corners have 3 neighbors, edges have 5
      expect(toRows(after)).toEqual([
        [1, 0, 1],
        [1, 0, 1],
      ]);
    });

    test("edge cells have 5 effective neighbors", async () => {
      const before = grid([
        [0, 0, 0],
        [0, 0, 0],
      ]);

      const after = await step(
        root,
        (vi) => {
          const a = vi.element("a", "#ff0000");
          const b = vi.element("b", "#00ff00");
          a.to(b).count(a, 5);
        },
        before,
      );

      // Only edge (non-corner) cells have exactly 5 neighbors
      expect(toRows(after)).toEqual([
        [0, 1, 0],
        [0, 1, 0],
      ]);
    });

    test("is condition returns false for OOB neighbor", async () => {
      // The left cell's LEFT would wrap to 'b', but it is OOB.
      const before = grid([[0, 1]]);

      const after = await step(
        root,
        (vi) => {
          const a = vi.element("a", "#ff0000");
          const b = vi.element("b", "#00ff00");
          a.to(b).is(Square.LEFT, b);
        },
        before,
      );

      expect(toRows(after)).toEqual(toRows(before));
    });
  });

  // ── Edge cases ────────────────────────────────────────────────

  describe("edge cases", () => {
    test("uniform grid with unconditional rule transitions all cells", async () => {
      const before = grid([[0]]);

      const after = await step(
        root,
        (vi) => {
          const a = vi.element("a", "#ff0000");
          const b = vi.element("b", "#00ff00");
          a.to(b);
        },
        before,
      );

      expect(toRows(after)).toEqual([[1]]);
    });

    test("count of 8 matches when all neighbors are the same element", async () => {
      // The grid is smaller than a workgroup, so invocations past the grid must not write over it.
      const before = grid([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
      ]);

      const after = await step(
        root,
        (vi) => {
          const a = vi.element("a", "#ff0000");
          const b = vi.element("b", "#00ff00");
          a.to(b).count(a, 8);
        },
        before,
      );

      // In non-wrapping mode, only interior cells (with full 8-neighbor Moore neighborhoods) match count=8.
      expect(toRows(after)).toEqual([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
      ]);
    });

    test("count of 0 does not match when all neighbors are the same element", async () => {
      const before = grid([[0, 0]]);

      const after = await step(
        root,
        (vi) => {
          const a = vi.element("a", "#ff0000");
          const b = vi.element("b", "#00ff00");
          a.to(b).count(a, 0);
        },
        before,
      );

      expect(toRows(after)).toEqual(toRows(before));
    });

    test("uniform grid: all same element with no matching rule stays unchanged", async () => {
      const before = grid([[0]]);

      const after = await step(
        root,
        (vi) => {
          const a = vi.element("a", "#ff0000");
          const b = vi.element("b", "#00ff00");
          a.to(b).count(b, 1);
        },
        before,
      );

      expect(toRows(after)).toEqual(toRows(before));
    });

    test("to point (SELF) keeps element as itself", async () => {
      const before = grid([[0, 1]]);

      const after = await step(
        root,
        (vi) => {
          const a = vi.element("a", "#ff0000");
          const b = vi.element("b", "#00ff00");
          a.to(Square.SELF);
          b.to(Square.SELF);
        },
        before,
      );

      expect(toRows(after)).toEqual(toRows(before));
    });

    test("to point copies neighbor's element", async () => {
      const before = grid([[1], [0]]);

      const after = await step(
        root,
        (vi) => {
          const a = vi.element("a", "#ff0000");
          vi.element("b", "#00ff00");
          a.to(Square.TOP);
        },
        before,
      );

      expect(toRows(after)).toEqual([[1], [1]]);
    });

    test("first matching rule wins (rules are ordered)", async () => {
      const before = grid([[0]]);

      const after = await step(
        root,
        (vi) => {
          const a = vi.element("a", "#ff0000");
          const b = vi.element("b", "#00ff00");
          const c = vi.element("c", "#0000ff");
          a.to(b);
          a.to(c);
        },
        before,
      );

      expect(toRows(after)).toEqual([[1]]);
    });
  });

  // ── Game of Life patterns ─────────────────────────────────────

  describe("Game of Life patterns", () => {
    const buildLife = (vi: ReturnType<typeof vivarium>) => {
      const dead = vi.element(".", "#000000");
      const alive = vi.element("#", "#ffffff");
      dead.to(alive).count(alive, 3);
      alive.to(alive).count(alive, 2, 3);
      alive.to(dead);
    };

    const D = 0;
    const A = 1;

    test("block (still life) remains stable", async () => {
      // Cells around a block have at most 2 alive neighbors, so the block alone is enough.
      const before = grid([
        [A, A],
        [A, A],
      ]);

      const after = await step(root, buildLife, before);

      expect(toRows(after)).toEqual(toRows(before));
    });

    test("blinker oscillates (period 2)", async () => {
      const before = grid([
        [D, D, D],
        [A, A, A],
        [D, D, D],
      ]);

      const vi = vivarium();
      buildLife(vi);
      const automaton = vi.create();

      const step1 = await gpuEvolve(root, automaton, before);

      expect(toRows(step1)).toEqual([
        [D, A, D],
        [D, A, D],
        [D, A, D],
      ]);

      const step2 = await gpuEvolve(root, automaton, step1);

      expect(toRows(step2)).toEqual(toRows(before));
    });

    test("single alive cell dies (underpopulation)", async () => {
      const before = grid([[A]]);

      const after = await step(root, buildLife, before);

      expect(toRows(after)).toEqual([[D]]);
    });

    test("two adjacent alive cells both die", async () => {
      const before = grid([[A, A]]);

      const after = await step(root, buildLife, before);

      expect(toRows(after)).toEqual([[D, D]]);
    });

    test("L-shape (4 cells) evolves correctly", async () => {
      const before = grid([
        [D, A, D],
        [D, A, D],
        [D, A, A],
      ]);

      const after = await step(root, buildLife, before);

      expect(toRows(after)).toEqual([
        [D, D, D],
        [A, A, D],
        [D, A, A],
      ]);
    });

    test("full board overcrowding: corners survive in non-wrapping mode", async () => {
      const before = grid([
        [A, A, A],
        [A, A, A],
      ]);

      const after = await step(root, buildLife, before);

      // In non-wrapping mode, corners have 3 alive neighbors → survive (2,3 rule).
      // Edge cells have 5 alive neighbors → die.
      expect(toRows(after)).toEqual([
        [A, D, A],
        [A, D, A],
      ]);
    });

    test("full board dies in wrapping mode (overcrowding)", async () => {
      // On a single cell, all 8 neighbors wrap to the cell itself.
      const before = grid([[A]]);

      const buildLifeWrapping = (vi: ReturnType<typeof vivarium>) => {
        const dead = vi.element(".", "#000000");
        const alive = vi.element("#", "#ffffff");
        dead.to(alive).count(alive, 3);
        alive.to(alive).count(alive, 2, 3);
        alive.to(dead);
      };

      const after = await step(root, buildLifeWrapping, before, undefined, { wrapping: true });

      // In wrapping mode, every cell has 8 alive neighbors → all die.
      expect(toRows(after)).toEqual([[D]]);
    });
  });

  // ── Cross neighborhood ────────────────────────────────────────

  describe("cross neighborhood", () => {
    test("cross neighborhood counts only cardinal neighbors (COUNT_ELEMENT)", async () => {
      // Center cell has 4 cardinal neighbors that are 'b' (top, left, right, bottom)
      // and 4 diagonal neighbors that are also 'b'. With cross neighborhood,
      // only the 4 cardinal ones should be counted.
      const before = grid([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
      ]);

      const after = await step(
        root,
        (vi) => {
          const a = vi.element("a", "#ff0000");
          const b = vi.element("b", "#00ff00");
          // With cross neighborhood, center has exactly 4 'b' cardinal neighbors
          a.to(b).count(b, 4);
        },
        before,
        "cross",
      );

      // Center cell (1,1) should transition because it has exactly 4 'b' cardinal neighbors
      expect(toRows(after)).toEqual([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
      ]);
    });

    test("cross neighborhood ignores diagonal neighbors (COUNT_ELEMENT)", async () => {
      // The top-left cell's only 'b' neighbor is diagonal.
      const before = grid([
        [0, 0],
        [0, 1],
      ]);

      const after = await step(
        root,
        (vi) => {
          const a = vi.element("a", "#ff0000");
          const b = vi.element("b", "#00ff00");
          // With cross neighborhood, the top-left cell has 0 'b' cardinal neighbors (diagonals don't count)
          a.to(b).count(b, 0);
        },
        before,
        "cross",
      );

      // The other 'a' cells have 'b' as a cardinal neighbor, so only the top-left one transitions
      expect(toRows(after)).toEqual([
        [1, 0],
        [0, 1],
      ]);
    });

    test("cross neighborhood counts only cardinal neighbors (COUNT_KIND)", async () => {
      // Center cell has 'cold'. Cardinal neighbors are 'fire'(1) and 'lava'(2).
      const before = grid([
        [0, 1, 0],
        [2, 0, 1],
        [0, 2, 0],
      ]);

      const after = await step(
        root,
        (vi) => {
          const warm = vi.kind("warm");
          const cold = vi.element("cold", "#000000");
          const fire = vi.element("fire", "#ff8800", [warm]);
          vi.element("lava", "#0088ff", [warm]);

          // Center has 4 cardinal warm neighbors (cross mode)
          cold.to(fire).count(warm, 4);
        },
        before,
        "cross",
      );

      // Center (1,1) should transition to fire, corners have only 2 warm neighbors
      expect(toRows(after)).toEqual([
        [0, 1, 0],
        [2, 1, 1],
        [0, 2, 0],
      ]);
    });
  });

  // ── Multi-step evolution ──────────────────────────────────────

  describe("multi-step evolution", () => {
    test("three-element cycle: a→b→c→a", async () => {
      const before = grid([[0]]);

      const vi = vivarium();
      const a = vi.element("a", "#ff0000");
      const b = vi.element("b", "#00ff00");
      const c = vi.element("c", "#0000ff");
      a.to(b);
      b.to(c);
      c.to(a);
      const automaton = vi.create();

      const step1 = await gpuEvolve(root, automaton, before);
      expect(toRows(step1)).toEqual([[1]]);

      const step2 = await gpuEvolve(root, automaton, step1);
      expect(toRows(step2)).toEqual([[2]]);

      const step3 = await gpuEvolve(root, automaton, step2);
      expect(toRows(step3)).toEqual(toRows(before));
    });

    test("Game of Life glider moves after 4 steps on large enough grid", async () => {
      const D = 0;
      const A = 1;

      const buildLife = (vi: ReturnType<typeof vivarium>) => {
        const dead = vi.element(".", "#000000");
        const alive = vi.element("#", "#ffffff");
        dead.to(alive).count(alive, 3);
        alive.to(alive).count(alive, 2, 3);
        alive.to(dead);
      };

      // Every phase of the glider over these 4 steps fits in 4×4 cells.
      const before = grid([
        [D, A, D, D],
        [D, D, A, D],
        [A, A, A, D],
        [D, D, D, D],
      ]);

      const vi = vivarium();
      buildLife(vi);
      const automaton = vi.create();

      const result = await gpuEvolveMulti(root, automaton, before, 4);

      expect(toRows(result)).toEqual([
        [D, D, D, D],
        [D, D, A, D],
        [D, D, D, A],
        [D, A, A, A],
      ]);
    });
  });

  // ── draw ─────────────────────────────────────────────────────

  describe("draw", () => {
    const RED = [255, 0, 0, 255];
    const BLUE = [0, 0, 255, 255];

    // Copies the webgpu canvas to a 2d one and returns its pixels as rows of RGBA arrays.
    const readPixels = (canvas: HTMLCanvasElement): number[][][] => {
      const copy = document.createElement("canvas");
      copy.width = canvas.width;
      copy.height = canvas.height;
      const g = copy.getContext("2d")!;
      g.drawImage(canvas, 0, 0);
      const { data } = g.getImageData(0, 0, canvas.width, canvas.height);
      return Array.from({ length: canvas.height }, (_, y) =>
        Array.from({ length: canvas.width }, (_, x) =>
          Array.from(data.slice((y * canvas.width + x) * 4, (y * canvas.width + x + 1) * 4)),
        ),
      );
    };

    test("renders each cell with its element color", async () => {
      const vi = vivarium();
      vi.element("red", "#ff0000");
      vi.element("blue", "#0000ff");
      const automaton = vi.create();

      const canvas = document.createElement("canvas");
      canvas.width = 2;
      canvas.height = 1;

      const { draw } = await setup({ canvas, automaton, initialGrid: [[0, 1]] });
      draw();

      expect(readPixels(canvas)).toEqual([[RED, BLUE]]);
    });

    test("stretches a smaller grid over the canvas", async () => {
      const vi = vivarium();
      vi.element("red", "#ff0000");
      vi.element("blue", "#0000ff");
      const automaton = vi.create();

      const canvas = document.createElement("canvas");
      canvas.width = 4;
      canvas.height = 4;
      // A checkerboard tells the axes apart, so a flipped or transposed grid fails too.
      const initialGrid = [
        [0, 1],
        [1, 0],
      ];

      const { draw } = await setup({ canvas, width: 2, height: 2, automaton, initialGrid });
      draw();

      expect(readPixels(canvas)).toEqual([
        [RED, RED, BLUE, BLUE],
        [RED, RED, BLUE, BLUE],
        [BLUE, BLUE, RED, RED],
        [BLUE, BLUE, RED, RED],
      ]);
    });
  });

  // ── readGrid and writeCell ───────────────────────────────────

  describe("readGrid and writeCell", () => {
    const createCanvas = (width: number, height: number): HTMLCanvasElement => {
      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      return canvas;
    };

    test("readGrid returns the initial grid", async () => {
      const vi = vivarium();
      vi.element("a", "#ff0000");
      vi.element("b", "#00ff00");
      const automaton = vi.create();

      const initialGrid = [[0, 1]];

      const canvas = createCanvas(2, 1);
      const { readGrid } = await setup({ canvas, automaton, initialGrid });
      const result = await readGrid();

      expect(result).toEqual(initialGrid.flat());
    });

    test("readGrid returns evolved state after evolve", async () => {
      const vi = vivarium();
      const a = vi.element("a", "#ff0000");
      const b = vi.element("b", "#00ff00");
      a.to(b);
      b.to(a);
      const automaton = vi.create();

      const canvas = createCanvas(2, 1);
      const { evolve, readGrid } = await setup({ canvas, automaton, initialGrid: [[0, 1]] });
      evolve();
      const result = await readGrid();

      expect(result).toEqual([1, 0]);
    });

    test("writeCell updates a single cell", async () => {
      const vi = vivarium();
      vi.element("a", "#ff0000");
      vi.element("b", "#00ff00");
      const automaton = vi.create();

      const canvas = createCanvas(2, 1);
      const { readGrid, writeCell } = await setup({
        canvas,
        automaton,
        initialGrid: [[0, 0]],
      });

      writeCell(1, 1);
      const result = await readGrid();

      expect(result).toEqual([0, 1]);
    });

    test("writeCell throws on out-of-bounds index", async () => {
      const vi = vivarium();
      vi.element("a", "#ff0000");
      const automaton = vi.create();

      const canvas = createCanvas(1, 1);
      const { writeCell } = await setup({ canvas, automaton, initialGrid: [[0]] });

      expect(() => writeCell(1, 0)).toThrow("out of bounds");
      expect(() => writeCell(-1, 0)).toThrow("out of bounds");
      expect(() => writeCell(0.5, 0)).toThrow("out of bounds");
      expect(() => writeCell(NaN, 0)).toThrow("out of bounds");
    });

    test("writeCell throws on invalid element index", async () => {
      const vi = vivarium();
      vi.element("a", "#ff0000");
      vi.element("b", "#00ff00");
      const automaton = vi.create();

      const canvas = createCanvas(1, 1);
      const { writeCell } = await setup({ canvas, automaton, initialGrid: [[0]] });

      expect(() => writeCell(0, 2)).toThrow("invalid");
      expect(() => writeCell(0, -1)).toThrow("invalid");
    });

    test("writeGrid overwrites the entire grid", async () => {
      const vi = vivarium();
      vi.element("a", "#ff0000");
      vi.element("b", "#00ff00");
      const automaton = vi.create();

      const canvas = createCanvas(2, 1);
      const { readGrid, writeGrid } = await setup({ canvas, automaton, initialGrid: [[0, 0]] });

      const snapshot = [0, 1];
      writeGrid(snapshot);
      const result = await readGrid();

      expect(result).toEqual(snapshot);
    });

    test("writeGrid throws on wrong length", async () => {
      const vi = vivarium();
      vi.element("a", "#ff0000");
      const automaton = vi.create();

      const canvas = createCanvas(1, 1);
      const { writeGrid } = await setup({ canvas, automaton, initialGrid: [[0]] });

      expect(() => writeGrid([0, 0])).toThrow("does not match");
    });

    test("writeGrid throws on invalid element index", async () => {
      const vi = vivarium();
      vi.element("a", "#ff0000");
      vi.element("b", "#00ff00");
      const automaton = vi.create();

      const canvas = createCanvas(1, 1);
      const { writeGrid } = await setup({ canvas, automaton, initialGrid: [[0]] });

      expect(() => writeGrid([2])).toThrow("invalid");
    });

    test("writeCellAt updates a cell by coordinates", async () => {
      const vi = vivarium();
      vi.element("a", "#ff0000");
      vi.element("b", "#00ff00");
      const automaton = vi.create();

      // 2×2 so that swapping x and y writes a different cell.
      const canvas = createCanvas(2, 2);
      const { readGrid, writeCellAt } = await setup({
        canvas,
        automaton,
        initialGrid: [
          [0, 0],
          [0, 0],
        ],
      });

      writeCellAt(0, 1, 1); // cell (0,1)
      const result = await readGrid();

      expect(result).toEqual([0, 0, 1, 0]);
    });

    test("writeCellAt throws on out-of-bounds coordinates", async () => {
      const vi = vivarium();
      vi.element("a", "#ff0000");
      const automaton = vi.create();

      const canvas = createCanvas(1, 1);
      const { writeCellAt } = await setup({ canvas, automaton, initialGrid: [[0]] });

      expect(() => writeCellAt(1, 0, 0)).toThrow("out of bounds");
      expect(() => writeCellAt(-1, 0, 0)).toThrow("out of bounds");
      expect(() => writeCellAt(0, 1, 0)).toThrow("out of bounds");
      expect(() => writeCellAt(0, -1, 0)).toThrow("out of bounds");
      expect(() => writeCellAt(0.5, 0, 0)).toThrow("out of bounds");
      expect(() => writeCellAt(0, NaN, 0)).toThrow("out of bounds");
    });
  });

  // ── CHANCE condition ─────────────────────────────────────────

  describe("CHANCE condition", () => {
    test("independent chance rules draw independent samples", async () => {
      const vi = vivarium();
      const a = vi.element("a", "#000000");
      const b = vi.element("b", "#ff0000");
      const c = vi.element("c", "#00ff00");
      a.to(b).chance(1, 2);
      a.to(c).chance(1, 2);
      const automaton = vi.create();

      // With 2500 cells, the standard deviation of each fraction is at most 0.01,
      // so the 0.05 tolerance below is 5 standard deviations.
      const size = 50;
      const canvas = document.createElement("canvas");
      canvas.width = size;
      canvas.height = size;

      const { update, readGrid } = await setup({
        canvas,
        automaton,
        initialGrid: Array.from({ length: size * size }, () => 0),
      });
      update();
      const result = await readGrid();

      // Expected: a → b with p = 1/2, otherwise a → c with p = 1/2 (so 1/4 overall).
      const fraction = (id: number) => result.filter((v) => v === id).length / result.length;
      expect(fraction(1)).toBeCloseTo(0.5, 1);
      expect(fraction(2)).toBeCloseTo(0.25, 1);
    });
  });

  describe("multiple simulations", () => {
    test("simulations stepped alternately match the same simulations run alone", async () => {
      const vi = vivarium();
      const a = vi.element("a", "#000000");
      const b = vi.element("b", "#ff0000");
      a.to(b).chance(1, 2);
      b.to(a).count(b, 2, 3).chance(1, 3);
      const automaton = vi.create();

      const sizes = [2, 3];
      const grids = sizes.map((size, n) =>
        Array.from({ length: size * size }, (_, i) => ((i * (n + 3)) % 7 < 3 ? 1 : 0)),
      );
      const seeds = [
        [0.11, 0.42, 0.73, 0.05, 0.9],
        [0.66, 0.27, 0.38, 0.81, 0.19],
      ];
      const steps = seeds[0].length;

      const create = (n: number) => {
        const canvas = document.createElement("canvas");
        canvas.width = sizes[n];
        canvas.height = sizes[n];
        return setup({ canvas, automaton, initialGrid: grids[n] });
      };

      // Each update draws its seed from Math.random, so feed the seeds in call order.
      const random = Math.random;
      const withSeeds = async <T>(queue: number[], run: () => Promise<T>): Promise<T> => {
        Math.random = () => queue.shift() ?? random();
        try {
          return await run();
        } finally {
          Math.random = random;
        }
      };

      const alone: number[][] = [];
      for (const n of [0, 1]) {
        alone.push(
          await withSeeds([...seeds[n]], async () => {
            const sim = await create(n);
            for (let i = 0; i < steps; i++) sim.update();
            return sim.readGrid();
          }),
        );
      }

      const interleaved = seeds[0].flatMap((s, i) => [s, seeds[1][i]]);
      const together = await withSeeds(interleaved, async () => {
        const sims = await Promise.all([create(0), create(1)]);
        for (let i = 0; i < steps; i++) {
          sims[0].update();
          sims[1].update();
        }
        return Promise.all(sims.map((sim) => sim.readGrid()));
      });

      expect(together).toEqual(alone);
    });

    test("simulations share the compiled shader", async () => {
      const vi = vivarium();
      vi.element("a", "#000000");
      const automaton = vi.create();
      const create = async () => {
        const canvas = document.createElement("canvas");
        canvas.width = 1;
        canvas.height = 1;
        const simulation = await setup({ canvas, automaton });
        simulation.evolve();
        simulation.destroy();
      };
      await create();

      const original = Object.getOwnPropertyDescriptor(GPUDevice.prototype, "createShaderModule")!;
      let count = 0;
      GPUDevice.prototype.createShaderModule = function (this: GPUDevice, descriptor) {
        count++;
        return (original.value as GPUDevice["createShaderModule"]).call(this, descriptor);
      };
      try {
        await create();
      } finally {
        Object.defineProperty(GPUDevice.prototype, "createShaderModule", original);
      }

      expect(count).toBe(0);
    });
  });

  describe("kind rules", () => {
    test("kind rules keep their own conditions on every extending element", async () => {
      const vi = vivarium();
      const k = vi.kind("k");
      const empty = vi.element("empty", "#000000");
      vi.element("bare", "#111111", [k]);
      const owner = vi.element("owner", "#222222", [k]);
      const marker = vi.element("marker", "#333333");
      const other = vi.element("other", "#444444");
      const first = vi.element("first", "#555555");
      const second = vi.element("second", "#666666");

      // owner's own rule never applies here; it puts owner's kind rules at different
      // positions in its rule list than bare's.
      owner.to(empty).count(other, 8);
      k.to(first).count(marker, 1);
      k.to(second).count(other, 1);
      const automaton = vi.create();

      // bare (1) and owner (2) share one marker (3) neighbor and have no other (4) neighbor,
      // so only the first kind rule's condition holds.
      const initialGrid = [[1, 3, 2]];

      const canvas = document.createElement("canvas");
      canvas.width = 3;
      canvas.height = 1;
      const { evolve, readGrid } = await setup({ canvas, automaton, initialGrid });
      evolve();
      const result = await readGrid();

      // Both step to first (5).
      expect(result).toEqual([5, 3, 5]);
    });
  });

  describe("setAutomaton", () => {
    test("sets cells of elements the new automaton doesn't have to element 0", async () => {
      const build = (names: string[]) => {
        const vi = vivarium();
        names.forEach((name, i) => vi.element(name, `#00000${i}`));
        return vi.create();
      };

      const canvas = document.createElement("canvas");
      canvas.width = 3;
      canvas.height = 1;
      const { readGrid, writeGrid, setAutomaton, toPng } = await setup({
        canvas,
        automaton: build(["a", "b", "c"]),
        initialGrid: [[0, 1, 2]],
      });

      setAutomaton({ automaton: build(["a", "b"]) });
      const grid = await readGrid();

      expect(grid).toEqual([0, 1, 0]);
      // A grid read back must stay writable and saveable.
      expect(() => writeGrid(grid)).not.toThrow();
      await expect(toPng()).resolves.toBeInstanceOf(Blob);
    });
  });

  describe("destroy", () => {
    const create = () => {
      const vi = vivarium();
      vi.element("a", "#000000");
      const automaton = vi.create();
      const canvas = document.createElement("canvas");
      canvas.width = 1;
      canvas.height = 1;
      return setup({ canvas, automaton, initialGrid: [0] });
    };

    const countDestroyedBuffers = (run: () => void): number => {
      const original = Object.getOwnPropertyDescriptor(GPUBuffer.prototype, "destroy")!;
      let count = 0;
      GPUBuffer.prototype.destroy = function (this: GPUBuffer) {
        count++;
        (original.value as (this: GPUBuffer) => void).call(this);
      };
      try {
        run();
      } finally {
        Object.defineProperty(GPUBuffer.prototype, "destroy", original);
      }
      return count;
    };

    // TypeGPU allocates GPU buffers lazily, so run a frame first to make them all exist.
    test("frees every buffer the simulation created", async () => {
      const { evolve, destroy } = await create();
      evolve();

      // 5 grid buffers (dimensions, wrapping, seed, ids ×2) + 6 automaton buffers.
      expect(countDestroyedBuffers(destroy)).toBe(11);
    });

    test("setAutomaton frees the previous automaton buffers", async () => {
      const vi = vivarium();
      vi.element("a", "#000000");
      const automaton = vi.create();
      const { update, setAutomaton, destroy } = await create();
      update();

      expect(countDestroyedBuffers(() => setAutomaton({ automaton }))).toBe(6);
      destroy();
    });

    test("update and draw do nothing after destroy", async () => {
      const { update, draw, evolve, destroy } = await create();
      destroy();
      destroy();

      expect(() => update()).not.toThrow();
      expect(() => draw()).not.toThrow();
      expect(() => evolve()).not.toThrow();
    });

    test("reads, writes and setAutomaton throw after destroy", async () => {
      const vi = vivarium();
      vi.element("a", "#000000");
      const automaton = vi.create();
      const { readGrid, writeCell, writeCellAt, writeGrid, setAutomaton, destroy } = await create();
      destroy();

      const error = /^Simulation is destroyed/;
      await expect(readGrid()).rejects.toThrow(error);
      expect(() => writeCell(0, 0)).toThrow(error);
      expect(() => writeCellAt(0, 0, 0)).toThrow(error);
      expect(() => writeGrid([0])).toThrow(error);
      expect(() => setAutomaton({ automaton })).toThrow(error);
    });

    test("a new simulation works after destroying another", async () => {
      (await create()).destroy();
      const { evolve, readGrid } = await create();
      evolve();

      expect(await readGrid()).toEqual([0]);
    });
  });

  describe("device loss", () => {
    const initialGrid = [0];

    const create = (onDeviceLost?: (info: GPUDeviceLostInfo) => void) => {
      const vi = vivarium();
      vi.element("a", "#000000");
      const canvas = document.createElement("canvas");
      canvas.width = 1;
      canvas.height = 1;
      return setup({ canvas, automaton: vi.create(), initialGrid, onDeviceLost });
    };

    // Runs a frame and catches the shared device while the simulation allocates its buffers.
    const createAndCaptureDevice = async (onDeviceLost?: (info: GPUDeviceLostInfo) => void) => {
      const original = Object.getOwnPropertyDescriptor(GPUDevice.prototype, "createBuffer")!;
      let device: GPUDevice | undefined;
      GPUDevice.prototype.createBuffer = function (this: GPUDevice, descriptor) {
        // oxlint-disable-next-line typescript/no-this-alias -- capturing the device is the point
        device = this;
        return (original.value as GPUDevice["createBuffer"]).call(this, descriptor);
      };
      try {
        const simulation = await create(onDeviceLost);
        simulation.evolve();
        return { simulation, device: device! };
      } finally {
        Object.defineProperty(GPUDevice.prototype, "createBuffer", original);
      }
    };

    test("setup after a lost device gets a new one", async () => {
      const { device } = await createAndCaptureDevice();
      device.destroy();
      await device.lost;

      const { evolve, readGrid } = await create();
      evolve();
      expect(await readGrid()).toEqual(initialGrid);
    });

    test("onDeviceLost is called and update and draw do nothing afterwards", async () => {
      const reasons: string[] = [];
      const { simulation, device } = await createAndCaptureDevice((info) => {
        reasons.push(info.reason);
      });
      device.destroy();
      await device.lost;

      expect(reasons).toEqual(["destroyed"]);
      expect(() => simulation.update()).not.toThrow();
      expect(() => simulation.evolve()).not.toThrow();
      expect(() => simulation.writeCell(0, 0)).toThrow(/^Simulation is destroyed/);
    });

    test("onDeviceLost is not called after destroy", async () => {
      let called = false;
      const { simulation, device } = await createAndCaptureDevice(() => {
        called = true;
      });
      simulation.destroy();
      device.destroy();
      await device.lost;

      expect(called).toBe(false);
    });
  });

  describe("setup validation", () => {
    const createCanvas = (width: number, height: number): HTMLCanvasElement => {
      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      return canvas;
    };

    test("accepts dimensions that are not powers of 2", async () => {
      const vi = vivarium("square", { wrapping: true });
      vi.element("a", "#ff0000");
      const automaton = vi.create();

      const canvas = createCanvas(3, 3);

      await expect(setup({ canvas, automaton })).resolves.toBeDefined();
    });

    test("throws when initialGrid length does not match dimensions", async () => {
      const vi = vivarium();
      vi.element("a", "#ff0000");
      const automaton = vi.create();

      const canvas = createCanvas(1, 1);

      await expect(setup({ canvas, automaton, initialGrid: [0, 0] })).rejects.toThrow(
        "does not match",
      );
    });

    test("throws when 2d initialGrid flattened length does not match dimensions", async () => {
      const vi = vivarium();
      vi.element("a", "#ff0000");
      const automaton = vi.create();

      const canvas = createCanvas(1, 1);

      await expect(setup({ canvas, automaton, initialGrid: [[0], [0]] })).rejects.toThrow(
        "does not match",
      );
    });

    test("initialGrid weights pick elements in proportion and skip zero weights", async () => {
      const vi = vivarium();
      vi.element("a", "#ff0000");
      vi.element("b", "#00ff00");
      vi.element("c", "#0000ff");
      const automaton = vi.create();

      const canvas = createCanvas(100, 100);
      const { readGrid } = await setup({
        canvas,
        automaton,
        initialGrid: { weights: { a: 30, c: 10 } },
      });
      const result = await readGrid();

      expect(result).not.toContain(1);
      // Expect 7500 cells of a out of 10000; 6 standard deviations is about 260.
      const aCount = result.filter((id) => id === 0).length;
      expect(Math.abs(aCount - 7500)).toBeLessThan(260);
    });

    test("throws on invalid initialGrid weights", async () => {
      const vi = vivarium();
      vi.element("a", "#ff0000");
      const automaton = vi.create();

      const canvas = createCanvas(1, 1);

      await expect(
        setup({ canvas, automaton, initialGrid: { weights: { z: 1 } } }),
      ).rejects.toThrow("no element");
      await expect(
        setup({ canvas, automaton, initialGrid: { weights: { a: -1 } } }),
      ).rejects.toThrow("invalid");
      await expect(
        setup({ canvas, automaton, initialGrid: { weights: { a: 0 } } }),
      ).rejects.toThrow("more than 0");
    });

    test("throws when grid dimensions are not positive integers", async () => {
      const vi = vivarium();
      vi.element("a", "#ff0000");
      const automaton = vi.create();

      const canvas = createCanvas(1, 1);

      await expect(setup({ canvas, automaton, width: 0 })).rejects.toThrow("positive integers");
      await expect(setup({ canvas, automaton, height: 2.5 })).rejects.toThrow("positive integers");
    });
  });
});
