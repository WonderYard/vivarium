/// <reference types="vite/client" />

import type { Automaton } from "@/automaton/types";
import { GpuCondition, GpuElement, GpuRule, Square, WORKGROUP_SIZE } from "@/common/constants";
import { vivarium } from "@/vivarium/vivarium";
import { compileGpuAutomaton } from "@/webgpu/compiler";
import { automatonLayout, gridLayout, compute, setSeed, setup } from "@/webgpu/setup";
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

  setSeed(root.createUniform(d.f32, 0));

  const WORKGROUP_COUNT_W = Math.ceil(width / WORKGROUP_SIZE[0]);
  const WORKGROUP_COUNT_H = Math.ceil(height / WORKGROUP_SIZE[1]);

  const { gpuNeighborhood, gpuElements, gpuRules, gpuConditions } = compileGpuAutomaton(automaton);

  const palette = gpuElements.map((el) => el.color);

  const dimensions = root.createBuffer(d.vec2u, d.vec2u(width, height)).$usage("uniform");

  const wrappingBuffer = root
    .createBuffer(d.u32, automaton.wrapping ? 1 : 0)
    .$usage("uniform");

  const colors0 = root.createBuffer(d.arrayOf(d.u32, width * height)).$usage("storage");

  const colors1 = root.createBuffer(d.arrayOf(d.u32, width * height)).$usage("storage");

  const ids0 = root.createBuffer(d.arrayOf(d.u32, width * height)).$usage("storage");

  const ids1 = root.createBuffer(d.arrayOf(d.u32, width * height)).$usage("storage");

  // Initialize with controlled data
  const initialColors = ids.map((id) => palette[id] ?? 0);
  colors0.write(initialColors);
  ids0.write([...ids]);

  const gridGroup = root.createBindGroup(gridLayout, {
    dimensions,
    wrapping: wrappingBuffer,
    colors: colors0,
    newColors: colors1,
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
      const before = grid([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 0],
      ]);

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

      expect(toRows(after)).toEqual([
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
      ]);
    });

    test("oscillator returns to original state after two steps", async () => {
      const original = grid([
        [0, 1, 0],
        [1, 0, 1],
        [0, 0, 0],
      ]);

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
      const before = grid([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 0],
      ]);

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
      const before = grid([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
      ]);

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
      const before = grid([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
      ]);

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
    test("Game of Life: birth rule — dead cell with exactly 3 alive neighbors becomes alive", async () => {
      const before = grid([
        [1, 1, 0],
        [1, 0, 0],
        [0, 0, 0],
      ]);

      const after = await step(
        root,
        (vi) => {
          const dead = vi.element(".", "#000000");
          const alive = vi.element("#", "#ffffff");
          dead.to(alive).count(alive, 3);
          alive.to(alive).count(alive, 2, 3);
          alive.to(dead);
        },
        before,
      );

      expect(after.ids[4]).toBe(1);
    });

    test("Game of Life: survival rule — alive cell with 2 or 3 alive neighbors survives", async () => {
      const before = grid([
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 0],
      ]);

      const after = await step(
        root,
        (vi) => {
          const dead = vi.element(".", "#000000");
          const alive = vi.element("#", "#ffffff");
          dead.to(alive).count(alive, 3);
          alive.to(alive).count(alive, 2, 3);
          alive.to(dead);
        },
        before,
      );

      expect(after.ids[4]).toBe(1);
    });

    test("Game of Life: death rule — alive cell with fewer than 2 alive neighbors dies", async () => {
      const before = grid([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
      ]);

      const after = await step(
        root,
        (vi) => {
          const dead = vi.element(".", "#000000");
          const alive = vi.element("#", "#ffffff");
          dead.to(alive).count(alive, 3);
          alive.to(alive).count(alive, 2, 3);
          alive.to(dead);
        },
        before,
      );

      expect(after.ids[4]).toBe(0);
    });

    test("Game of Life: death by overcrowding — alive cell with 4+ neighbors dies", async () => {
      const before = grid([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
      ]);

      const after = await step(
        root,
        (vi) => {
          const dead = vi.element(".", "#000000");
          const alive = vi.element("#", "#ffffff");
          dead.to(alive).count(alive, 3);
          alive.to(alive).count(alive, 2, 3);
          alive.to(dead);
        },
        before,
      );

      expect(after.ids[4]).toBe(0);
    });

    test("count of 0 matches when no neighbors of that type exist", async () => {
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
          a.to(b).count(b, 0);
        },
        before,
      );

      expect(toRows(after)).toEqual([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
      ]);
    });
  });

  // ── COUNT_KIND condition ─────────────────────────────────────

  describe("COUNT_KIND condition", () => {
    test("Wireworld: wire becomes head when exactly 1 or 2 head neighbors", async () => {
      const before = grid([
        [0, 2, 0],
        [0, 1, 0],
        [0, 0, 0],
      ]);

      const after = await step(
        root,
        (vi) => {
          vi.element("empty", "#000000");
          const wire = vi.element("wire", "#ff8800");
          const head = vi.element("head", "#0088ff");
          vi.element("tail", "#ffffff");

          head.to(head).count(head, 8);
          wire.to(head).count(head, 1, 2);
        },
        before,
      );

      expect(after.ids[4]).toBe(2);
    });

    test("kind count matches across multiple elements in the kind", async () => {
      const before = grid([
        [1, 0, 2],
        [0, 0, 0],
        [0, 0, 0],
      ]);

      const after = await step(
        root,
        (vi) => {
          const conductor = vi.kind("conductor");
          const empty = vi.element("empty", "#000000");
          const wire = vi.element("wire", "#ff8800", [conductor]);
          vi.element("head", "#0088ff", [conductor]);
          vi.element("tail", "#ffffff", [conductor]);

          empty.to(wire).count(conductor, 2);
        },
        before,
      );

      expect(after.ids[1]).toBe(1);
    });
  });

  // ── IS_ELEMENT condition ──────────────────────────────────────

  describe("IS_ELEMENT condition", () => {
    test("cell transitions when specific neighbor matches element", async () => {
      const before = grid([
        [0, 1, 0],
        [0, 0, 0],
        [0, 0, 0],
      ]);

      const after = await step(
        root,
        (vi) => {
          const a = vi.element("a", "#ff0000");
          const b = vi.element("b", "#00ff00");
          a.to(b).is(Square.TOP, b);
        },
        before,
      );

      expect(after.ids[4]).toBe(1);
      expect(after.ids[3]).toBe(0);
    });

    test("is condition does not match when neighbor is different", async () => {
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
      const before = grid([
        [1, 0, 1],
        [0, 0, 0],
        [0, 0, 0],
      ]);

      const after = await step(
        root,
        (vi) => {
          const a = vi.element("a", "#ff0000");
          const b = vi.element("b", "#00ff00");
          a.to(b).is(Square.LEFT, Square.RIGHT);
        },
        before,
      );

      expect(after.ids[1]).toBe(1);
      expect(after.ids[4]).toBe(1);
    });

    test("is point does not match when positions have different elements", async () => {
      const before = grid([
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
      ]);

      const after = await step(
        root,
        (vi) => {
          const a = vi.element("a", "#ff0000");
          const b = vi.element("b", "#00ff00");
          a.to(b).is(Square.LEFT, Square.RIGHT);
        },
        before,
      );

      expect(after.ids[1]).toBe(0);
    });
  });

  // ── Accept strategies ─────────────────────────────────────────

  describe("accept strategies", () => {
    test("accept ALL: all conditions must pass", async () => {
      const before = grid([
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
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

      expect(after.ids[3]).toBe(1);
      expect(after.ids[1]).toBe(0);
    });

    test("accept ANY: at least one condition must pass", async () => {
      const before = grid([
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
      ]);

      const after = await step(
        root,
        (vi) => {
          const a = vi.element("a", "#ff0000");
          const b = vi.element("b", "#00ff00");
          a.to(b).count(b, 3).is(Square.TOP, b).accept("any");
        },
        before,
      );

      expect(after.ids[3]).toBe(1);
    });

    test("accept ONE: exactly one condition must pass", async () => {
      const before = grid([
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
      ]);

      const after = await step(
        root,
        (vi) => {
          const a = vi.element("a", "#ff0000");
          const b = vi.element("b", "#00ff00");
          a.to(b).count(b, 3).is(Square.TOP, b).accept("one");
        },
        before,
      );

      expect(after.ids[3]).toBe(1);
    });

    test("accept ONE: fails when both conditions pass", async () => {
      const before = grid([
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
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

      expect(after.ids[3]).toBe(0);
    });

    test("accept NONE: transitions when no conditions pass", async () => {
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
          a.to(b).count(b, 1).accept("none");
        },
        before,
      );

      expect(toRows(after)).toEqual([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
      ]);
    });

    test("accept NONE: does not transition when a condition passes", async () => {
      const before = grid([
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
      ]);

      const after = await step(
        root,
        (vi) => {
          const a = vi.element("a", "#ff0000");
          const b = vi.element("b", "#00ff00");
          a.to(b).count(b, 1).accept("none");
        },
        before,
      );

      expect(after.ids[1]).toBe(0);
    });
  });

  // ── Wrapping (toroidal grid) ──────────────────────────────────

  describe("wrapping (toroidal grid)", () => {
    test("bottom edge wraps to top", async () => {
      // 4×4 grid: 'b' at top row. Cell at bottom row checks BOTTOM, which wraps to top.
      const before = grid([
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
      ]);

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

      // Cell (1,3) BOTTOM wraps to (1,0) which is 'b' → transitions
      expect(after.ids[3 * 4 + 1]).toBe(1);
    });

    test("right edge wraps to left", async () => {
      // 4×4 grid: 'b' at left column. Cell at right column checks RIGHT, which wraps to left.
      const before = grid([
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
      ]);

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

      // Cell (3,0) RIGHT wraps to (0,0) which is 'b' → transitions
      expect(after.ids[3]).toBe(1);
    });

    test("corner wraps diagonally", async () => {
      // 4×4 grid: 'b' at top-left corner. Cell at bottom-right checks BOTTOM_RIGHT, which wraps.
      const before = grid([
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
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

      // Cell (3,3) BOTTOM_RIGHT wraps to (0,0) which is 'b' → transitions
      expect(after.ids[3 * 4 + 3]).toBe(1);
    });

    test("count of 8 matches all cells in wrapping mode", async () => {
      const before = grid([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
      ]);

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

      // In wrapping mode, every cell has 8 neighbors
      expect(toRows(after)).toEqual([
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
      ]);
    });
  });

  // ── Non-wrapping (bounded grid) ──────────────────────────────

  describe("non-wrapping (bounded grid)", () => {
    test("edge cell does not see wrapped neighbors", async () => {
      // 4×4 grid: 'b' at top-left corner. Cell at bottom-right checks BOTTOM_RIGHT.
      const before = grid([
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
      ]);

      const after = await step(
        root,
        (vi) => {
          const a = vi.element("a", "#ff0000");
          const b = vi.element("b", "#00ff00");
          a.to(b).is(Square.BOTTOM_RIGHT, b);
        },
        before,
      );

      // In non-wrapping mode, cell (3,3) BOTTOM_RIGHT is OOB → no match
      expect(after.ids[3 * 4 + 3]).toBe(0);
    });

    test("corner cells have fewer effective neighbors", async () => {
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
          // Corner cells have 3 neighbors, so count=3 should match corners only
          a.to(b).count(a, 3);
        },
        before,
      );

      // Corners have 3 neighbors, edges have 5, center has 8
      expect(toRows(after)).toEqual([
        [1, 0, 1],
        [0, 0, 0],
        [1, 0, 1],
      ]);
    });

    test("edge cells have 5 effective neighbors", async () => {
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
          a.to(b).count(a, 5);
        },
        before,
      );

      // Only edge (non-corner) cells have exactly 5 neighbors
      expect(toRows(after)).toEqual([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
      ]);
    });

    test("is condition returns false for OOB neighbor", async () => {
      // 'b' at (0,0). Cell (0,1) checks is(LEFT, b).
      // LEFT of (0,1) is (-1,1) which is OOB in non-wrapping.
      const before = grid([
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
      ]);

      const after = await step(
        root,
        (vi) => {
          const a = vi.element("a", "#ff0000");
          const b = vi.element("b", "#00ff00");
          a.to(b).is(Square.LEFT, b);
        },
        before,
      );

      // (0,1) LEFT is OOB → condition fails → stays 'a'
      expect(after.ids[3]).toBe(0);
      // (1,0) LEFT is (0,0) = 'b' → condition passes → becomes 'b'
      expect(after.ids[1]).toBe(1);
    });
  });

  // ── Edge cases ────────────────────────────────────────────────

  describe("edge cases", () => {
    test("uniform grid with unconditional rule transitions all cells", async () => {
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
          a.to(b);
        },
        before,
      );

      expect(toRows(after)).toEqual([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
      ]);
    });

    test("count of 8 matches when all neighbors are the same element", async () => {
      const before = grid([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
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
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
      ]);
    });

    test("count of 0 does not match when all neighbors are the same element", async () => {
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
          a.to(b).count(a, 0);
        },
        before,
      );

      expect(toRows(after)).toEqual([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
      ]);
    });

    test("uniform grid: all same element with no matching rule stays unchanged", async () => {
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
          a.to(b).count(b, 1);
        },
        before,
      );

      expect(toRows(after)).toEqual(toRows(before));
    });

    test("to point (SELF) keeps element as itself", async () => {
      const before = grid([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 0],
      ]);

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
      const before = grid([
        [0, 1, 0],
        [0, 0, 0],
        [0, 0, 0],
      ]);

      const after = await step(
        root,
        (vi) => {
          const a = vi.element("a", "#ff0000");
          vi.element("b", "#00ff00");
          a.to(Square.TOP);
        },
        before,
      );

      expect(toRows(after)).toEqual([
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 0],
      ]);
    });

    test("first matching rule wins (rules are ordered)", async () => {
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
          const c = vi.element("c", "#0000ff");
          a.to(b);
          a.to(c);
        },
        before,
      );

      expect(toRows(after)).toEqual([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
      ]);
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
      const before = grid([
        [D, D, D, D],
        [D, A, A, D],
        [D, A, A, D],
        [D, D, D, D],
      ]);

      const after = await step(root, buildLife, before);

      expect(toRows(after)).toEqual(toRows(before));
    });

    test("blinker oscillates (period 2)", async () => {
      const before = grid([
        [D, D, D, D, D],
        [D, D, D, D, D],
        [D, A, A, A, D],
        [D, D, D, D, D],
        [D, D, D, D, D],
      ]);

      const vi = vivarium();
      buildLife(vi);
      const automaton = vi.create();

      const step1 = await gpuEvolve(root, automaton, before);

      expect(toRows(step1)).toEqual([
        [D, D, D, D, D],
        [D, D, A, D, D],
        [D, D, A, D, D],
        [D, D, A, D, D],
        [D, D, D, D, D],
      ]);

      const step2 = await gpuEvolve(root, automaton, step1);

      expect(toRows(step2)).toEqual(toRows(before));
    });

    test("single alive cell dies (underpopulation)", async () => {
      const before = grid([
        [D, D, D, D, D],
        [D, D, D, D, D],
        [D, D, A, D, D],
        [D, D, D, D, D],
        [D, D, D, D, D],
      ]);

      const after = await step(root, buildLife, before);

      expect(after.ids[12]).toBe(D);
    });

    test("two adjacent alive cells both die", async () => {
      const before = grid([
        [D, D, D, D, D],
        [D, D, D, D, D],
        [D, D, A, A, D],
        [D, D, D, D, D],
        [D, D, D, D, D],
      ]);

      const after = await step(root, buildLife, before);

      expect(after.ids[12]).toBe(D);
      expect(after.ids[13]).toBe(D);
    });

    test("L-shape (4 cells) evolves correctly", async () => {
      const before = grid([
        [D, D, D, D, D],
        [D, D, A, D, D],
        [D, D, A, D, D],
        [D, D, A, A, D],
        [D, D, D, D, D],
      ]);

      const after = await step(root, buildLife, before);

      expect(after.ids[1 * 5 + 2]).toBe(D);
      expect(after.ids[2 * 5 + 2]).toBe(A);
      expect(after.ids[3 * 5 + 2]).toBe(A);
      expect(after.ids[3 * 5 + 3]).toBe(A);
      expect(after.ids[2 * 5 + 1]).toBe(A);
    });

    test("full board overcrowding: corners survive in non-wrapping mode", async () => {
      const before = grid([
        [A, A, A],
        [A, A, A],
        [A, A, A],
      ]);

      const after = await step(root, buildLife, before);

      // In non-wrapping mode, corners have 3 alive neighbors → survive (2,3 rule).
      // Edge cells have 5 alive neighbors → die. Center has 8 → die.
      expect(toRows(after)).toEqual([
        [A, D, A],
        [D, D, D],
        [A, D, A],
      ]);
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

      // Center cell (index 4) should transition because it has exactly 4 'b' cardinal neighbors
      expect(after.ids[4]).toBe(1);
    });

    test("cross neighborhood ignores diagonal neighbors (COUNT_ELEMENT)", async () => {
      // Only diagonals have 'b', no cardinal neighbors are 'b'
      const before = grid([
        [1, 0, 1],
        [0, 0, 0],
        [1, 0, 1],
      ]);

      const after = await step(
        root,
        (vi) => {
          const a = vi.element("a", "#ff0000");
          const b = vi.element("b", "#00ff00");
          // With cross neighborhood, center has 0 'b' cardinal neighbors (diagonals don't count)
          a.to(b).count(b, 0);
        },
        before,
        "cross",
      );

      // Center cell should transition because it has 0 'b' neighbors in cross mode
      expect(after.ids[4]).toBe(1);
    });

    test("cross neighborhood counts only cardinal neighbors (COUNT_KIND)", async () => {
      // Center cell has 'empty'. Cardinal neighbors are 'wire'(1) and 'head'(2).
      const before = grid([
        [0, 1, 0],
        [2, 0, 1],
        [0, 2, 0],
      ]);

      const after = await step(
        root,
        (vi) => {
          const conductor = vi.kind("conductor");
          const empty = vi.element("empty", "#000000");
          const wire = vi.element("wire", "#ff8800", [conductor]);
          vi.element("head", "#0088ff", [conductor]);

          // Center has 4 cardinal conductor neighbors (cross mode)
          empty.to(wire).count(conductor, 4);
        },
        before,
        "cross",
      );

      // Center (index 4) should transition to wire
      expect(after.ids[4]).toBe(1);
    });
  });

  // ── Multi-step evolution ──────────────────────────────────────

  describe("multi-step evolution", () => {
    test("three-element cycle: a→b→c→a", async () => {
      const before = grid([
        [0, 0, 0],
        [0, 1, 2],
        [0, 0, 0],
      ]);

      const vi = vivarium();
      const a = vi.element("a", "#ff0000");
      const b = vi.element("b", "#00ff00");
      const c = vi.element("c", "#0000ff");
      a.to(b);
      b.to(c);
      c.to(a);
      const automaton = vi.create();

      const step1 = await gpuEvolve(root, automaton, before);
      expect(toRows(step1)).toEqual([
        [1, 1, 1],
        [1, 2, 0],
        [1, 1, 1],
      ]);

      const step2 = await gpuEvolve(root, automaton, step1);
      expect(toRows(step2)).toEqual([
        [2, 2, 2],
        [2, 0, 1],
        [2, 2, 2],
      ]);

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
      const automaton = vi.create();

      const result = await gpuEvolveMulti(root, automaton, before, 4);

      expect(toRows(result)).toEqual([
        [D, D, D, D, D, D],
        [D, D, A, D, D, D],
        [D, D, D, A, D, D],
        [D, A, A, A, D, D],
        [D, D, D, D, D, D],
        [D, D, D, D, D, D],
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

      const initialGrid = [
        [0, 1, 0],
        [1, 0, 1],
        [0, 0, 0],
      ];

      const canvas = createCanvas(3, 3);
      const { readGrid } = setup({ canvas, automaton, initialGrid });
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

      const initialGrid = [
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 0],
      ];

      const canvas = createCanvas(3, 3);
      const { evolve, readGrid } = setup({ canvas, automaton, initialGrid });
      await evolve();
      const result = await readGrid();

      expect(result).toEqual([1, 0, 1, 0, 1, 1, 1, 1, 1]);
    });

    test("writeCell updates a single cell", async () => {
      const vi = vivarium();
      vi.element("a", "#ff0000");
      vi.element("b", "#00ff00");
      const automaton = vi.create();

      const initialGrid = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
      ];

      const canvas = createCanvas(3, 3);
      const { readGrid, writeCell } = setup({
        canvas,
        automaton,
        initialGrid,
      });

      writeCell(4, 1);
      const result = await readGrid();

      expect(result).toEqual([0, 0, 0, 0, 1, 0, 0, 0, 0]);
    });

    test("writeCell throws on out-of-bounds index", () => {
      const vi = vivarium();
      vi.element("a", "#ff0000");
      const automaton = vi.create();

      const canvas = createCanvas(3, 3);
      const { writeCell } = setup({
        canvas,
        automaton,
        initialGrid: [
          [0, 0, 0],
          [0, 0, 0],
          [0, 0, 0],
        ],
      });

      expect(() => writeCell(9, 0)).toThrow("out of bounds");
      expect(() => writeCell(-1, 0)).toThrow("out of bounds");
    });

    test("writeCell throws on invalid element index", () => {
      const vi = vivarium();
      vi.element("a", "#ff0000");
      vi.element("b", "#00ff00");
      const automaton = vi.create();

      const canvas = createCanvas(3, 3);
      const { writeCell } = setup({
        canvas,
        automaton,
        initialGrid: [
          [0, 0, 0],
          [0, 0, 0],
          [0, 0, 0],
        ],
      });

      expect(() => writeCell(0, 2)).toThrow("invalid");
      expect(() => writeCell(0, -1)).toThrow("invalid");
    });

    test("writeGrid overwrites the entire grid", async () => {
      const vi = vivarium();
      vi.element("a", "#ff0000");
      vi.element("b", "#00ff00");
      const automaton = vi.create();

      const canvas = createCanvas(3, 3);
      const { readGrid, writeGrid } = setup({
        canvas,
        automaton,
        initialGrid: [
          [0, 0, 0],
          [0, 0, 0],
          [0, 0, 0],
        ],
      });

      const snapshot = [1, 0, 1, 0, 1, 0, 1, 0, 1];
      writeGrid(snapshot);
      const result = await readGrid();

      expect(result).toEqual(snapshot);
    });

    test("writeGrid throws on wrong length", () => {
      const vi = vivarium();
      vi.element("a", "#ff0000");
      const automaton = vi.create();

      const canvas = createCanvas(3, 3);
      const { writeGrid } = setup({
        canvas,
        automaton,
        initialGrid: [
          [0, 0, 0],
          [0, 0, 0],
          [0, 0, 0],
        ],
      });

      expect(() => writeGrid([0, 0])).toThrow("does not match");
    });

    test("writeGrid throws on invalid element index", () => {
      const vi = vivarium();
      vi.element("a", "#ff0000");
      vi.element("b", "#00ff00");
      const automaton = vi.create();

      const canvas = createCanvas(3, 3);
      const { writeGrid } = setup({
        canvas,
        automaton,
        initialGrid: [
          [0, 0, 0],
          [0, 0, 0],
          [0, 0, 0],
        ],
      });

      expect(() => writeGrid([0, 0, 0, 0, 2, 0, 0, 0, 0])).toThrow("invalid");
    });

    test("writeCellAt updates a cell by coordinates", async () => {
      const vi = vivarium();
      vi.element("a", "#ff0000");
      vi.element("b", "#00ff00");
      const automaton = vi.create();

      const canvas = createCanvas(3, 3);
      const { readGrid, writeCellAt } = setup({
        canvas,
        automaton,
        initialGrid: [
          [0, 0, 0],
          [0, 0, 0],
          [0, 0, 0],
        ],
      });

      writeCellAt(1, 1, 1); // center cell
      const result = await readGrid();

      expect(result).toEqual([0, 0, 0, 0, 1, 0, 0, 0, 0]);
    });

    test("writeCellAt throws on out-of-bounds coordinates", () => {
      const vi = vivarium();
      vi.element("a", "#ff0000");
      const automaton = vi.create();

      const canvas = createCanvas(3, 3);
      const { writeCellAt } = setup({
        canvas,
        automaton,
        initialGrid: [
          [0, 0, 0],
          [0, 0, 0],
          [0, 0, 0],
        ],
      });

      expect(() => writeCellAt(3, 0, 0)).toThrow("out of bounds");
      expect(() => writeCellAt(-1, 0, 0)).toThrow("out of bounds");
      expect(() => writeCellAt(0, 3, 0)).toThrow("out of bounds");
      expect(() => writeCellAt(0, -1, 0)).toThrow("out of bounds");
    });
  });

  // ── setup validation ────────────────────────────────────────

  describe("setup validation", () => {
    const createCanvas = (width: number, height: number): HTMLCanvasElement => {
      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      return canvas;
    };

    test("wrapping mode throws on non-power-of-2 width", () => {
      const vi = vivarium("square", { wrapping: true });
      vi.element("a", "#ff0000");
      const automaton = vi.create();

      const canvas = createCanvas(3, 4);

      expect(() => setup({ canvas, automaton })).toThrow("powers of 2");
    });

    test("wrapping mode throws on non-power-of-2 height", () => {
      const vi = vivarium("square", { wrapping: true });
      vi.element("a", "#ff0000");
      const automaton = vi.create();

      const canvas = createCanvas(4, 3);

      expect(() => setup({ canvas, automaton })).toThrow("powers of 2");
    });

    test("wrapping mode accepts power-of-2 dimensions", () => {
      const vi = vivarium("square", { wrapping: true });
      vi.element("a", "#ff0000");
      const automaton = vi.create();

      const canvas = createCanvas(8, 8);

      expect(() =>
        setup({
          canvas,
          automaton,
          initialGrid: Array.from({ length: 64 }, () => 0),
        }),
      ).not.toThrow();
    });

    test("non-wrapping mode accepts non-power-of-2 dimensions", () => {
      const vi = vivarium();
      vi.element("a", "#ff0000");
      const automaton = vi.create();

      const canvas = createCanvas(3, 5);

      expect(() =>
        setup({
          canvas,
          automaton,
          initialGrid: Array.from({ length: 15 }, () => 0),
        }),
      ).not.toThrow();
    });

    test("throws when initialGrid length does not match dimensions", () => {
      const vi = vivarium();
      vi.element("a", "#ff0000");
      const automaton = vi.create();

      const canvas = createCanvas(3, 3);

      expect(() =>
        setup({
          canvas,
          automaton,
          initialGrid: [0, 0, 0, 0, 0],
        }),
      ).toThrow("does not match");
    });

    test("throws when 2d initialGrid flattened length does not match dimensions", () => {
      const vi = vivarium();
      vi.element("a", "#ff0000");
      const automaton = vi.create();

      const canvas = createCanvas(3, 3);

      expect(() =>
        setup({
          canvas,
          automaton,
          initialGrid: [
            [0, 0, 0],
            [0, 0, 0],
          ],
        }),
      ).toThrow("does not match");
    });
  });
});
