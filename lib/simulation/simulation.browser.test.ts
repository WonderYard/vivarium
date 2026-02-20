/// <reference types="vite/client" />

import type { Automaton } from "@/automaton/types";
import { GpuCondition, GpuElement, GpuRule, Square, WORKGROUP_SIZE } from "@/common/constants";
import { vivarium } from "@/vivarium/vivarium";
import { compileGpuAutomaton } from "@/webgpu/compiler";
import { automatonLayout, gridLayout, mainCompute, setSeed, setup } from "@/webgpu/setup";
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
  const pipeline = root["~unstable"].withCompute(mainCompute).createPipeline();

  setSeed(root.createUniform(d.f32, 0));

  const WORKGROUP_COUNT_W = Math.ceil(width / WORKGROUP_SIZE[0]);
  const WORKGROUP_COUNT_H = Math.ceil(height / WORKGROUP_SIZE[1]);

  const { gpuNeighborhood, gpuElements, gpuRules, gpuConditions } = compileGpuAutomaton(automaton);

  const palette = gpuElements.map((el) => el.color);

  const dimensions = root.createBuffer(d.vec2u, d.vec2u(width, height)).$usage("uniform");

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
): Promise<Grid> => {
  const vi = vivarium(neighborhood);
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
    test("top edge wraps to bottom", async () => {
      const before = grid([
        [0, 0, 0],
        [0, 0, 0],
        [0, 1, 0],
      ]);

      const after = await step(
        root,
        (vi) => {
          const a = vi.element("a", "#ff0000");
          const b = vi.element("b", "#00ff00");
          a.to(b).is(Square.BOTTOM, b);
        },
        before,
      );

      expect(after.ids[7]).toBe(1);
      expect(after.ids[4]).toBe(1);
    });

    test("left edge wraps to right", async () => {
      const before = grid([
        [0, 0, 1],
        [0, 0, 0],
        [0, 0, 0],
      ]);

      const after = await step(
        root,
        (vi) => {
          const a = vi.element("a", "#ff0000");
          const b = vi.element("b", "#00ff00");
          a.to(b).is(Square.RIGHT, b);
        },
        before,
      );

      expect(after.ids[1]).toBe(1);
    });

    test("corner wraps diagonally", async () => {
      const before = grid([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 1],
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

      expect(after.ids[4]).toBe(1);
      expect(after.ids[0]).toBe(0);
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

      expect(toRows(after)).toEqual([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
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

    test("full board dies (overcrowding)", async () => {
      const before = grid([
        [A, A, A],
        [A, A, A],
        [A, A, A],
      ]);

      const after = await step(root, buildLife, before);

      expect(toRows(after)).toEqual([
        [D, D, D],
        [D, D, D],
        [D, D, D],
      ]);
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
});
