import { describe, expect, test } from "vitest";
import tgpu, { type TgpuUniform } from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { randf } from "@typegpu/noise";
import type { Automaton } from "@/automaton/types";
import {
  Accept,
  GpuCondition,
  GpuElement,
  GpuRule,
  Opcode,
  Square,
  To,
  WORKGROUP_SIZE,
} from "@/common/constants";
import { evaluateCondition, accepted } from "@/simulation/kernel";
import { compileGpuAutomaton } from "@/webgpu/compiler";
import { vivarium } from "@/vivarium/vivarium";

// ── GPU test harness ────────────────────────────────────────────────

/**
 * Minimal GPU pipeline that mirrors setup.ts but allows controlled
 * initial state and reading back IDs for assertions.
 */
async function createGpuTestHarness() {
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error("WebGPU adapter not available");
  }
  const device = await adapter.requestDevice();
  const root = tgpu.initFromDevice({ device });

  return { root, device };
}

const gridLayout = tgpu.bindGroupLayout({
  dimensions: { uniform: d.vec2u },
  colors: { storage: d.arrayOf(d.u32), access: "mutable" },
  newColors: { storage: d.arrayOf(d.u32), access: "mutable" },
  ids: { storage: d.arrayOf(d.u32), access: "mutable" },
  newIds: { storage: d.arrayOf(d.u32), access: "mutable" },
});

const automatonLayout = tgpu.bindGroupLayout({
  neighborhood: { uniform: d.u32 },
  elements: {
    storage: d.arrayOf(GpuElement),
    access: "readonly",
  },
  rules: {
    storage: d.arrayOf(GpuRule),
    access: "readonly",
  },
  conditions: {
    storage: d.arrayOf(GpuCondition),
    access: "readonly",
  },
});

const pointToIndex = (x: number, y: number) => {
  "use gpu";
  return (
    (y % gridLayout.bound.dimensions.$.y) * gridLayout.bound.dimensions.$.x +
    (x % gridLayout.bound.dimensions.$.x)
  );
};

const idAt = (x: number, y: number) => {
  "use gpu";
  return gridLayout.bound.ids.$[pointToIndex(x, y)];
};

let seed: TgpuUniform<d.F32>;

const mainCompute = tgpu["~unstable"].computeFn({
  workgroupSize: WORKGROUP_SIZE,
  in: { pos: d.builtin.globalInvocationId },
})(({ pos }) => {
  const x = pos.x;
  const y = pos.y;
  const index = pointToIndex(x, y);

  const color = gridLayout.bound.colors.$[index];
  const id = gridLayout.bound.ids.$[index];

  const element = automatonLayout.bound.elements.$[id];

  const ruleStart = element.ruleStart;
  const ruleEnd = element.ruleEnd;

  const n0 = idAt(x - 1, y - 1);
  const n1 = idAt(x, y - 1);
  const n2 = idAt(x + 1, y - 1);
  const n3 = idAt(x - 1, y);
  const n4 = idAt(x + 1, y);
  const n5 = idAt(x - 1, y + 1);
  const n6 = idAt(x, y + 1);
  const n7 = idAt(x + 1, y + 1);

  for (let i = ruleStart; i < ruleEnd; i++) {
    const rule = automatonLayout.bound.rules.$[i];
    const accept = rule.accept as Accept;

    let passing = d.u32(0);

    const conditionsStart = rule.conditionsStart;
    const conditionsEnd = rule.conditionsEnd;
    const conditionsCount = conditionsEnd - conditionsStart;

    for (let j = conditionsStart; j < conditionsEnd; j++) {
      const condition = automatonLayout.bound.conditions.$[j];
      const opcode = condition.opcode as Opcode;

      const comparePoint = condition.checkPointOrComparePoint;
      const compareId = idAt(x + comparePoint.x, y + comparePoint.y);
      const withPoint = condition.withPoint;
      const withId = idAt(x + withPoint.x, y + withPoint.y);

      if (opcode === Opcode.CHANCE) {
        const chance = condition.chance;
        randf.seed3(
          d.vec3f(
            std.div(
              d.vec2f(pos.xy),
              d.vec2f(gridLayout.bound.dimensions.$.xy)
            ),
            seed.$
          )
        );
        passing += std.select(d.u32(0), d.u32(1), randf.sample() < chance);
      } else {
        passing += evaluateCondition(
          opcode,
          n0, n1, n2, n3, n4, n5, n6, n7,
          condition.checkId,
          condition.countOrWithId,
          compareId,
          withId
        );
      }
    }

    if (accepted(accept, passing, conditionsCount)) {
      let resolvedId = rule.toId;

      const toType = rule.toType as To;

      if (toType === To.POINT) {
        const pointIndex = pointToIndex(
          x + d.u32(rule.toNeighbor.x),
          y + d.u32(rule.toNeighbor.y)
        );
        resolvedId = gridLayout.bound.ids.$[pointIndex];
      }

      gridLayout.bound.newIds.$[index] = resolvedId;
      gridLayout.bound.newColors.$[index] =
        automatonLayout.bound.elements.$[resolvedId].color;

      return;
    }
  }

  gridLayout.bound.newIds.$[index] = id;
  gridLayout.bound.newColors.$[index] = color;
});

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

/**
 * Run one GPU evolution step: write initial IDs, dispatch compute, read back IDs.
 */
async function gpuEvolve(
  root: ReturnType<typeof tgpu.initFromDevice>,
  automaton: Automaton,
  inputGrid: Grid
): Promise<Grid> {
  const { width, height, ids } = inputGrid;
  const pipeline = root["~unstable"]
    .withCompute(mainCompute)
    .createPipeline();

  seed = root.createUniform(d.f32, 0);

  const WORKGROUP_COUNT_W = Math.ceil(width / WORKGROUP_SIZE[0]);
  const WORKGROUP_COUNT_H = Math.ceil(height / WORKGROUP_SIZE[1]);

  const { gpuNeighborhood, gpuElements, gpuRules, gpuConditions } =
    compileGpuAutomaton(automaton);

  const palette = gpuElements.map((el) => el.color);

  const dimensions = root
    .createBuffer(d.vec2u, d.vec2u(width, height))
    .$usage("uniform");

  const colors0 = root
    .createBuffer(d.arrayOf(d.u32, width * height))
    .$usage("storage");

  const colors1 = root
    .createBuffer(d.arrayOf(d.u32, width * height))
    .$usage("storage");

  const ids0 = root
    .createBuffer(d.arrayOf(d.u32, width * height))
    .$usage("storage");

  const ids1 = root
    .createBuffer(d.arrayOf(d.u32, width * height))
    .$usage("storage");

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

  const automatonGroup = root.createBindGroup(automatonLayout, {
    neighborhood: root
      .createBuffer(d.u32, gpuNeighborhood)
      .$usage("uniform"),
    elements: root
      .createBuffer(
        d.arrayOf(GpuElement, Math.max(gpuElements.length, 1)),
        gpuElements
      )
      .$usage("storage"),
    rules: root
      .createBuffer(
        d.arrayOf(GpuRule, Math.max(gpuRules.length, 1)),
        gpuRules
      )
      .$usage("storage"),
    conditions: root
      .createBuffer(
        d.arrayOf(GpuCondition, Math.max(gpuConditions.length, 1)),
        gpuConditions
      )
      .$usage("storage"),
  });

  // Dispatch compute
  pipeline
    .with(automatonGroup)
    .with(gridGroup)
    .dispatchWorkgroups(WORKGROUP_COUNT_W, WORKGROUP_COUNT_H);

  // Read back IDs from the output buffer (ids1)
  const resultIds = await ids1.read();

  return { width, height, ids: Array.from(resultIds) };
}

/**
 * Run multiple GPU evolution steps sequentially.
 */
async function gpuEvolveMulti(
  root: ReturnType<typeof tgpu.initFromDevice>,
  automaton: Automaton,
  inputGrid: Grid,
  steps: number
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
  neighborhood?: "square" | "cross"
): Promise<Grid> => {
  const vi = vivarium(neighborhood);
  build(vi);
  return gpuEvolve(root, vi.create(), inputGrid);
};

// ── Tests ───────────────────────────────────────────────────────────

describe("GPU simulation", async () => {
  const { root } = await createGpuTestHarness();

  // ── Unconditional rules ─────────────────────────────────────────

  describe("unconditional rules", () => {
    test("oscillator: every cell toggles between two elements", async () => {
      const before = grid([
        [0, 1],
        [1, 0],
      ]);

      const after = await step(
        root,
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

    test("oscillator returns to original state after two steps", async () => {
      const original = grid([
        [0, 1, 0],
        [1, 0, 1],
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
        [0, 1],
        [1, 0],
      ]);

      const after = await step(
        root,
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

  // ── No matching rule ──────────────────────────────────────────

  describe("no matching rule", () => {
    test("cells with no rules remain unchanged", async () => {
      const before = grid([
        [0, 0],
        [0, 0],
      ]);

      const after = await step(
        root,
        (vi) => {
          vi.element("a", "#ff0000");
        },
        before
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
        before
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
          const dead = vi.element("dead", "#000000");
          const alive = vi.element("alive", "#ffffff");
          dead.to(alive).count(alive, 3);
          alive.to(alive).count(alive, 2, 3);
          alive.to(dead);
        },
        before
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
          const dead = vi.element("dead", "#000000");
          const alive = vi.element("alive", "#ffffff");
          dead.to(alive).count(alive, 3);
          alive.to(alive).count(alive, 2, 3);
          alive.to(dead);
        },
        before
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
          const dead = vi.element("dead", "#000000");
          const alive = vi.element("alive", "#ffffff");
          dead.to(alive).count(alive, 3);
          alive.to(alive).count(alive, 2, 3);
          alive.to(dead);
        },
        before
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
          const dead = vi.element("dead", "#000000");
          const alive = vi.element("alive", "#ffffff");
          dead.to(alive).count(alive, 3);
          alive.to(alive).count(alive, 2, 3);
          alive.to(dead);
        },
        before
      );

      expect(after.ids[4]).toBe(0);
    });

    test("count of 0 matches when no neighbors of that type exist", async () => {
      const before = grid([
        [0, 0],
        [0, 0],
      ]);

      const after = await step(
        root,
        (vi) => {
          const a = vi.element("a", "#ff0000");
          const b = vi.element("b", "#00ff00");
          a.to(b).count(b, 0);
        },
        before
      );

      expect(toRows(after)).toEqual([
        [1, 1],
        [1, 1],
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
        before
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
        before
      );

      expect(after.ids[1]).toBe(1);
    });
  });

  // ── IS_ELEMENT condition ──────────────────────────────────────

  describe("IS_ELEMENT condition", () => {
    test("cell transitions when specific neighbor matches element", async () => {
      const before = grid([
        [0, 1],
        [0, 0],
      ]);

      const after = await step(
        root,
        (vi) => {
          const a = vi.element("a", "#ff0000");
          const b = vi.element("b", "#00ff00");
          a.to(b).is(Square.TOP, b);
        },
        before
      );

      expect(after.ids[3]).toBe(1);
      expect(after.ids[2]).toBe(0);
    });

    test("is condition does not match when neighbor is different", async () => {
      const before = grid([
        [0, 0],
        [0, 0],
      ]);

      const after = await step(
        root,
        (vi) => {
          const a = vi.element("a", "#ff0000");
          const b = vi.element("b", "#00ff00");
          a.to(b).is(Square.TOP, b);
        },
        before
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
        before
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
        before
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
        before
      );

      expect(after.ids[3]).toBe(1);
      expect(after.ids[1]).toBe(0);
    });

    test("accept ANY: at least one condition must pass", async () => {
      const before = grid([
        [1, 0],
        [0, 0],
      ]);

      const after = await step(
        root,
        (vi) => {
          const a = vi.element("a", "#ff0000");
          const b = vi.element("b", "#00ff00");
          a.to(b).count(b, 3).is(Square.TOP, b).accept("any");
        },
        before
      );

      expect(after.ids[2]).toBe(1);
    });

    test("accept ONE: exactly one condition must pass", async () => {
      const before = grid([
        [1, 0],
        [0, 0],
      ]);

      const after = await step(
        root,
        (vi) => {
          const a = vi.element("a", "#ff0000");
          const b = vi.element("b", "#00ff00");
          a.to(b).count(b, 3).is(Square.TOP, b).accept("one");
        },
        before
      );

      expect(after.ids[2]).toBe(1);
    });

    test("accept ONE: fails when both conditions pass", async () => {
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
        before
      );

      expect(after.ids[2]).toBe(0);
    });

    test("accept NONE: transitions when no conditions pass", async () => {
      const before = grid([
        [0, 0],
        [0, 0],
      ]);

      const after = await step(
        root,
        (vi) => {
          const a = vi.element("a", "#ff0000");
          const b = vi.element("b", "#00ff00");
          a.to(b).count(b, 1).accept("none");
        },
        before
      );

      expect(toRows(after)).toEqual([
        [1, 1],
        [1, 1],
      ]);
    });

    test("accept NONE: does not transition when a condition passes", async () => {
      const before = grid([
        [1, 0],
        [0, 0],
      ]);

      const after = await step(
        root,
        (vi) => {
          const a = vi.element("a", "#ff0000");
          const b = vi.element("b", "#00ff00");
          a.to(b).count(b, 2).accept("none");
        },
        before
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
        before
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
        before
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
        before
      );

      expect(after.ids[4]).toBe(1);
      expect(after.ids[0]).toBe(0);
    });
  });

  // ── Edge cases ────────────────────────────────────────────────

  describe("edge cases", () => {
    test("single cell grid with unconditional rule transitions", async () => {
      const before = grid([[0]]);

      const after = await step(
        root,
        (vi) => {
          const a = vi.element("a", "#ff0000");
          const b = vi.element("b", "#00ff00");
          a.to(b);
        },
        before
      );

      expect(toRows(after)).toEqual([[1]]);
    });

    test("single cell grid: all neighbors wrap to self", async () => {
      const before = grid([[0]]);

      const after = await step(
        root,
        (vi) => {
          const a = vi.element("a", "#ff0000");
          const b = vi.element("b", "#00ff00");
          a.to(b).count(a, 8);
        },
        before
      );

      expect(toRows(after)).toEqual([[1]]);
    });

    test("single cell grid: count excludes self", async () => {
      const before = grid([[0]]);

      const after = await step(
        root,
        (vi) => {
          const a = vi.element("a", "#ff0000");
          const b = vi.element("b", "#00ff00");
          a.to(b).count(a, 0);
        },
        before
      );

      expect(toRows(after)).toEqual([[0]]);
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
        before
      );

      expect(toRows(after)).toEqual(toRows(before));
    });

    test("uniform grid: all cells transition when unconditional", async () => {
      const before = grid([
        [0, 0],
        [0, 0],
      ]);

      const after = await step(
        root,
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

    test("to point (SELF) keeps element as itself", async () => {
      const before = grid([
        [0, 1],
        [1, 0],
      ]);

      const after = await step(
        root,
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

    test("to point copies neighbor's element", async () => {
      const before = grid([
        [0, 1],
        [0, 0],
      ]);

      const after = await step(
        root,
        (vi) => {
          const a = vi.element("a", "#ff0000");
          vi.element("b", "#00ff00");
          a.to(Square.TOP);
        },
        before
      );

      expect(after.ids[0]).toBe(0);
      expect(after.ids[1]).toBe(1);
      expect(after.ids[2]).toBe(0);
      expect(after.ids[3]).toBe(1);
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
        before
      );

      expect(toRows(after)).toEqual([[1]]);
    });
  });

  // ── Game of Life patterns ─────────────────────────────────────

  describe("Game of Life patterns", () => {
    const buildLife = (vi: ReturnType<typeof vivarium>) => {
      const dead = vi.element("dead", "#000000");
      const alive = vi.element("alive", "#ffffff");
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
      const before = grid([[0, 1, 2]]);

      const vi = vivarium();
      const a = vi.element("a", "#ff0000");
      const b = vi.element("b", "#00ff00");
      const c = vi.element("c", "#0000ff");
      a.to(b);
      b.to(c);
      c.to(a);
      const automaton = vi.create();

      const step1 = await gpuEvolve(root, automaton, before);
      expect(toRows(step1)).toEqual([[1, 2, 0]]);

      const step2 = await gpuEvolve(root, automaton, step1);
      expect(toRows(step2)).toEqual([[2, 0, 1]]);

      const step3 = await gpuEvolve(root, automaton, step2);
      expect(toRows(step3)).toEqual([[0, 1, 2]]);
    });

    test("Game of Life glider moves after 4 steps on large enough grid", async () => {
      const D = 0;
      const A = 1;

      const buildLife = (vi: ReturnType<typeof vivarium>) => {
        const dead = vi.element("dead", "#000000");
        const alive = vi.element("alive", "#ffffff");
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
});
