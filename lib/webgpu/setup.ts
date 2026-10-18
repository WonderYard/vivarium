import type { Automaton } from "@/automaton/types";
import {
  Accept,
  GpuCondition,
  GpuElement,
  GpuRule,
  Opcode,
  To,
  WORKGROUP_SIZE,
} from "@/common/constants";
import { mulberry32 } from "@/common/utils";
import { toPng as encodePng } from "@/png/png";
import { hash, randf } from "@typegpu/noise";
import tgpu, { common, type TgpuBindGroup } from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { compileGpuAutomaton, type GpuAutomaton } from "./compiler";

// A lost device can't be revived: forget it so the next setup creates a new root.
const watchDeviceLoss = async (device: GPUDevice): Promise<void> => {
  const info = await device.lost;
  gpuPromise = undefined;
  deviceLostListeners.forEach((listener) => listener(info));
  deviceLostListeners.clear();
};

// One root (and its device) and one set of compiled pipelines are shared by all simulations, as
// TypeGPU recommends. They're created on the first setup call, so importing the library
// works without WebGPU (SSR, Node, unsupported browsers).
const createGpu = async () => {
  try {
    const root = await tgpu.init();
    const pipeline = root.createComputePipeline({ compute });
    const clearPipeline = root.createComputePipeline({ compute: clearMissingElements });
    const renderPipeline = root.createRenderPipeline({
      vertex: common.fullScreenTriangle,
      fragment: render,
    });
    void watchDeviceLoss(root.device);

    return { root, pipeline, clearPipeline, renderPipeline };
  } catch (error) {
    // Let the next setup try again instead of failing forever.
    gpuPromise = undefined;
    throw error;
  }
};

let gpuPromise: ReturnType<typeof createGpu> | undefined;

// One per live simulation, removed by destroy(). Kept here rather than as a device.lost
// callback per simulation, which would keep a destroyed simulation in memory until the device dies.
const deviceLostListeners = new Set<(info: GPUDeviceLostInfo) => void>();

const getGpu = (): ReturnType<typeof createGpu> => (gpuPromise ??= createGpu());

// layouts are predefined
const gridLayout = tgpu.bindGroupLayout({
  dimensions: { uniform: d.vec2u },
  wrapping: { uniform: d.u32 },
  // x is the simulation seed, y the step number.
  seed: { uniform: d.vec2u },
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
  kindWords: { uniform: d.u32 },
  kindBits: {
    storage: d.arrayOf(d.u32),
    access: "readonly",
  },
});

// Because functions reference layouts and not groups,
// they can be defined once

const isOutOfBounds = (x: number, y: number) => {
  "use gpu";

  // No need to check if less than 0 because we are working in unsigned space.
  // In unsigned space, 0 - 1 = 4294967295, which is >= any reasonable dimension.
  return std.select(
    d.u32(0),
    d.u32(1),
    x >= gridLayout.$.dimensions.x || y >= gridLayout.$.dimensions.y,
  );
};

const inBoundsMask = (x: number, y: number) => {
  "use gpu";

  // Returns 1 when coordinates are valid (in bounds or wrapping mode),
  // 0 when out of bounds in non-wrapping mode.
  // wrapping=1: 1 - oob * (1 - 1) = 1 (always valid)
  // wrapping=0, in bounds: 1 - 0 * 1 = 1
  // wrapping=0, OOB: 1 - 1 * 1 = 0
  const oob = isOutOfBounds(x, y);
  return d.u32(1) - oob * (d.u32(1) - gridLayout.$.wrapping);
};

const pointToIndex = (x: number, y: number) => {
  "use gpu";

  const width = gridLayout.$.dimensions.x;
  const height = gridLayout.$.dimensions.y;

  // Neighbor offsets are at most 1 cell, so coordinates range from -1 to the dimension.
  // Adding the dimension before the modulo keeps -1 working in unsigned space:
  // 0 - 1 = 4294967295, and 4294967295 + width overflows to width - 1.
  // In wrapping mode this wraps OOB coordinates to the opposite edge correctly.
  // In non-wrapping mode, this leads to wrong indices at the edges,
  // but in this mode OOB results are discarded via inBoundsMask,
  // so the wrapped index is wrong, but harmless (it still falls within bounds).
  return ((y + height) % height) * width + ((x + width) % width);
};

const testNeighbor = (checkId: number, x: number, y: number) => {
  "use gpu";

  const selected = std.select(d.u32(0), d.u32(1), gridLayout.$.ids[pointToIndex(x, y)] === checkId);
  return selected * inBoundsMask(x, y);
};

const testKind = (kind: number, x: number, y: number) => {
  "use gpu";

  const id = gridLayout.$.ids[pointToIndex(x, y)];

  // Each element has a row of kindWords words in kindBits, with one bit per kind.
  const word = automatonLayout.$.kindBits[id * automatonLayout.$.kindWords + (kind >>> d.u32(5))];
  const selected = (word >>> (kind & d.u32(31))) & d.u32(1);
  return selected * inBoundsMask(x, y);
};

/**
 * Compare the occurrences of checkId in the neighborhood with count.
 * When the neighborhood is cross, only cardinal directions are counted.
 * When the neighborhood is square, all eight directions are counted.
 */
const checkIdCount = (x: number, y: number, checkId: number, packedCount: number) => {
  "use gpu";

  const crossCountMask =
    testNeighbor(checkId, x, y - 1) +
    testNeighbor(checkId, x - 1, y) +
    testNeighbor(checkId, x + 1, y) +
    testNeighbor(checkId, x, y + 1);

  const squareCountMask =
    testNeighbor(checkId, x - 1, y - 1) +
    testNeighbor(checkId, x + 1, y - 1) +
    testNeighbor(checkId, x - 1, y + 1) +
    testNeighbor(checkId, x + 1, y + 1);

  // neighborhood is 0 when cross, 1 when square
  const countMask = crossCountMask + automatonLayout.$.neighborhood * squareCountMask;

  // We select the bit in count using the number of occurrences as a mask
  // packedCount is representing a 9 bit array of flags
  // example: count = [2, 3] -> packedCount = 0b000001100
  // if mask is 3, it means we will check if the 4th LSB is a 1.
  // We are checking "!= 0u" and not "== 1u" because we are moving
  // the bit of the mask, and NOT the bit we are reading.
  return std.select(d.u32(0), d.u32(1), (packedCount & (d.u32(1) << countMask)) !== d.u32(0));
};

const checkKindCount = (x: number, y: number, kind: number, packedCount: number) => {
  "use gpu";

  const crossCountMask =
    testKind(kind, x, y - 1) +
    testKind(kind, x - 1, y) +
    testKind(kind, x + 1, y) +
    testKind(kind, x, y + 1);

  const squareCountMask =
    testKind(kind, x - 1, y - 1) +
    testKind(kind, x + 1, y - 1) +
    testKind(kind, x - 1, y + 1) +
    testKind(kind, x + 1, y + 1);

  const countMask = crossCountMask + automatonLayout.$.neighborhood * squareCountMask;

  return std.select(d.u32(0), d.u32(1), (packedCount & (d.u32(1) << countMask)) !== d.u32(0));
};

const checkPointCount = (x: number, y: number, checkPoint: d.v2u, packedCount: number) => {
  "use gpu";

  const px = x + checkPoint.x;
  const py = y + checkPoint.y;
  const pointIndex = pointToIndex(px, py);
  const checkId = gridLayout.$.ids[pointIndex];
  const selected = checkIdCount(x, y, checkId, packedCount);
  return selected * inBoundsMask(px, py);
};

const comparePointWithId = (x: number, y: number, comparePoint: d.v2u, withId: number) => {
  "use gpu";

  const cx = x + comparePoint.x;
  const cy = y + comparePoint.y;
  const comparePointIndex = pointToIndex(cx, cy);

  const selected = std.select(d.u32(0), d.u32(1), gridLayout.$.ids[comparePointIndex] === withId);
  return selected * inBoundsMask(cx, cy);
};

const comparePointWithKind = (x: number, y: number, comparePoint: d.v2u, kind: number) => {
  "use gpu";

  return testKind(kind, x + comparePoint.x, y + comparePoint.y);
};

const comparePointWithPoint = (x: number, y: number, comparePoint: d.v2u, withPoint: d.v2u) => {
  "use gpu";

  const cx = x + comparePoint.x;
  const cy = y + comparePoint.y;
  const wx = x + withPoint.x;
  const wy = y + withPoint.y;
  const comparePointIndex = pointToIndex(cx, cy);
  const withPointIndex = pointToIndex(wx, wy);

  const selected = std.select(
    d.u32(0),
    d.u32(1),
    gridLayout.$.ids[comparePointIndex] === gridLayout.$.ids[withPointIndex],
  );
  return selected * inBoundsMask(cx, cy) * inBoundsMask(wx, wy);
};

// also the main compute function has no variable dependencies
const compute = tgpu.computeFn({
  workgroupSize: WORKGROUP_SIZE,
  in: { pos: d.builtin.globalInvocationId },
})(({ pos }) => {
  const x = pos.x;
  const y = pos.y;

  // The last workgroups overhang the grid when its size isn't a multiple of WORKGROUP_SIZE.
  if (x >= gridLayout.$.dimensions.x || y >= gridLayout.$.dimensions.y) {
    return;
  }

  const index = pointToIndex(x, y);

  const id = gridLayout.$.ids[index];

  const element = automatonLayout.$.elements[id];

  // Seed once per invocation so that each CHANCE condition draws a fresh sample.
  // The hash gives each step of a seed a distinct u32, passed as two 16-bit halves that f32 holds
  // exactly: the step itself as an f32 would repeat after 2^24 steps.
  const stepSeed = hash(gridLayout.$.seed.y) ^ gridLayout.$.seed.x;
  randf.seed4(
    d.vec4f(d.vec2f(pos.xy), d.f32(stepSeed & d.u32(0xffff)), d.f32(stepSeed >>> d.u32(16))),
  );

  const ruleStart = element.ruleStart;
  const ruleEnd = element.ruleEnd;

  for (let i = ruleStart; i < ruleEnd; i++) {
    const rule = automatonLayout.$.rules[i];
    const accept = rule.accept as Accept;

    let passing = d.u32(0);

    const conditionsStart = rule.conditionsStart;
    const conditionsEnd = rule.conditionsEnd;
    const conditionsCount = conditionsEnd - conditionsStart;

    for (let j = conditionsStart; j < conditionsEnd; j++) {
      const condition = automatonLayout.$.conditions[j];
      const opcode = condition.opcode as Opcode;

      switch (opcode) {
        case Opcode.COUNT_ELEMENT: {
          passing += checkIdCount(x, y, condition.checkId, condition.countOrWithId);
          break;
        }
        case Opcode.COUNT_POINT: {
          passing += checkPointCount(
            x,
            y,
            condition.checkPointOrComparePoint,
            condition.countOrWithId,
          );
          break;
        }
        case Opcode.COUNT_KIND: {
          passing += checkKindCount(x, y, condition.checkId, condition.countOrWithId);
          break;
        }
        case Opcode.IS_ELEMENT: {
          passing += comparePointWithId(
            x,
            y,
            condition.checkPointOrComparePoint,
            condition.countOrWithId,
          );
          break;
        }
        case Opcode.IS_POINT: {
          passing += comparePointWithPoint(
            x,
            y,
            condition.checkPointOrComparePoint,
            condition.withPoint,
          );
          break;
        }
        case Opcode.IS_KIND: {
          passing += comparePointWithKind(
            x,
            y,
            condition.checkPointOrComparePoint,
            condition.countOrWithId,
          );
          break;
        }
        case Opcode.CHANCE: {
          passing += std.select(d.u32(0), d.u32(1), randf.sample() < condition.chance);
          break;
        }
      }
    }

    if (
      (accept === Accept.ALL && passing === conditionsCount) ||
      (accept === Accept.ANY && passing >= d.u32(1)) ||
      (accept === Accept.ONE && passing === d.u32(1)) ||
      (accept === Accept.NONE && passing === d.u32(0))
    ) {
      let resolvedId = rule.toId;

      const toType = rule.toType as To;

      if (toType === To.POINT) {
        const nx = x + rule.toNeighbor.x;
        const ny = y + rule.toNeighbor.y;
        // pointToIndex always returns a valid array index, even for OOB coordinates.
        // When OOB in non-wrapping mode, mask is 0, so the read value is multiplied away.
        const pointIndex = pointToIndex(nx, ny);
        const mask = inBoundsMask(nx, ny);
        // When OOB in non-wrapping mode, keep the current id instead.
        resolvedId = gridLayout.$.ids[pointIndex] * mask + id * (d.u32(1) - mask);
      }

      gridLayout.$.newIds[index] = resolvedId;

      return;
    }
  }

  gridLayout.$.newIds[index] = id;
});

// Sets cells whose element index is past the end of the elements array to element 0.
const clearMissingElements = tgpu.computeFn({
  workgroupSize: WORKGROUP_SIZE,
  in: { pos: d.builtin.globalInvocationId },
})(({ pos }) => {
  if (pos.x >= gridLayout.$.dimensions.x || pos.y >= gridLayout.$.dimensions.y) {
    return;
  }

  const index = pointToIndex(pos.x, pos.y);

  if (gridLayout.$.ids[index] >= std.arrayLength(automatonLayout.$.elements)) {
    gridLayout.$.ids[index] = d.u32(0);
  }
});

// Draws the grid stretched over the whole canvas by looking up each cell's element color.
const render = tgpu.fragmentFn({
  in: { uv: d.vec2f },
  out: d.vec4f,
})(({ uv }) => {
  const dimensions = gridLayout.$.dimensions;
  // uv goes from (0, 0) at the top left to (1, 1) at the bottom right; min guards against rounding up at 1.
  const cell = std.min(d.vec2u(std.mul(uv, d.vec2f(dimensions))), std.sub(dimensions, d.vec2u(1)));
  const index = cell.y * dimensions.x + cell.x;
  const id = gridLayout.$.ids[index];

  // Colors are packed as ABGR, so the lowest byte is red.
  return std.unpack4x8unorm(automatonLayout.$.elements[id].color);
});

// Cells hold element indices, so initialGrid, writeCell and writeGrid accept the same ones.
const isElementIndex = (value: number, elementCount: number): boolean =>
  Number.isInteger(value) && value >= 0 && value < elementCount;

/**
 * Initializes the WebGPU simulation for a given canvas and automaton. The grid is randomly initialized
 * with the defined elements, each equally likely, unless an `initialGrid` gives the cells or the weights.
 *
 * Async because the first call initializes WebGPU, which later simulations share.
 * If the device is lost, the next call requests a new one.
 *
 * Use `index` found in `ElementBlueprint` instances to reference elements when building the initialGrid.
 *
 * @param options - An object containing the `canvas` element and the compiled `automaton`.
 * @param options.canvas - The HTML canvas element. By default, its `width` and `height` define the grid dimensions. The simulation renders to it through a `webgpu` context, so the canvas must not already have a `2d` or `webgl` context.
 * @param options.width - The number of grid columns, any positive integer. Defaults to `canvas.width`. `width * height` can be at most 2^25 (about 5792×5792), and `width` and `height` at most 524280 each; larger grids throw. The grid is stretched over the canvas, so a small grid on a large canvas draws each cell as a block of pixels.
 * @param options.height - The number of grid rows, any positive integer. Defaults to `canvas.height`.
 * @param options.automaton - The compiled automaton produced by {@link VivariumBlueprint.create | vivarium().create()}.
 * @param options.initialGrid - An optional 2d array of element indices to use instead of random initialization. Alternatively, you can pass a flat (1d) array with one index per cell, row-major order.
 * Or pass `{ weights }` to keep random initialization with element names mapped to relative weights: `{ weights: { empty: 9, tree: 1 } }`
 * fills about 90% of the cells with `empty` and 10% with `tree`, same as `{ empty: 90, tree: 10 }`. Elements left out are never picked.
 * Prefer taking the names from `ElementBlueprint.name`, e.g. `{ weights: { [empty.name]: 9, [tree.name]: 1 } }`, so renaming an element can't break the weights.
 * @param options.seed - An integer in [0, 2^32) that makes the run reproducible: the same seed, automaton, options and calls give the same random initial grid and the same `chance` outcomes at every step, on any device and browser. A new vivarium version may change the outcomes of a seed. Defaults to a random one, readable as `seed` on the result to replay the run.
 * @param options.onDeviceLost - Called if the GPU device is lost (e.g. driver crash or reset). The simulation then behaves as after `destroy` and its grid is gone; call `setup` again to recreate it on a new device. To resume, pass a grid saved earlier with `readGrid` as `initialGrid`.
 * @throws If the automaton is not one {@link VivariumBlueprint.create | create()} could have made, apart from its ids (which only need to be unique) and elements sharing a color. The error names the first bad field, e.g. `rules[0].when[1].count[2]`.
 * @returns A promise of an object with:
 * - `update` — advances the simulation by one step (synchronous, GPU only).
 * - `draw` — renders the latest state to the canvas (synchronous, GPU only).
 * - `evolve` — convenience shorthand for `update()` + `draw()`.
 * - `readGrid` — reads the current grid state.
 * - `writeCell` — updates a single cell by flat index.
 * - `writeCellAt` — updates a single cell by grid coordinates.
 * - `writeGrid` — overwrites the entire grid.
 * - `setAutomaton` — replaces the automaton. Cells whose element index the new automaton doesn't have become element 0.
 * - `toPng` — saves the current grid and automaton as a PNG, loadable with {@link fromPng}.
 * - `destroy` — frees the GPU resources; afterwards `update`, `draw` and `evolve` do nothing and the other functions throw.
 * - `seed` — the seed of this run, the given one or the random default.
 */
export const setup = async ({
  canvas,
  width = canvas.width,
  height = canvas.height,
  automaton,
  initialGrid,
  seed = Math.floor(Math.random() * 2 ** 32),
  onDeviceLost,
}: {
  canvas: HTMLCanvasElement;
  width?: number;
  height?: number;
  automaton: Automaton;
  initialGrid?: number[] | number[][] | { weights: Record<string, number> };
  seed?: number;
  onDeviceLost?: (info: GPUDeviceLostInfo) => void;
}) => {
  // Run every check before the first allocation, so a throw leaks no GPU buffer.
  if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1) {
    throw new Error(`Width and height must be positive integers, but got ${width}x${height}.`);
  }

  if (!Number.isInteger(seed) || seed < 0 || seed >= 2 ** 32) {
    throw new Error(`Seed ${seed} is invalid. Expected an integer in range [0, 2^32).`);
  }

  const random = mulberry32(seed);

  const flatGrid = Array.isArray(initialGrid) ? initialGrid.flat() : undefined;

  if (flatGrid !== undefined && flatGrid.length !== width * height) {
    throw new Error(
      `initialGrid length ${flatGrid.length} does not match the expected length of ${width * height} (width ${width} x height ${height}).`,
    );
  }

  const ids = new Uint32Array(width * height);

  const elementsLength = automaton.elements.length;

  // Running totals of the weights, by element index: a cell picks the first one above a random point in [0, total).
  const cumulativeWeights: number[] = [];
  let totalWeight = 0;

  if (initialGrid !== undefined && !Array.isArray(initialGrid)) {
    const { weights } = initialGrid;

    for (const name of Object.keys(weights)) {
      if (!automaton.elements.some((element) => element.name === name)) {
        throw new Error(
          `Weight given for "${name}", but the automaton has no element with that name.`,
        );
      }
    }

    for (const { name } of automaton.elements) {
      // hasOwn so an element named like an Object.prototype key (e.g. toString) reads as left out.
      const weight = Object.hasOwn(weights, name) ? weights[name] : 0;

      if (!Number.isFinite(weight) || weight < 0) {
        throw new Error(`Weight ${weight} of "${name}" is invalid. Expected a finite number >= 0.`);
      }

      totalWeight += weight;
      cumulativeWeights.push(totalWeight);
    }

    if (totalWeight <= 0) {
      throw new Error("Weights must add up to more than 0.");
    }
  }

  for (let i = 0; i < ids.length; i++) {
    let elementIndex: number;

    if (flatGrid !== undefined) {
      const rawIndex = flatGrid[i] ?? 0;

      if (!isElementIndex(rawIndex, elementsLength)) {
        throw new Error(
          `Element index ${rawIndex} from initialGrid is invalid. Expected a finite integer in range [0, ${elementsLength - 1}]. Make sure to use ElementBlueprint.index to build the initialGrid with pre-existing indices.`,
        );
      }

      elementIndex = rawIndex;
    } else if (totalWeight > 0) {
      const point = random() * totalWeight;
      elementIndex = cumulativeWeights.findIndex((weight) => point < weight);
    } else {
      elementIndex = Math.floor(random() * elementsLength);
    }
    ids[i] = elementIndex;
  }

  const gpuAutomaton = compileGpuAutomaton(automaton);

  const { root, pipeline, clearPipeline, renderPipeline } = await getGpu();
  const { device } = root;

  // Past this, WebGPU rejects the grid's bind groups without throwing, and every update silently
  // zeroes the grid. The device is requested with default limits: 128 MiB, about 33.5M cells.
  const maxCells = Math.floor(
    Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize) / 4,
  );

  if (width * height > maxCells) {
    throw new Error(
      `Grid ${width}x${height} has ${width * height} cells, but this device allows at most ${maxCells}.`,
    );
  }

  // Past this, the update's dispatch is rejected without throwing, and the grid never changes.
  // Default limits allow 65535 workgroups per dimension: 524280 cells of width or height.
  const maxWorkgroups = device.limits.maxComputeWorkgroupsPerDimension;
  const maxWidth = maxWorkgroups * WORKGROUP_SIZE[0];
  const maxHeight = maxWorkgroups * WORKGROUP_SIZE[1];

  if (width > maxWidth || height > maxHeight) {
    throw new Error(
      `Grid ${width}x${height} is too wide or tall: this device allows at most ${maxWidth} columns and ${maxHeight} rows.`,
    );
  }

  let frames = 0;
  let destroyed = false;

  // update, draw and evolve return early instead, so a loop still running doesn't crash.
  const assertNotDestroyed = (): void => {
    if (destroyed) {
      throw new Error(
        "Simulation is destroyed or its device was lost. Call setup again to create a new one.",
      );
    }
  };

  const context = root.configureContext({ canvas, alphaMode: "opaque" });

  const WORKGROUP_COUNT_W = Math.ceil(width / WORKGROUP_SIZE[0]);
  const WORKGROUP_COUNT_H = Math.ceil(height / WORKGROUP_SIZE[1]);

  // Define all the buffers. Creating them depends on width and height only.
  // When automaton changes we don't need to recreate them.

  const seedBuffer = root.createUniform(d.vec2u);

  const dimensions = root.createUniform(d.vec2u, [width, height]);

  const wrappingBuffer = root.createUniform(d.u32, automaton.wrapping ? 1 : 0);

  const ids0 = root.createBuffer(d.arrayOf(d.u32, width * height)).$usage("storage");

  const ids1 = root.createBuffer(d.arrayOf(d.u32, width * height)).$usage("storage");

  const gridGroup0 = root.createBindGroup(gridLayout, {
    dimensions,
    wrapping: wrappingBuffer,
    seed: seedBuffer,
    ids: ids0,
    newIds: ids1,
  });

  const gridGroup1 = root.createBindGroup(gridLayout, {
    dimensions,
    wrapping: wrappingBuffer,
    seed: seedBuffer,
    ids: ids1,
    newIds: ids0,
  });

  // Each update reads the current grid, writes the other one and increments `frames`,
  // so the parity of `frames` tells which one holds the latest state.
  const grid0 = { ids: ids0, group: gridGroup0 };
  const grid1 = { ids: ids1, group: gridGroup1 };
  const currentGrid = () => (frames % 2 === 0 ? grid0 : grid1);

  // using let because we can reassign this when we update the automaton
  let automatonGroup: TgpuBindGroup;
  let automatonBuffers: { destroy: () => void }[] = [];
  let elementCount = 0;
  let currentAutomaton = automaton;

  /**
   * Replaces the automaton, keeping the grid.
   *
   * Cells refer to elements by index, so the new automaton's element at index `i` takes over
   * every cell holding `i`. When it has fewer elements, cells holding an index it doesn't have
   * are set to element 0, so the grid stays valid for `readGrid`, `writeGrid` and `toPng`.
   * Adding the elements back later doesn't restore those cells.
   *
   * To load a grid made for another automaton (e.g. from {@link fromPng}), call `setAutomaton`
   * before `writeGrid`, which only accepts indices of the current automaton.
   *
   * @throws If the automaton is invalid, as `setup` does, leaving the current one running.
   */
  const setAutomaton = ({ automaton }: { automaton: Automaton }): void => {
    assertNotDestroyed();

    applyAutomaton(automaton, compileGpuAutomaton(automaton));
  };

  // setup calls it directly with the automaton it compiled before allocating anything.
  const applyAutomaton = (
    automaton: Automaton,
    {
      gpuNeighborhood,
      gpuElements,
      gpuRules,
      gpuConditions,
      gpuKindWords,
      gpuKindBits,
    }: GpuAutomaton,
  ): void => {
    const previousElementCount = elementCount;
    elementCount = gpuElements.length;

    const rules = gpuRules.length > 0 ? gpuRules : [GpuRule()];
    const conditions = gpuConditions.length > 0 ? gpuConditions : [GpuCondition()];

    const neighborhood = root.createUniform(d.u32, gpuNeighborhood);
    const elements = root.createReadonly(
      d.arrayOf(GpuElement, Math.max(gpuElements.length, 1)),
      gpuElements,
    );
    const rulesBuffer = root.createReadonly(d.arrayOf(GpuRule, rules.length), rules);
    const conditionsBuffer = root.createReadonly(
      d.arrayOf(GpuCondition, conditions.length),
      conditions,
    );
    const kindWords = root.createUniform(d.u32, gpuKindWords);
    const kindBits = root.createReadonly(
      d.arrayOf(d.u32, Math.max(gpuKindBits.length, 1)),
      gpuKindBits,
    );

    automatonGroup = root.createBindGroup(automatonLayout, {
      neighborhood,
      elements,
      rules: rulesBuffer,
      conditions: conditionsBuffer,
      kindWords,
      kindBits,
    });

    // Work already submitted with the old buffers still completes after destroy().
    automatonBuffers.forEach((buffer) => buffer.destroy());
    automatonBuffers = [
      neighborhood.buffer,
      elements.buffer,
      rulesBuffer.buffer,
      conditionsBuffer.buffer,
      kindWords.buffer,
      kindBits.buffer,
    ];

    wrappingBuffer.write(automaton.wrapping ? 1 : 0);
    currentAutomaton = automaton;

    if (elementCount < previousElementCount) {
      clearPipeline
        .with(automatonGroup)
        .with(currentGrid().group)
        .dispatchWorkgroups(WORKGROUP_COUNT_W, WORKGROUP_COUNT_H);
    }
  };

  applyAutomaton(automaton, gpuAutomaton);

  ids0.write(ids);

  /**
   * Reads the current grid state from the GPU and returns a flat array of element indices.
   */
  const readGrid = async (): Promise<number[]> => {
    assertNotDestroyed();

    return currentGrid().ids.read();
  };

  /**
   * Writes a single cell in the current grid.
   *
   * @param index - The flat index of the cell to update (row-major order).
   * @param elementIndex - The element index to assign to the cell.
   */
  const writeCell = (index: number, elementIndex: number): void => {
    assertNotDestroyed();

    if (!Number.isInteger(index) || index < 0 || index >= width * height) {
      throw new Error(
        `Cell index ${index} is out of bounds. Expected an integer in range [0, ${width * height - 1}].`,
      );
    }

    if (!isElementIndex(elementIndex, elementCount)) {
      throw new Error(
        `Element index ${elementIndex} is invalid. Expected a finite integer in range [0, ${elementCount - 1}].`,
      );
    }

    // Raw write instead of TypeGPU's write/patch: these keep a CPU copy of the whole grid
    // (4 bytes per cell, per buffer) only to stage the bytes, and patch was ~4x slower per call
    // (typegpu 0.12.6). A u32 array needs no padding, so the raw bytes are the layout.
    device.queue.writeBuffer(currentGrid().ids.buffer, index * 4, new Uint32Array([elementIndex]));
  };

  /**
   * Writes a single cell by grid coordinates.
   *
   * @param x - The column of the cell (0-based, from left).
   * @param y - The row of the cell (0-based, from top).
   * @param elementIndex - The element index to assign to the cell.
   */
  const writeCellAt = (x: number, y: number, elementIndex: number): void => {
    if (!Number.isInteger(x) || x < 0 || x >= width) {
      throw new Error(
        `Column ${x} is out of bounds. Expected an integer in range [0, ${width - 1}].`,
      );
    }

    if (!Number.isInteger(y) || y < 0 || y >= height) {
      throw new Error(
        `Row ${y} is out of bounds. Expected an integer in range [0, ${height - 1}].`,
      );
    }

    writeCell(y * width + x, elementIndex);
  };

  /**
   * Overwrites the entire grid with the given flat array of element indices.
   * Useful for restoring a snapshot or painting the grid in bulk.
   *
   * @param grid - A flat array of element indices (row-major order) whose length must equal `width * height`.
   */
  const writeGrid = (grid: number[]): void => {
    assertNotDestroyed();

    const totalCells = width * height;

    if (grid.length !== totalCells) {
      throw new Error(
        `Grid length ${grid.length} does not match the expected length of ${totalCells}.`,
      );
    }

    const newIds = new Uint32Array(totalCells);

    for (let i = 0; i < totalCells; i++) {
      const elementIndex = grid[i];

      if (!isElementIndex(elementIndex, elementCount)) {
        throw new Error(
          `Element index ${elementIndex} at position ${i} is invalid. Expected a finite integer in range [0, ${elementCount - 1}].`,
        );
      }

      newIds[i] = elementIndex;
    }

    // Raw write for the same reason as in writeCell.
    device.queue.writeBuffer(currentGrid().ids.buffer, 0, newIds);
  };

  /**
   * Saves the current grid and automaton as a PNG, one pixel per cell.
   * Load it back with {@link fromPng}. Rejects if two elements share a color.
   */
  const toPng = async (): Promise<Blob> =>
    encodePng({ automaton: currentAutomaton, grid: await readGrid(), width, height });

  /**
   * Advances the simulation by one step on the GPU. This function is synchronous
   * — it only queues GPU commands without waiting for them to complete.
   *
   * Call `update` as many times as needed to run multiple simulation steps,
   * then call `draw` once to render the latest state to the canvas.
   */
  const update = (): void => {
    if (destroyed) {
      return;
    }

    seedBuffer.write([seed, frames >>> 0]);

    pipeline
      .with(automatonGroup)
      .with(currentGrid().group)
      .dispatchWorkgroups(WORKGROUP_COUNT_W, WORKGROUP_COUNT_H);

    frames++;
  };

  /**
   * Renders the latest simulation state to the canvas. Like `update`, it is synchronous:
   * it only queues GPU commands, and nothing is read back to the CPU.
   */
  const draw = (): void => {
    if (destroyed) {
      return;
    }

    renderPipeline
      .with(automatonGroup)
      .with(currentGrid().group)
      .withColorAttachment({ view: context })
      .draw(3);
  };

  /**
   * Convenience function that advances the simulation by one step and draws
   * the result. Equivalent to calling `update()` followed by `draw()`.
   */
  const evolve = (): void => {
    update();
    draw();
  };

  /**
   * Frees the GPU resources owned by this simulation. Afterwards `update`, `draw`
   * and `evolve` do nothing, so a loop still running (e.g. during a React effect
   * cleanup) doesn't crash, while reads, writes, `setAutomaton` and `toPng` throw.
   * Calls made before it still complete: a pending `readGrid` or `toPng` resolves
   * with the grid as it was. Calling it more than once is safe.
   */
  const destroy = (): void => {
    if (destroyed) {
      return;
    }
    destroyed = true;
    deviceLostListeners.delete(handleDeviceLost);

    // The root is shared, so it stays alive; free only this simulation's buffers.
    [
      dimensions.buffer,
      wrappingBuffer.buffer,
      seedBuffer.buffer,
      ids0,
      ids1,
      ...automatonBuffers,
    ].forEach((buffer) => buffer.destroy());
  };

  const handleDeviceLost = (info: GPUDeviceLostInfo): void => {
    // Every resource of a lost device is gone, so the simulation is as good as destroyed.
    destroyed = true;
    onDeviceLost?.(info);
  };

  deviceLostListeners.add(handleDeviceLost);

  return {
    update,
    draw,
    evolve,
    readGrid,
    writeCell,
    writeCellAt,
    writeGrid,
    setAutomaton,
    toPng,
    destroy,
    seed,
  };
};
