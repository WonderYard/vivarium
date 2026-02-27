import type { Automaton } from "@/automaton/types";
import {
  Accept,
  GpuCondition,
  GpuElement,
  GpuRule,
  MIN_GRID_SIZE,
  Opcode,
  To,
  WORKGROUP_SIZE,
} from "@/common/constants";
import { randf } from "@typegpu/noise";
import tgpu, { type TgpuBindGroup, type TgpuUniform } from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { compileGpuAutomaton } from "./compiler";

const adapter = await navigator.gpu.requestAdapter();

const device = await adapter?.requestDevice();

void device?.lost.then(() => {
  throw new Error("Device lost");
});

let frames = 0;
let seed: TgpuUniform<d.F32>;

export const setSeed = (s: TgpuUniform<d.F32>) => {
  seed = s;
};

// layouts are predefined
export const gridLayout = tgpu.bindGroupLayout({
  dimensions: { uniform: d.vec2u },
  wrapping: { uniform: d.u32 },
  colors: { storage: d.arrayOf(d.u32), access: "mutable" },
  newColors: { storage: d.arrayOf(d.u32), access: "mutable" },
  ids: { storage: d.arrayOf(d.u32), access: "mutable" },
  newIds: { storage: d.arrayOf(d.u32), access: "mutable" },
});

export const automatonLayout = tgpu.bindGroupLayout({
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

  const widthMask = gridLayout.$.dimensions.x - d.u32(1);
  const heightMask = gridLayout.$.dimensions.y - d.u32(1);

  // Bitwise AND with (dim - 1) only performs correct modulo when dim is a power of 2
  // (the mask is then all-ones in the lower bits, e.g. 1024 - 1 = 0x3FF).
  // In wrapping mode dimensions are guaranteed to be powers of 2, so this is safe.
  // Example: 0 - 1 = 4294967295 in unsigned space, and (0 - 1) & 1023 = 1023.
  // WARNING: in non-wrapping mode dimensions may NOT be powers of 2.
  // When they aren't, the bitmask produces wrong indices even for in-bounds
  // coordinates (not only OOB ones). All existing tests use power-of-2 grids,
  // so this hasn't been caught yet. A proper fix would branch on wrapping and
  // use std.min-based clamping for non-wrapping mode.
  return (y & heightMask) * gridLayout.$.dimensions.x + (x & widthMask);
};

const testNeighbor = (checkId: number, x: number, y: number) => {
  "use gpu";

  const selected = std.select(d.u32(0), d.u32(1), gridLayout.$.ids[pointToIndex(x, y)] === checkId);
  return selected * inBoundsMask(x, y);
};

const testIdInPack = (packedIds: number, x: number, y: number) => {
  "use gpu";

  const idMask = gridLayout.$.ids[pointToIndex(x, y)];

  // check if id is in the bits
  const selected = std.select(
    d.u32(0),
    d.u32(1),
    // Note: in unsigned space if idMask is > 31 it's gonna loop back to 0,
    // so we cannot allow ids greater than 31 here. However no error is thrown,
    // so to keep gpu logic simple we do the check during the compile step.
    (packedIds & (d.u32(1) << idMask)) !== d.u32(0),
  );
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

const checkIdsCount = (x: number, y: number, packedIds: number, packedCount: number) => {
  "use gpu";

  const crossCountMask =
    testIdInPack(packedIds, x, y - 1) +
    testIdInPack(packedIds, x - 1, y) +
    testIdInPack(packedIds, x + 1, y) +
    testIdInPack(packedIds, x, y + 1);

  const squareCountMask =
    testIdInPack(packedIds, x - 1, y - 1) +
    testIdInPack(packedIds, x + 1, y - 1) +
    testIdInPack(packedIds, x - 1, y + 1) +
    testIdInPack(packedIds, x + 1, y + 1);

  const countMask = crossCountMask + automatonLayout.$.neighborhood * squareCountMask;

  return std.select(d.u32(0), d.u32(1), (packedCount & (d.u32(1) << countMask)) !== d.u32(0));
};

const checkPointCount = (x: number, y: number, checkPoint: d.v2u, packedCount: number) => {
  "use gpu";

  const px = x + d.u32(checkPoint.x);
  const py = y + d.u32(checkPoint.y);
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

const comparePointWithKindId = (x: number, y: number, comparePoint: d.v2u, packedIds: number) => {
  "use gpu";

  return testIdInPack(packedIds, x + comparePoint.x, y + comparePoint.y);
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
export const compute = tgpu.computeFn({
  workgroupSize: WORKGROUP_SIZE,
  in: { pos: d.builtin.globalInvocationId },
})(({ pos }) => {
  const x = pos.x;
  const y = pos.y;

  const index = pointToIndex(x, y);

  const color = gridLayout.$.colors[index];
  const id = gridLayout.$.ids[index];

  const element = automatonLayout.$.elements[id];

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

      if (opcode === Opcode.COUNT_ELEMENT) {
        const checkId = condition.checkId;
        const count = condition.countOrWithId;
        passing += checkIdCount(x, y, checkId, count);
      } else if (opcode === Opcode.COUNT_POINT) {
        const checkPoint = condition.checkPointOrComparePoint;
        const count = condition.countOrWithId;
        passing += checkPointCount(x, y, checkPoint, count);
      } else if (opcode === Opcode.COUNT_KIND) {
        const packedIds = condition.checkId;
        const count = condition.countOrWithId;
        passing += checkIdsCount(x, y, packedIds, count);
      } else if (opcode === Opcode.IS_ELEMENT) {
        const comparePoint = condition.checkPointOrComparePoint;
        const withId = condition.countOrWithId;
        passing += comparePointWithId(x, y, comparePoint, withId);
      } else if (opcode === Opcode.IS_POINT) {
        const comparePoint = condition.checkPointOrComparePoint;
        const withPoint = condition.withPoint;
        passing += comparePointWithPoint(x, y, comparePoint, withPoint);
      } else if (opcode === Opcode.IS_KIND) {
        const comparePoint = condition.checkPointOrComparePoint;
        const packedIds = condition.countOrWithId;
        passing += comparePointWithKindId(x, y, comparePoint, packedIds);
      } else if (opcode === Opcode.CHANCE) {
        const chance = condition.chance;
        randf.seed3(d.vec3f(std.div(d.vec2f(pos.xy), d.vec2f(gridLayout.$.dimensions.xy)), seed.$));
        passing += std.select(d.u32(0), d.u32(1), randf.sample() < chance);
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
        const nx = x + d.u32(rule.toNeighbor.x);
        const ny = y + d.u32(rule.toNeighbor.y);
        // pointToIndex always returns a valid array index, even for OOB coordinates.
        // When OOB in non-wrapping mode, mask is 0, so the read value is multiplied away.
        const pointIndex = pointToIndex(nx, ny);
        const mask = inBoundsMask(nx, ny);
        // When OOB in non-wrapping mode, keep the current id instead.
        resolvedId = gridLayout.$.ids[pointIndex] * mask + id * (d.u32(1) - mask);
      }

      gridLayout.$.newIds[index] = resolvedId;
      gridLayout.$.newColors[index] = automatonLayout.$.elements[resolvedId].color;

      return;
    }
  }

  gridLayout.$.newIds[index] = id;
  gridLayout.$.newColors[index] = color;
});

/**
 * Initializes the WebGPU simulation for a given canvas and automaton. The grid is randomly initialized
 * with the defined elements unless an `initialGrid` is provided.
 *
 * Use `index` found in `ElementBlueprint` instances to reference elements when building the initialGrid.
 *
 * @param options - An object containing the `canvas` element and the compiled `automaton`.
 * @param options.canvas - The HTML canvas element. Its `width` and `height` define the grid dimensions.
 * @param options.automaton - The compiled automaton produced by {@link VivariumBlueprint.create | vivarium().create()}.
 * @param options.initialGrid - An optional 2d array of element indices to use instead of random initialization. Alternatively, you can pass a flat (1d) array with one index per cell, row-major order.
 * @returns An object with:
 * - `update` — advances the simulation by one step (synchronous, GPU only).
 * - `draw` — reads the latest GPU state and renders it to the canvas (async).
 * - `evolve` — convenience shorthand for `update()` + `await draw()`.
 * - `readGrid` — reads the current grid state.
 * - `writeCell` — updates a single cell by flat index.
 * - `writeCellAt` — updates a single cell by grid coordinates.
 * - `writeGrid` — overwrites the entire grid.
 * - `setAutomaton` — replaces the automaton rules.
 * - `tgpuRoot` — the underlying TypeGPU root.
 */
export const setup = ({
  canvas,
  automaton,
  initialGrid,
}: {
  canvas: HTMLCanvasElement;
  automaton: Automaton;
  initialGrid?: number[] | number[][];
}) => {
  if (!device) {
    throw Error("No adapter");
  }

  const root = tgpu.initFromDevice({ device });
  const pipeline = root.createComputePipeline({ compute });

  frames = 0;
  seed = root.createUniform(d.f32);

  const { width, height } = canvas;
  const g = canvas.getContext("2d") as CanvasRenderingContext2D;

  const isPowerOf2 = (n: number) => n > 0 && (n & (n - 1)) === 0;

  if (width < MIN_GRID_SIZE || height < MIN_GRID_SIZE) {
    throw new Error(
      `Minimum values for width and height must be >= ${MIN_GRID_SIZE}, but got ${width}x${height}.`,
    );
  }

  if (automaton.wrapping && (!isPowerOf2(width) || !isPowerOf2(height))) {
    throw new Error(
      `Wrapping mode requires width and height to be powers of 2, but got ${width}x${height}.`,
    );
  }

  const flatGrid = initialGrid?.flat();

  if (flatGrid !== undefined && flatGrid.length !== width * height) {
    throw new Error(
      `initialGrid length ${flatGrid.length} does not match the expected length of ${width * height} (width ${width} x height ${height}).`,
    );
  }

  const WORKGROUP_COUNT_W = Math.ceil(width / WORKGROUP_SIZE[0]);
  const WORKGROUP_COUNT_H = Math.ceil(height / WORKGROUP_SIZE[1]);

  // Define all the buffers. Creating them depends on width and height only.
  // When automaton changes we don't need to recreate them.

  const dimensions = root.createBuffer(d.vec2u, d.vec2u(width, height)).$usage("uniform");

  const wrappingBuffer = root.createBuffer(d.u32, automaton.wrapping ? 1 : 0).$usage("uniform");

  const colors0 = root.createBuffer(d.arrayOf(d.u32, width * height)).$usage("storage");

  const colors1 = root.createBuffer(d.arrayOf(d.u32, width * height)).$usage("storage");

  const ids0 = root.createBuffer(d.arrayOf(d.u32, width * height)).$usage("storage");

  const ids1 = root.createBuffer(d.arrayOf(d.u32, width * height)).$usage("storage");

  const colorsStagingBuffer = root
    .createBuffer(d.arrayOf(d.u32, width * height))
    .$addFlags(GPUBufferUsage.MAP_READ);

  // no staging buffer needed for ids, since we only need to draw the result

  const gridGroup0 = root.createBindGroup(gridLayout, {
    dimensions,
    wrapping: wrappingBuffer,
    colors: colors0,
    newColors: colors1,
    ids: ids0,
    newIds: ids1,
  });

  const gridGroup1 = root.createBindGroup(gridLayout, {
    dimensions,
    wrapping: wrappingBuffer,
    colors: colors1,
    newColors: colors0,
    ids: ids1,
    newIds: ids0,
  });

  // using let because we can reassign this when we update the automaton
  let automatonGroup: TgpuBindGroup;
  let palette: number[] = [];

  const setAutomaton = ({ automaton }: { automaton: Automaton }): void => {
    const { gpuNeighborhood, gpuElements, gpuRules, gpuConditions } =
      compileGpuAutomaton(automaton);

    palette = gpuElements.map((element) => element.color);

    const rules = gpuRules.length > 0 ? gpuRules : [GpuRule()];
    const conditions = gpuConditions.length > 0 ? gpuConditions : [GpuCondition()];

    automatonGroup = root.createBindGroup(automatonLayout, {
      neighborhood: root.createBuffer(d.u32, gpuNeighborhood).$usage("uniform"),
      elements: root
        .createBuffer(d.arrayOf(GpuElement, Math.max(gpuElements.length, 1)), gpuElements)
        .$usage("storage"),
      rules: root.createBuffer(d.arrayOf(GpuRule, rules.length), rules).$usage("storage"),
      conditions: root
        .createBuffer(d.arrayOf(GpuCondition, conditions.length), conditions)
        .$usage("storage"),
    });
  };

  setAutomaton({ automaton });

  const colors = new Uint32Array(width * height);
  const ids = new Uint32Array(width * height);

  const elementsLength = automaton.elements.length;

  for (let i = 0; i < colors.length; i++) {
    let elementIndex: number;

    if (flatGrid !== undefined) {
      const rawIndex = flatGrid[i] ?? 0;

      if (
        !Number.isFinite(rawIndex) ||
        !Number.isInteger(rawIndex) ||
        rawIndex < 0 ||
        rawIndex >= elementsLength
      ) {
        throw new Error(
          `Element index ${rawIndex} from initialGrid is invalid. Expected a finite integer in range [0, ${elementsLength - 1}]. Make sure to use ElementBlueprint.index to build the initialGrid with pre-existing indices.`,
        );
      }

      elementIndex = rawIndex;
    } else {
      elementIndex = Math.floor(Math.random() * elementsLength);
    }
    colors[i] = palette[elementIndex];
    ids[i] = elementIndex;
  }

  // and we write the inizialization to the buffers
  colors0.write(Array.from(colors));
  ids0.write(Array.from(ids));

  const imageData = g.createImageData(width, height);
  const pixels = new Uint32Array(imageData.data.buffer);

  /**
   * Reads the current grid state from the GPU and returns a flat array of element indices.
   */
  const readGrid = async (): Promise<number[]> => {
    const currentIds = frames % 2 === 0 ? ids0 : ids1;
    const result = await currentIds.read();
    return Array.from(result);
  };

  /**
   * Writes a single cell in the current grid, updating both its element index and color.
   *
   * @param index - The flat index of the cell to update (row-major order).
   * @param elementIndex - The element index to assign to the cell.
   */
  const writeCell = (index: number, elementIndex: number): void => {
    if (index < 0 || index >= width * height) {
      throw new Error(
        `Cell index ${index} is out of bounds. Expected a value in range [0, ${width * height - 1}].`,
      );
    }

    if (
      !Number.isFinite(elementIndex) ||
      !Number.isInteger(elementIndex) ||
      elementIndex < 0 ||
      elementIndex >= palette.length
    ) {
      throw new Error(
        `Element index ${elementIndex} is invalid. Expected a finite integer in range [0, ${palette.length - 1}].`,
      );
    }

    const currentIds = frames % 2 === 0 ? ids0 : ids1;
    const currentColors = frames % 2 === 0 ? colors0 : colors1;

    const data = new Uint32Array([elementIndex]);
    device.queue.writeBuffer(currentIds.buffer, index * 4, data);

    const colorData = new Uint32Array([palette[elementIndex]]);
    device.queue.writeBuffer(currentColors.buffer, index * 4, colorData);
  };

  /**
   * Writes a single cell by grid coordinates, updating both its element index and color.
   *
   * @param x - The column of the cell (0-based, from left).
   * @param y - The row of the cell (0-based, from top).
   * @param elementIndex - The element index to assign to the cell.
   */
  const writeCellAt = (x: number, y: number, elementIndex: number): void => {
    if (x < 0 || x >= width) {
      throw new Error(`Column ${x} is out of bounds. Expected a value in range [0, ${width - 1}].`);
    }

    if (y < 0 || y >= height) {
      throw new Error(`Row ${y} is out of bounds. Expected a value in range [0, ${height - 1}].`);
    }

    writeCell(y * width + x, elementIndex);
  };

  /**
   * Overwrites the entire grid with the given flat array of element indices,
   * updating both the ids and colors buffers. Useful for restoring a snapshot
   * or painting the grid in bulk.
   *
   * @param grid - A flat array of element indices (row-major order) whose length must equal `width * height`.
   */
  const writeGrid = (grid: number[]): void => {
    const totalCells = width * height;

    if (grid.length !== totalCells) {
      throw new Error(
        `Grid length ${grid.length} does not match the expected length of ${totalCells}.`,
      );
    }

    const newIds = new Uint32Array(totalCells);
    const newColors = new Uint32Array(totalCells);

    for (let i = 0; i < totalCells; i++) {
      const elementIndex = grid[i];

      if (
        !Number.isFinite(elementIndex) ||
        !Number.isInteger(elementIndex) ||
        elementIndex < 0 ||
        elementIndex >= palette.length
      ) {
        throw new Error(
          `Element index ${elementIndex} at position ${i} is invalid. Expected a finite integer in range [0, ${palette.length - 1}].`,
        );
      }

      newIds[i] = elementIndex;
      newColors[i] = palette[elementIndex];
    }

    const currentIds = frames % 2 === 0 ? ids0 : ids1;
    const currentColors = frames % 2 === 0 ? colors0 : colors1;

    device.queue.writeBuffer(currentIds.buffer, 0, newIds);
    device.queue.writeBuffer(currentColors.buffer, 0, newColors);
  };

  /**
   * Advances the simulation by one step on the GPU. This function is synchronous
   * — it only queues GPU commands without waiting for them to complete.
   *
   * Call `update` as many times as needed to run multiple simulation steps,
   * then call `draw` once to render the latest state to the canvas.
   */
  const update = (): void => {
    seed.write(Math.random());

    pipeline
      .with(automatonGroup)
      .with(frames % 2 === 0 ? gridGroup0 : gridGroup1)
      .dispatchWorkgroups(WORKGROUP_COUNT_W, WORKGROUP_COUNT_H);

    frames++;
  };

  /**
   * Reads the latest simulation state from the GPU and renders it to the canvas.
   * Must be called after at least one `update` call.
   *
   * This function is asynchronous because it waits for the GPU to finish
   * processing before reading the result.
   */
  const draw = async (): Promise<void> => {
    // In <=1.3.0 there was a bug in the evolve function that would cause the drawing
    // of old data instead of new one. `frames` was updated after updating + drawing.
    // Now we determine which colors buffer to read using the *updated* frames value
    // (updated at the end of the update function). For instance, after an update
    // if frames has become odd (from even), colors1 is going to hold the new colors,
    // because gridGroup0 (even-frame update) uses colors0 as old and colors1 as new.
    colorsStagingBuffer.copyFrom(frames % 2 === 0 ? colors0 : colors1);

    // We are manually doing these steps, from map to unmap, even though
    // TgpuBuffer.read exists, because we notice heavy work happening JS-side
    // due to its readers. Since we don't need to parse data other than putting
    // it on the canvas, our approach is correct and fast. In the future, if
    // beneficial, consider drawing GPU-side to avoid reading data every frame.

    const rawBuffer = colorsStagingBuffer.buffer;
    await rawBuffer.mapAsync(GPUMapMode.READ);

    pixels.set(new Uint32Array(rawBuffer.getMappedRange()));
    g.putImageData(imageData, 0, 0);

    rawBuffer.unmap();
  };

  /**
   * Convenience function that advances the simulation by one step and draws
   * the result. Equivalent to calling `update()` followed by `await draw()`.
   */
  const evolve = async (): Promise<void> => {
    update();
    await draw();
  };

  return {
    update,
    draw,
    evolve,
    readGrid,
    writeCell,
    writeCellAt,
    writeGrid,
    setAutomaton,
    tgpuRoot: root,
  };
};
