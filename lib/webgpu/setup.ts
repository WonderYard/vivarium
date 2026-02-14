import { randf } from "@typegpu/noise";
import tgpu, { type TgpuBindGroup, type TgpuUniform } from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
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
import {
  evaluateCondition,
  accepted,
} from "@/simulation/kernel";
import { compileGpuAutomaton } from "./compiler";

const adapter = await navigator.gpu.requestAdapter();
if (!adapter) {
  throw new Error("No adapter");
}
const device = await adapter.requestDevice();

void device.lost.then(() => {
  throw Error("Device lost");
});

let frames = 0;
let seed: TgpuUniform<d.F32>;

// layouts are predefined
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

// Because functions reference layouts and not groups,
// they can be defined once

const pointToIndex = (x: number, y: number) => {
  "use gpu";

  // Note: keep in mind that we are performing subtraction in unsigned space.
  // The modulo here is the only thing that allows us to use unsigned ints everywhere.
  // Example 0 - 1 = 4294967295 in unsigned space, and (0 - 1) % 1024 = 1023 as expected.
  return (
    (y % gridLayout.bound.dimensions.$.y) * gridLayout.bound.dimensions.$.x +
    (x % gridLayout.bound.dimensions.$.x)
  );
};

const idAt = (x: number, y: number) => {
  "use gpu";
  return gridLayout.bound.ids.$[pointToIndex(x, y)];
};

// also the main compute function has no variable dependencies
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

  // Read the 8 neighbor IDs once, reused across conditions
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

      // Resolve compare/with IDs from grid (binding-dependent)
      const comparePoint = condition.checkPointOrComparePoint;
      const compareId = idAt(x + comparePoint.x, y + comparePoint.y);
      const withPoint = condition.withPoint;
      const withId = idAt(x + withPoint.x, y + withPoint.y);

      if (opcode === Opcode.CHANCE) {
        const chance = condition.chance;
        randf.seed3(
          d.vec3f(
            std.div(d.vec2f(pos.xy), d.vec2f(gridLayout.bound.dimensions.$.xy)),
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

export const setup = ({
  canvas,
  automaton,
}: {
  canvas: HTMLCanvasElement;
  automaton: Automaton;
}) => {
  const root = tgpu.initFromDevice({ device });
  const pipeline = root["~unstable"].withCompute(mainCompute).createPipeline();

  frames = 0;
  seed = root.createUniform(d.f32);

  const { width, height } = canvas;
  const g = canvas.getContext("2d") as CanvasRenderingContext2D;

  const WORKGROUP_COUNT_W = Math.ceil(width / WORKGROUP_SIZE[0]);
  const WORKGROUP_COUNT_H = Math.ceil(height / WORKGROUP_SIZE[1]);

  // Define all the buffers. Creating them depends on width and height only.
  // When automaton changes we don't need to recreate them.

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

  const colorsStagingBuffer = root
    .createBuffer(d.arrayOf(d.u32, width * height))
    .$addFlags(GPUBufferUsage.MAP_READ);

  // no staging buffer needed for ids, since we only need to draw the result

  const gridGroup0 = root.createBindGroup(gridLayout, {
    dimensions,
    colors: colors0,
    newColors: colors1,
    ids: ids0,
    newIds: ids1,
  });

  const gridGroup1 = root.createBindGroup(gridLayout, {
    dimensions,
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

    automatonGroup = root.createBindGroup(automatonLayout, {
      neighborhood: root.createBuffer(d.u32, gpuNeighborhood).$usage("uniform"),
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
  };

  setAutomaton({ automaton });

  const colors = new Uint32Array(width * height);
  const ids = new Uint32Array(width * height);

  // TODO: for now we initialize like this
  for (let i = 0; i < colors.length; i++) {
    const randomIndex = Math.floor(Math.random() * automaton.elements.length);
    colors[i] = palette[randomIndex];
    ids[i] = randomIndex;
  }

  // and we write the inizialization to the buffers
  colors0.write(Array.from(colors));
  ids0.write(Array.from(ids));

  const imageData = g.createImageData(width, height);
  const pixels = new Uint32Array(imageData.data.buffer);

  const evolve = async (): Promise<void> => {
    seed.write(Math.random());

    pipeline
      .with(automatonGroup)
      .with(frames % 2 === 0 ? gridGroup0 : gridGroup1)
      .dispatchWorkgroups(WORKGROUP_COUNT_W, WORKGROUP_COUNT_H);

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

    frames++;
  };

  return { evolve, setAutomaton, tgpuRoot: root };
};
