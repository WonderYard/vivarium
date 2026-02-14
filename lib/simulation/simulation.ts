import type { Automaton } from "@/automaton/types";
import { Accept, Opcode, To } from "@/common/constants";
import { type GpuAutomaton, compileGpuAutomaton } from "@/webgpu/compiler";
import {
  evaluateCondition,
  accepted,
} from "./kernel";

export type Grid = {
  width: number;
  height: number;
  ids: number[];
};

const mod = (a: number, b: number): number => ((a % b) + b) % b;

const pointToIndex = (
  x: number,
  y: number,
  width: number,
  height: number
): number => {
  return mod(y, height) * width + mod(x, width);
};

const idAt = (
  ids: number[],
  x: number,
  y: number,
  width: number,
  height: number
): number => {
  return ids[pointToIndex(x, y, width, height)];
};

// Convert vec2u neighbor offset back to signed coordinates.
// In the GPU, -1 is stored as 0xFFFFFFFF (unsigned). We interpret values > 0x7FFFFFFF as negative.
const toSigned = (v: number): number => {
  return v > 0x7fffffff ? v - 0x100000000 : v;
};

export const evolve = (grid: Grid, gpu: GpuAutomaton): Grid => {
  const { width, height, ids } = grid;
  const { gpuElements, gpuRules, gpuConditions } = gpu;
  const newIds = new Array<number>(ids.length);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const index = y * width + x;
      const id = ids[index];
      const element = gpuElements[id];

      let transitioned = false;

      // Read the 8 neighbor IDs once, reused across conditions
      const n0 = idAt(ids, x - 1, y - 1, width, height);
      const n1 = idAt(ids, x, y - 1, width, height);
      const n2 = idAt(ids, x + 1, y - 1, width, height);
      const n3 = idAt(ids, x - 1, y, width, height);
      const n4 = idAt(ids, x + 1, y, width, height);
      const n5 = idAt(ids, x - 1, y + 1, width, height);
      const n6 = idAt(ids, x, y + 1, width, height);
      const n7 = idAt(ids, x + 1, y + 1, width, height);

      for (let i = element.ruleStart; i < element.ruleEnd; i++) {
        const rule = gpuRules[i];
        const accept = rule.accept as Accept;

        let passing = 0;

        const conditionsCount = rule.conditionsEnd - rule.conditionsStart;

        for (let j = rule.conditionsStart; j < rule.conditionsEnd; j++) {
          const condition = gpuConditions[j];
          const opcode = condition.opcode as Opcode;

          // Resolve compare/with IDs from grid (CPU array access)
          const compareId = idAt(
            ids,
            x + toSigned(condition.checkPointOrComparePoint.x),
            y + toSigned(condition.checkPointOrComparePoint.y),
            width,
            height
          );
          const withId = idAt(
            ids,
            x + toSigned(condition.withPoint.x),
            y + toSigned(condition.withPoint.y),
            width,
            height
          );

          passing += evaluateCondition(
            opcode,
            n0, n1, n2, n3, n4, n5, n6, n7,
            condition.checkId,
            condition.countOrWithId,
            compareId,
            withId
          );
          // CHANCE is handled as a no-op (returns 0) — tests should use deterministic rules
        }

        if (accepted(accept, passing, conditionsCount)) {
          let resolvedId = rule.toId;

          const toType = rule.toType as To;

          if (toType === To.POINT) {
            const pointIndex = pointToIndex(
              x + toSigned(rule.toNeighbor.x),
              y + toSigned(rule.toNeighbor.y),
              width,
              height
            );
            resolvedId = ids[pointIndex];
          }

          newIds[index] = resolvedId;
          transitioned = true;
          break;
        }
      }

      if (!transitioned) {
        newIds[index] = id;
      }
    }
  }

  return { width, height, ids: newIds };
};

export const createGrid = (
  width: number,
  height: number,
  ids: number[]
): Grid => {
  return { width, height, ids: [...ids] };
};

export const compileAndEvolve = (grid: Grid, automaton: Automaton): Grid => {
  const gpu = compileGpuAutomaton(automaton);
  return evolve(grid, gpu);
};
