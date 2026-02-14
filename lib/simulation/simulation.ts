import type { Automaton } from "@/automaton/types";
import { Accept, Opcode, To } from "@/common/constants";
import { type GpuAutomaton, compileGpuAutomaton } from "@/webgpu/compiler";
import {
  matchesElement,
  matchesPackedCount,
  matchesPackedIds,
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

const checkIdCount = (
  ids: number[],
  x: number,
  y: number,
  checkId: number,
  packedCount: number,
  width: number,
  height: number
): number => {
  const countMask =
    matchesElement(idAt(ids, x - 1, y - 1, width, height), checkId) +
    matchesElement(idAt(ids, x, y - 1, width, height), checkId) +
    matchesElement(idAt(ids, x + 1, y - 1, width, height), checkId) +
    matchesElement(idAt(ids, x - 1, y, width, height), checkId) +
    matchesElement(idAt(ids, x + 1, y, width, height), checkId) +
    matchesElement(idAt(ids, x - 1, y + 1, width, height), checkId) +
    matchesElement(idAt(ids, x, y + 1, width, height), checkId) +
    matchesElement(idAt(ids, x + 1, y + 1, width, height), checkId);

  return matchesPackedCount(countMask, packedCount);
};

const checkIdsCount = (
  ids: number[],
  x: number,
  y: number,
  packedIds: number,
  packedCount: number,
  width: number,
  height: number
): number => {
  const countMask =
    matchesPackedIds(idAt(ids, x - 1, y - 1, width, height), packedIds) +
    matchesPackedIds(idAt(ids, x, y - 1, width, height), packedIds) +
    matchesPackedIds(idAt(ids, x + 1, y - 1, width, height), packedIds) +
    matchesPackedIds(idAt(ids, x - 1, y, width, height), packedIds) +
    matchesPackedIds(idAt(ids, x + 1, y, width, height), packedIds) +
    matchesPackedIds(idAt(ids, x - 1, y + 1, width, height), packedIds) +
    matchesPackedIds(idAt(ids, x, y + 1, width, height), packedIds) +
    matchesPackedIds(idAt(ids, x + 1, y + 1, width, height), packedIds);

  return matchesPackedCount(countMask, packedCount);
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

      for (let i = element.ruleStart; i < element.ruleEnd; i++) {
        const rule = gpuRules[i];
        const accept = rule.accept as Accept;

        let passing = 0;

        const conditionsCount = rule.conditionsEnd - rule.conditionsStart;

        for (let j = rule.conditionsStart; j < rule.conditionsEnd; j++) {
          const condition = gpuConditions[j];
          const opcode = condition.opcode as Opcode;

          if (opcode === Opcode.COUNT_ELEMENT) {
            passing += checkIdCount(
              ids,
              x,
              y,
              condition.checkId,
              condition.countOrWithId,
              width,
              height
            );
          } else if (opcode === Opcode.COUNT_POINT) {
            const checkId = idAt(
              ids,
              x + toSigned(condition.checkPointOrComparePoint.x),
              y + toSigned(condition.checkPointOrComparePoint.y),
              width,
              height
            );
            passing += checkIdCount(
              ids,
              x,
              y,
              checkId,
              condition.countOrWithId,
              width,
              height
            );
          } else if (opcode === Opcode.COUNT_KIND) {
            passing += checkIdsCount(
              ids,
              x,
              y,
              condition.checkId,
              condition.countOrWithId,
              width,
              height
            );
          } else if (opcode === Opcode.IS_ELEMENT) {
            const compareId = idAt(
              ids,
              x + toSigned(condition.checkPointOrComparePoint.x),
              y + toSigned(condition.checkPointOrComparePoint.y),
              width,
              height
            );
            passing += matchesElement(compareId, condition.countOrWithId);
          } else if (opcode === Opcode.IS_POINT) {
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
            passing += matchesElement(compareId, withId);
          } else if (opcode === Opcode.IS_KIND) {
            const compareId = idAt(
              ids,
              x + toSigned(condition.checkPointOrComparePoint.x),
              y + toSigned(condition.checkPointOrComparePoint.y),
              width,
              height
            );
            passing += matchesPackedIds(compareId, condition.countOrWithId);
          }
          // CHANCE is intentionally skipped — tests should use deterministic rules
        }

        if (
          (accept === Accept.ALL && passing === conditionsCount) ||
          (accept === Accept.ANY && passing >= 1) ||
          (accept === Accept.ONE && passing === 1) ||
          (accept === Accept.NONE && passing === 0)
        ) {
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
