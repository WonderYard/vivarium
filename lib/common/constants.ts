import * as d from "typegpu/data";
import type { Point, Strategy } from "../automaton/types";

export enum Square {
  TOP_LEFT = "TOP_LEFT",
  TOP = "TOP",
  TOP_RIGHT = "TOP_RIGHT",
  LEFT = "LEFT",
  SELF = "SELF",
  RIGHT = "RIGHT",
  BOTTOM_LEFT = "BOTTOM_LEFT",
  BOTTOM = "BOTTOM",
  BOTTOM_RIGHT = "BOTTOM_RIGHT",
}

export enum Cross {
  TOP = "TOP",
  LEFT = "LEFT",
  SELF = "SELF",
  RIGHT = "RIGHT",
  BOTTOM = "BOTTOM",
}

export const neighborhoodPoints: Record<Square, Point> = {
  [Square.TOP_LEFT]: { x: -1, y: -1 },
  [Square.TOP]: { x: 0, y: -1 },
  [Square.TOP_RIGHT]: { x: 1, y: -1 },
  [Square.LEFT]: { x: -1, y: 0 },
  [Square.SELF]: { x: 0, y: 0 },
  [Square.RIGHT]: { x: 1, y: 0 },
  [Square.BOTTOM_LEFT]: { x: -1, y: 1 },
  [Square.BOTTOM]: { x: 0, y: 1 },
  [Square.BOTTOM_RIGHT]: { x: 1, y: 1 },
};

export enum To {
  ELEMENT_ID,
  POINT,
}

export enum Opcode {
  NOOP,
  COUNT_ELEMENT,
  COUNT_POINT,
  COUNT_KIND,
  IS_ELEMENT,
  IS_POINT,
  IS_KIND,
  CHANCE,
}

export enum Accept {
  ALL,
  ANY,
  ONE,
  NONE,
}

export const acceptMap: Record<Strategy, Accept> = {
  all: Accept.ALL,
  any: Accept.ANY,
  one: Accept.ONE,
  none: Accept.NONE,
};

export const GpuElement = d.struct({
  color: d.u32,
  ruleStart: d.u32,
  ruleEnd: d.u32,
});

export const GpuRule = d.struct({
  toType: d.u32,
  toId: d.u32,
  toNeighbor: d.vec2u,
  accept: d.u32,
  conditionsStart: d.u32,
  conditionsEnd: d.u32,
});

export const GpuCondition = d.struct({
  opcode: d.u32,
  checkId: d.u32,
  countOrWithId: d.u32,
  checkPointOrComparePoint: d.vec2u,
  withPoint: d.vec2u,
  chance: d.f32,
});

export const WORKGROUP_SIZE = [8, 8] as [number, number];
export const MIN_GRID_SIZE = WORKGROUP_SIZE[0];
