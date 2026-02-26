import * as d from "typegpu/data";
import type { Neighborhood, Point, Strategy } from "../automaton/types";

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

/**
 * Hexagonal neighborhood positions. Uses offset coordinates (even-row convention).
 * The GPU shader adjusts offsets for odd rows at runtime.
 */
export enum Hexagonal {
  TOP_LEFT = "HEX_TOP_LEFT",
  TOP_RIGHT = "HEX_TOP_RIGHT",
  LEFT = "HEX_LEFT",
  SELF = "HEX_SELF",
  RIGHT = "HEX_RIGHT",
  BOTTOM_LEFT = "HEX_BOTTOM_LEFT",
  BOTTOM_RIGHT = "HEX_BOTTOM_RIGHT",
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

/**
 * Hexagonal neighborhood offsets using even-row convention.
 * For even rows (y % 2 == 0) these offsets are used as-is.
 * For odd rows, the GPU shader adds 1 to the x offset when dy != 0.
 */
export const hexagonalNeighborhoodPoints: Record<Hexagonal, Point> = {
  [Hexagonal.TOP_LEFT]: { x: -1, y: -1 },
  [Hexagonal.TOP_RIGHT]: { x: 0, y: -1 },
  [Hexagonal.LEFT]: { x: -1, y: 0 },
  [Hexagonal.SELF]: { x: 0, y: 0 },
  [Hexagonal.RIGHT]: { x: 1, y: 0 },
  [Hexagonal.BOTTOM_LEFT]: { x: -1, y: 1 },
  [Hexagonal.BOTTOM_RIGHT]: { x: 0, y: 1 },
};

/**
 * Returns the correct neighborhood points record for the given neighborhood type.
 */
export const getNeighborhoodPoints = (neighborhood: Neighborhood): Record<string, Point> => {
  return neighborhood === "hexagonal"
    ? (hexagonalNeighborhoodPoints as Record<string, Point>)
    : (neighborhoodPoints as Record<string, Point>);
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
