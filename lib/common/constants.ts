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

/** The highest neighbor count a cell can have: 8 in a square neighborhood, 4 in a cross. */
export const maxCount = (neighborhood?: Neighborhood): number => (neighborhood === "cross" ? 4 : 8);

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

export const WORKGROUP_SIZE = [8, 8] as [number, number];
