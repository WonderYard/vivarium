import type { Cross, Hexagonal, Square } from "@/common/constants";

export type AnyNeighbor = Cross | Square | Hexagonal;
export type Neighbor<N> = N extends "cross"
  ? Cross
  : N extends "hexagonal"
    ? Hexagonal
    : Square;

export type ConditionStrategy = "all" | "any" | "one" | "none";

export type ConditionExpression = (neighborhood: Uint8Array, x: number, y: number) => boolean;

export type BlueprintContext = { created: boolean };
