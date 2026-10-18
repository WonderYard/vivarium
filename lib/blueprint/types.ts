import type { Cross, Square } from "@/common/constants";

export type AnyNeighbor = Cross | Square;
export type Neighbor<N> = N extends "cross" ? Cross : Square;

export type BlueprintContext = { created: boolean; nextId: number };
