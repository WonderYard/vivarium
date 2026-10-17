import type { Neighborhood, Point } from "@/automaton/types";
import type { RefBlueprint } from "@/blueprint/ref-blueprint";
import { Cross, maxCount, neighborhoodPoints, Square } from "@/common/constants";
import type { AnyNeighbor } from "./types";

// Types only stop TS callers, so plain JS can still pass a diagonal in a cross vivarium, or any string.
export const toPoint = (neighbor: AnyNeighbor, neighborhood: Neighborhood): Point => {
  const neighbors: string[] = Object.values(neighborhood === "cross" ? Cross : Square);

  if (!neighbors.includes(neighbor)) {
    throw new Error(
      `Neighbor "${neighbor}" is invalid. Expected one of ${neighbors.join(", ")} for a ${neighborhood} neighborhood.`,
    );
  }

  return neighborhoodPoints[neighbor];
};

export const toRefIdOrPoint = (
  refBlueprintOrNeighbor: RefBlueprint | AnyNeighbor,
  neighborhood: Neighborhood,
): string | Point => {
  return typeof refBlueprintOrNeighbor === "string"
    ? toPoint(refBlueprintOrNeighbor, neighborhood)
    : refBlueprintOrNeighbor.id;
};

export const assertCounts = (counts: number[], neighborhood?: Neighborhood): void => {
  const max = maxCount(neighborhood);
  const invalid = counts.find((count) => !Number.isInteger(count) || count < 0 || count > max);

  if (invalid !== undefined) {
    throw new Error(
      `Count ${invalid} is invalid. Expected a whole number in range [0, ${max}] for a ${neighborhood ?? "square"} neighborhood.`,
    );
  }
};
