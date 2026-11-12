import type { Neighborhood, Point } from "@/automaton/types";
import type { RefBlueprint } from "@/blueprint/ref-blueprint";
import { isCount, isNeighbor } from "@/automaton/validate";
import { Cross, maxCount, neighborhoodPoints, Square } from "@/common/constants";
import type { AnyNeighbor } from "./types";

// Types only stop TS callers, so plain JS can still pass a diagonal in a cross vivarium, or any string.
export const toPoint = (neighbor: AnyNeighbor, neighborhood: Neighborhood): Point => {
  const point: unknown = neighborhoodPoints[neighbor];

  if (!isNeighbor(point, neighborhood)) {
    const neighbors = Object.values(neighborhood === "cross" ? Cross : Square);
    throw new Error(
      `Neighbor "${neighbor}" is invalid. Expected one of ${neighbors.join(", ")} for a ${neighborhood} neighborhood.`,
    );
  }

  return point;
};

export const toRefIdOrPoint = (
  refBlueprintOrNeighbor: RefBlueprint | AnyNeighbor,
  neighborhood: Neighborhood,
): number | Point => {
  return typeof refBlueprintOrNeighbor === "string"
    ? toPoint(refBlueprintOrNeighbor, neighborhood)
    : refBlueprintOrNeighbor.id;
};

// The counts passed to count() or pointed(). None means any count from 1 to the neighborhood size.
export const toCounts = (count: (number | number[])[], neighborhood?: Neighborhood): number[] => {
  const counts =
    count.length !== 0
      ? count.flat()
      : Array.from({ length: maxCount(neighborhood) }, (_, i) => i + 1);
  assertCounts(counts, neighborhood);
  return counts;
};

export const assertCounts = (counts: number[], neighborhood?: Neighborhood): void => {
  const max = maxCount(neighborhood);
  const invalid = counts.find((count) => !isCount(count, neighborhood));

  if (invalid !== undefined) {
    throw new Error(
      `Count ${invalid} is invalid. Expected a whole number in range [0, ${max}] for a ${neighborhood ?? "square"} neighborhood.`,
    );
  }
};
