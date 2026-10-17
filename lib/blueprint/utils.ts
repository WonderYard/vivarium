import type { Neighborhood, Point } from "@/automaton/types";
import type { RefBlueprint } from "@/blueprint/ref-blueprint";
import { maxCount, neighborhoodPoints } from "@/common/constants";
import type { AnyNeighbor } from "./types";

export const toRefIdOrPoint = (
  refBlueprintOrNeighbor: RefBlueprint | AnyNeighbor,
): string | Point => {
  return typeof refBlueprintOrNeighbor === "string"
    ? neighborhoodPoints[refBlueprintOrNeighbor]
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
