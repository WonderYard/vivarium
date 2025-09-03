import type { Point } from "@/automaton/types";
import type { RefBlueprint } from "@/blueprint/ref-blueprint";
import { neighborhoodPoints } from "@/common/constants";
import type { AnyNeighbor } from "./types";

export const toRefIdOrPoint = (
  refBlueprintOrNeighbor: RefBlueprint | AnyNeighbor
): string | Point => {
  return typeof refBlueprintOrNeighbor === "string"
    ? neighborhoodPoints[refBlueprintOrNeighbor]
    : refBlueprintOrNeighbor.id;
};
