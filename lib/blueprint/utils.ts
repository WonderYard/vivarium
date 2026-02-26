import type { Neighborhood, Point } from "@/automaton/types";
import type { RefBlueprint } from "@/blueprint/ref-blueprint";
import { getNeighborhoodPoints } from "@/common/constants";
import type { AnyNeighbor } from "./types";

export const toRefIdOrPoint = (
  refBlueprintOrNeighbor: RefBlueprint | AnyNeighbor,
  neighborhood: Neighborhood = "square",
): string | Point => {
  return typeof refBlueprintOrNeighbor === "string"
    ? getNeighborhoodPoints(neighborhood)[refBlueprintOrNeighbor]
    : refBlueprintOrNeighbor.id;
};
