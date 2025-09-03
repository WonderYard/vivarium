import type { Neighborhood } from "@/automaton/types";
import { VivariumBlueprint } from "@/blueprint/vivarium-blueprint";

export function vivarium<N extends Neighborhood = "square">(
  neighborhoodName?: N
): VivariumBlueprint<N> {
  return new VivariumBlueprint(neighborhoodName);
}
