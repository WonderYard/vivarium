import type { Neighborhood } from "@/automaton/types";
import { VivariumBlueprint } from "@/blueprint/vivarium-blueprint";

/**
 * Creates a new vivarium builder used to define elements, kinds, and rules.
 *
 * @param neighborhoodName - The neighborhood type to use. Defaults to `"square"` (Moore neighborhood, 8 neighbors). Use `"cross"` for a von Neumann neighborhood (4 neighbors).
 * @returns A {@link VivariumBlueprint} instance.
 */
export function vivarium<N extends Neighborhood = "square">(
  neighborhoodName?: N
): VivariumBlueprint<N> {
  return new VivariumBlueprint(neighborhoodName);
}
