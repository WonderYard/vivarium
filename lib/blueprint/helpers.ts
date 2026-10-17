import type { Neighborhood } from "@/automaton/types";
import { assertCounts } from "@/blueprint/utils";
import { maxCount } from "@/common/constants";

/**
 * Helper functions for constructing count arrays used in conditions.
 * Accessed via `vi.helpers`.
 */
export class Helpers {
  constructor(private neighborhoodName?: Neighborhood) {}

  /**
   * Returns every valid count value except the ones specified.
   *
   * @param value - A number or array of numbers to exclude.
   * @returns An array of all valid count values minus the excluded ones.
   * @throws If a value is not a whole number between 0 and the neighborhood size.
   */
  public not = (value: number | number[]): number[] => {
    const excluded = [value].flat();
    assertCounts(excluded, this.neighborhoodName);

    return Array.from({ length: maxCount(this.neighborhoodName) + 1 }, (_, i) => i).filter(
      (count) => !excluded.includes(count),
    );
  };

  /**
   * Generates a range of whole numbers from `min` to `max` (inclusive).
   *
   * @param min - The start of the range (must be ≥ 0).
   * @param max - The end of the range (must be ≥ `min` and ≤ the neighborhood size: 8 for square, 4 for cross).
   * @returns An array of numbers from `min` to `max`.
   * @throws If `min` or `max` is not a whole number between 0 and the neighborhood size, or `max` is less than `min`.
   */
  public between = (min: number, max: number): number[] => {
    assertCounts([min, max], this.neighborhoodName);

    if (max < min) {
      throw new Error(`max ${max} cannot be less than min ${min}`);
    }

    return Array.from({ length: max - min + 1 }, (_, i) => min + i);
  };

  /**
   * Returns all even count values (zero included) for the current neighborhood type.
   *
   * @returns `[0, 2, 4, 6, 8]` for square, `[0, 2, 4]` for cross.
   */
  public even = (): number[] => {
    return this.neighborhoodName === "cross" ? [0, 2, 4] : [0, 2, 4, 6, 8];
  };

  /**
   * Returns all odd count values for the current neighborhood type.
   *
   * @returns `[1, 3, 5, 7]` for square, `[1, 3]` for cross.
   */
  public odd = (): number[] => {
    return this.neighborhoodName === "cross" ? [1, 3] : [1, 3, 5, 7];
  };
}
