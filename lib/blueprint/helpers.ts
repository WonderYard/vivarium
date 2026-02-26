import type { Neighborhood } from "@/automaton/types";

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
   */
  public not(value: number | number[]): number[] {
    const not =
      this.neighborhoodName === "cross"
        ? new Set([0, 1, 2, 3, 4])
        : new Set([0, 1, 2, 3, 4, 5, 6, 7, 8]);

    if (typeof value === "number") {
      not.delete(value);
    } else if (Array.isArray(value)) {
      value.forEach((value) => {
        not.delete(value);
      });
    }

    return Array.from(not);
  }

  /**
   * Generates a range of whole numbers from `min` to `max` (inclusive).
   *
   * @param min - The start of the range (must be ≥ 0).
   * @param max - The end of the range (must be ≤ 8 and greater than `min`). For cross neighborhoods, should not exceed 4.
   * @returns An array of numbers from `min` to `max`.
   */
  public between(min: number, max: number): number[] {
    if (min < 0) {
      throw new Error("min cannot be less than 0");
    }
    if (max > 8) {
      throw new Error("max cannot be more than 8");
    }
    if (max <= min) {
      throw new Error("max cannot be less or equal than min");
    }

    const range = [];
    for (let i = min; i <= max; i++) {
      range.push(i);
    }

    return range;
  }

  /**
   * Returns all even count values (zero included) for the current neighborhood type.
   *
   * @returns `[0, 2, 4, 6, 8]` for square, `[0, 2, 4]` for cross.
   */
  public even(): number[] {
    return this.neighborhoodName === "cross" ? [0, 2, 4] : [0, 2, 4, 6, 8];
  }

  /**
   * Returns all odd count values for the current neighborhood type.
   *
   * @returns `[1, 3, 5, 7]` for square, `[1, 3]` for cross.
   */
  public odd(): number[] {
    return this.neighborhoodName === "cross" ? [1, 3] : [1, 3, 5, 7];
  }
}
