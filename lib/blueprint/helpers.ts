import type { Neighborhood } from "@/automaton/types";

export class Helpers {
  constructor(private neighborhoodName?: Neighborhood) {}

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

  public even(): number[] {
    return this.neighborhoodName === "cross" ? [0, 2, 4] : [0, 2, 4, 6, 8];
  }

  public odd(): number[] {
    return this.neighborhoodName === "cross" ? [1, 3] : [1, 3, 5, 7];
  }
}
