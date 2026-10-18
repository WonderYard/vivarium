import type { Automaton, Condition, Neighborhood, Rule, Strategy } from "@/automaton/types";
import { BaseBlueprint } from "@/blueprint/base-blueprint";
import type { RefBlueprint } from "@/blueprint/ref-blueprint";
import type { BlueprintContext, Neighbor } from "@/blueprint/types";
import { assertCounts, toPoint, toRefIdOrPoint } from "@/blueprint/utils";
import { acceptMap, maxCount } from "@/common/constants";

/**
 * Blueprint for a rule. Provides methods to add conditions. Returned by {@link RefBlueprint.to | element.to()} or {@link RefBlueprint.to | kind.to()}.
 */
export class RuleBlueprint<N extends Neighborhood = Neighborhood> extends BaseBlueprint {
  constructor(
    protected context: BlueprintContext,
    protected automaton: Automaton,
    protected rule: Rule,
  ) {
    super();
  }

  private condition(when: Condition): RuleBlueprintWithAccept<N> {
    this.assertNotCreated();

    this.rule.when = [...this.rule.when, when];

    return new RuleBlueprintWithAccept<N>(this.context, this.automaton, this.rule);
  }

  /**
   * Adds a count condition to the rule. The condition passes only if the number of matching neighbors
   * equals one of the specified count values.
   *
   * @param refBlueprintOrNeighbor - The element, kind, or neighbor reference to count.
   * @param count - The accepted count values. Can be individual numbers or arrays. When omitted, matches any count from 1 to the neighborhood size.
   * @throws If a count value is not a whole number between 0 and the neighborhood size.
   * @returns A {@link RuleBlueprintWithAccept} for chaining more conditions or setting an accept strategy.
   */
  public count(
    refBlueprintOrNeighbor: RefBlueprint | Neighbor<N>,
    ...count: (number | number[])[]
  ): RuleBlueprintWithAccept<N> {
    const max = maxCount(this.automaton.neighborhood);
    const flatCount =
      count.length !== 0 ? count.flat() : Array.from({ length: max }, (_, i) => i + 1);

    assertCounts(flatCount, this.automaton.neighborhood);

    return this.condition({
      type: "count",
      id: this.context.nextId++,
      check: toRefIdOrPoint(refBlueprintOrNeighbor, this.automaton.neighborhood),
      count: flatCount,
    });
  }

  /**
   * Adds an `is` condition to the rule. Checks whether a specific neighbor position matches a given element, kind, or another neighbor position.
   *
   * @param neighbor - The neighbor position to inspect (e.g. `vi.neighbor.TOP`).
   * @param refBlueprintOrNeighbor - The element, kind, or neighbor position to compare against.
   * @returns A {@link RuleBlueprintWithAccept} for chaining more conditions or setting an accept strategy.
   */
  public is(
    neighbor: Neighbor<N>,
    refBlueprintOrNeighbor: RefBlueprint | Neighbor<N>,
  ): RuleBlueprintWithAccept<N> {
    return this.condition({
      type: "is",
      id: this.context.nextId++,
      compare: toPoint(neighbor, this.automaton.neighborhood),
      with: toRefIdOrPoint(refBlueprintOrNeighbor, this.automaton.neighborhood),
    });
  }

  /**
   * Adds a chance condition to the rule. The condition passes with a probability of `part / whole`.
   *
   * @param part - The numerator of the probability fraction, between `0` and `whole`.
   * @param whole - The denominator of the probability fraction, greater than `0`. Defaults to `100`.
   * @throws If `whole` is not greater than `0`, or `part` is not between `0` and `whole`.
   * @returns A {@link RuleBlueprintWithAccept} for chaining more conditions or setting an accept strategy.
   */
  public chance(part: number, whole = 100): RuleBlueprintWithAccept<N> {
    if (!(whole > 0) || !Number.isFinite(whole)) {
      throw new Error(`Chance whole ${whole} is invalid. Expected a finite number greater than 0.`);
    }
    if (!(part >= 0 && part <= whole)) {
      throw new Error(`Chance part ${part} is invalid. Expected a number in range [0, ${whole}].`);
    }

    return this.condition({
      type: "chance",
      id: this.context.nextId++,
      ratio: { part, whole },
    });
  }
}

/**
 * Extended rule blueprint that additionally exposes the {@link RuleBlueprintWithAccept.accept | accept} method
 * for choosing how multiple conditions are evaluated.
 */
export class RuleBlueprintWithAccept<
  N extends Neighborhood = Neighborhood,
> extends RuleBlueprint<N> {
  /**
   * Sets the acceptance strategy for multiple conditions on this rule.
   *
   * - `"all"` (default) — every condition must pass.
   * - `"any"` — at least one condition must pass.
   * - `"one"` — exactly one condition must pass.
   * - `"none"` — no condition must pass (negation).
   *
   * @param strategy - The acceptance strategy.
   */
  public accept(strategy: Strategy): void {
    this.assertNotCreated();

    this.rule.accept = acceptMap[strategy];
  }
}
