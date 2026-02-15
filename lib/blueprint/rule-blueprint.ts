import { nanoid } from "nanoid";
import type { Automaton, Condition, Rule, Strategy } from "@/automaton/types";
import { BaseBlueprint } from "@/blueprint/base-blueprint";
import type { RefBlueprint } from "@/blueprint/ref-blueprint";
import type { AnyNeighbor, BlueprintContext } from "@/blueprint/types";
import { toRefIdOrPoint } from "@/blueprint/utils";
import { acceptMap, neighborhoodPoints } from "@/common/constants";

/**
 * Blueprint for a rule. Provides methods to add conditions. Returned by {@link RefBlueprint.to | element.to()} or {@link RefBlueprint.to | kind.to()}.
 */
export class RuleBlueprint extends BaseBlueprint {
  constructor(
    protected context: BlueprintContext,
    protected automaton: Automaton,
    protected rule: Rule
  ) {
    super();
  }

  private condition(when: Condition): RuleBlueprintWithAccept {
    this.assertNotCreated();

    this.rule.when = [...this.rule.when, when];

    return new RuleBlueprintWithAccept(this.context, this.automaton, this.rule);
  }

  /**
   * Adds a count condition to the rule. The rule passes only if the number of matching neighbors
   * equals one of the specified count values.
   *
   * @param refBlueprintOrNeighbor - The element, kind, or neighbor reference to count.
   * @param count - The accepted count values. Can be individual numbers or arrays. When omitted, matches any count from 1 to the neighborhood size.
   * @returns A {@link RuleBlueprintWithAccept} for chaining more conditions or setting an accept strategy.
   */
  public count(
    refBlueprintOrNeighbor: RefBlueprint | AnyNeighbor,
    ...count: (number | number[])[]
  ): RuleBlueprintWithAccept {
    let flatCount =
      this.automaton.neighborhood === "cross"
        ? [1, 2, 3, 4]
        : [1, 2, 3, 4, 5, 6, 7, 8];

    if (count.length !== 0) {
      flatCount = count.flat();
    }

    return this.condition({
      type: "count",
      id: nanoid(),
      check: toRefIdOrPoint(refBlueprintOrNeighbor),
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
    neighbor: AnyNeighbor,
    refBlueprintOrNeighbor: RefBlueprint | AnyNeighbor
  ): RuleBlueprintWithAccept {
    return this.condition({
      type: "is",
      id: nanoid(),
      compare: neighborhoodPoints[neighbor],
      with: toRefIdOrPoint(refBlueprintOrNeighbor),
    });
  }

  /**
   * Adds a chance condition to the rule. The rule passes with a probability of `part / whole`.
   *
   * @param part - The numerator of the probability fraction.
   * @param whole - The denominator of the probability fraction. Defaults to `100`.
   * @returns A {@link RuleBlueprintWithAccept} for chaining more conditions or setting an accept strategy.
   */
  public chance(part: number, whole = 100): RuleBlueprintWithAccept {
    if (part < 0) {
      part = 0;
    }
    if (whole <= 0) {
      whole = 1;
    }
    if (part > whole) {
      part = whole;
    }

    part = Math.floor(part);
    whole = Math.floor(whole);

    return this.condition({
      type: "chance",
      id: nanoid(),
      ratio: { part, whole },
    });
  }
}

/**
 * Extended rule blueprint that additionally exposes the {@link RuleBlueprintWithAccept.accept | accept} method
 * for choosing how multiple conditions are evaluated.
 */
export class RuleBlueprintWithAccept extends RuleBlueprint {
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
