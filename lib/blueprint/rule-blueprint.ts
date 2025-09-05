import { nanoid } from "nanoid";
import type { Automaton, Condition, Rule, Strategy } from "@/automaton/types";
import { BaseBlueprint } from "@/blueprint/base-blueprint";
import type { RefBlueprint } from "@/blueprint/ref-blueprint";
import type { AnyNeighbor, BlueprintContext } from "@/blueprint/types";
import { toRefIdOrPoint } from "@/blueprint/utils";
import { acceptMap, neighborhoodPoints } from "@/common/constants";

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

class RuleBlueprintWithAccept extends RuleBlueprint {
  public accept(strategy: Strategy): void {
    this.assertNotCreated();

    this.rule.accept = acceptMap[strategy];
  }
}
