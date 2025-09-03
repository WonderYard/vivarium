import type {
  Automaton,
  Element,
  Kind,
  RefType,
  Rule,
} from "@/automaton/types";
import { BaseBlueprint } from "@/blueprint/base-blueprint";
import { RuleBlueprint } from "@/blueprint/rule-blueprint";
import type { AnyNeighbor, BlueprintContext } from "@/blueprint/types";
import { toRefIdOrPoint } from "@/blueprint/utils";
import { Accept } from "@/common/constants";

export abstract class RefBlueprint extends BaseBlueprint {
  public id: string;
  public name: string;
  public type: RefType;

  constructor(
    protected automaton: Automaton,
    ref: Element | Kind
  ) {
    super();

    this.id = ref.id;
    this.name = ref.name;
    this.type = ref.type;
  }

  public to(
    elementBlueprintOrNeighbor: ElementBlueprint | AnyNeighbor
  ): RuleBlueprint {
    this.assertNotCreated();

    const rule = {
      fromId: this.id,
      fromType: this.type,
      to: toRefIdOrPoint(elementBlueprintOrNeighbor),
      when: [],
      accept: Accept.ALL,
    } satisfies Rule;

    this.automaton.rules.push(rule);

    return new RuleBlueprint(this.context, this.automaton, rule);
  }
}

export class ElementBlueprint extends RefBlueprint {
  type = "element" as const;
  id: string;

  constructor(
    protected context: BlueprintContext,
    automaton: Automaton,
    element: Element
  ) {
    super(automaton, element);

    this.id = element.id;
  }
}

export class KindBlueprint extends RefBlueprint {
  type = "kind" as const;
  id: string;

  constructor(
    protected context: BlueprintContext,
    automaton: Automaton,
    kind: Kind
  ) {
    super(automaton, kind);

    this.id = kind.id;
  }
}
