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

/**
 * Base class for element and kind blueprints. Provides the {@link RefBlueprint.to | to} method for defining rules.
 */
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

  /**
   * Creates a new rule that transitions cells of this element (or kind) to another element or neighbor reference.
   *
   * @param elementBlueprintOrNeighbor - The target element, or a neighbor position reference (e.g. `vi.neighbor.TOP`).
   * @returns A {@link RuleBlueprint} for chaining conditions.
   */
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

/**
 * Blueprint for an element. Returned by {@link VivariumBlueprint.element | vivarium().element()}.
 * Use {@link RefBlueprint.to | to()} to define rules on this element.
 */
export class ElementBlueprint extends RefBlueprint {
  type = "element" as const;
  id: string;
  index: number;

  constructor(
    protected context: BlueprintContext,
    automaton: Automaton,
    element: Element,
    index: number
  ) {
    super(automaton, element);

    this.id = element.id;
    this.index = index;
  }
}

/**
 * Blueprint for a kind. Returned by {@link VivariumBlueprint.kind | vivarium().kind()}.
 * Use {@link RefBlueprint.to | to()} to define shared rules for all elements extending this kind.
 */
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
