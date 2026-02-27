import { nanoid } from "nanoid";
import type { Automaton, Color, Element, Kind, Neighborhood } from "@/automaton/types";
import { BaseBlueprint } from "@/blueprint/base-blueprint";
import { Helpers } from "@/blueprint/helpers";
import { ElementBlueprint, KindBlueprint } from "@/blueprint/ref-blueprint";
import { Cross, Square } from "@/common/constants";

/**
 * The main builder for defining a vivarium. Use it to create elements, kinds, and rules,
 * then call {@link VivariumBlueprint.create | create} to produce an {@link Automaton} that can be passed to {@link setup}.
 */
export class VivariumBlueprint<N extends Neighborhood = "square"> extends BaseBlueprint {
  protected automaton: Automaton;
  protected context = { created: false };

  /**
   * Named constants for referring to specific positions in the neighborhood.
   * Available positions depend on the neighborhood type.
   *
   * For `"square"`: `TOP_LEFT`, `TOP`, `TOP_RIGHT`, `LEFT`, `SELF`, `RIGHT`, `BOTTOM_LEFT`, `BOTTOM`, `BOTTOM_RIGHT`.
   *
   * For `"cross"`: `TOP`, `LEFT`, `SELF`, `RIGHT`, `BOTTOM`.
   */
  public neighbor: N extends "cross" ? typeof Cross : typeof Square;

  /**
   * Helper functions for constructing count arrays used in conditions.
   * Includes {@link Helpers.not | not}, {@link Helpers.between | between}, {@link Helpers.even | even}, and {@link Helpers.odd | odd}.
   */
  public helpers: Helpers;

  constructor(
    private neighborhoodName?: N,
    options?: { wrapping?: boolean },
  ) {
    super();

    this.helpers = new Helpers(this.neighborhoodName);

    this.neighbor = (this.neighborhoodName === "cross" ? Cross : Square) as N extends "cross"
      ? typeof Cross
      : typeof Square;

    this.automaton = {
      neighborhood: neighborhoodName ?? "square",
      wrapping: options?.wrapping ?? false,
      elements: [],
      kinds: [],
      rules: [],
    };
  }

  /**
   * Defines a new element in the vivarium.
   *
   * @param name - A unique name identifying this element.
   * @param color - A unique color used to draw this element. Accepts CSS named colors or hex values (e.g. `"green"`, `"#FF7A45"`).
   * @param extensions - An optional array of kinds this element extends.
   * @returns An {@link ElementBlueprint} that can be used to define rules.
   */
  public element(name: string, color: Color, extensions: KindBlueprint[] = []): ElementBlueprint {
    this.assertNotCreated();
    this.assertUniqueName(name);
    this.assertUniqueColor(color);

    const element = {
      type: "element",
      id: nanoid(),
      name,
      color,
      extensions: extensions.map((element) => element.id),
    } satisfies Element;

    const index = this.automaton.elements.push(element) - 1;

    return new ElementBlueprint(this.context, this.automaton, element, index);
  }

  /**
   * Defines a new kind in the vivarium. A kind groups elements that share common behavior.
   *
   * @param name - A unique name identifying this kind.
   * @returns A {@link KindBlueprint} that can be used to define shared rules.
   */
  public kind(name: string): KindBlueprint {
    this.assertNotCreated();
    this.assertUniqueName(name);

    const kind = {
      type: "kind",
      id: nanoid(),
      name,
    } satisfies Kind;

    this.automaton.kinds.push(kind);

    return new KindBlueprint(this.context, this.automaton, kind);
  }

  /**
   * Finalizes the vivarium and produces an {@link Automaton} object. After calling this method,
   * no further elements, kinds, or rules can be added.
   *
   * @returns The compiled {@link Automaton} to pass to {@link setup}.
   */
  public create(): Automaton {
    this.assertNotCreated();
    this.assertNotEmpty();

    const automaton = this.automaton;

    this.context.created = true;

    return automaton;
  }
}
