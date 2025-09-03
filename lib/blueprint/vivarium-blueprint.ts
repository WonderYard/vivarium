import { nanoid } from "nanoid";
import type {
  Automaton,
  Color,
  Element,
  Kind,
  Neighborhood,
} from "@/automaton/types";
import { BaseBlueprint } from "@/blueprint/base-blueprint";
import { Helpers } from "@/blueprint/helpers";
import { ElementBlueprint, KindBlueprint } from "@/blueprint/ref-blueprint";
import { Cross, Square } from "@/common/constants";

export class VivariumBlueprint<
  N extends Neighborhood = "square",
> extends BaseBlueprint {
  protected automaton: Automaton;
  protected context = { created: false };

  public neighbor: N extends "cross" ? typeof Cross : typeof Square;
  public helpers: Helpers;

  constructor(private neighborhoodName?: N) {
    super();

    this.helpers = new Helpers(this.neighborhoodName);

    this.neighbor = (
      this.neighborhoodName === "cross" ? Cross : Square
    ) as N extends "cross" ? typeof Cross : typeof Square;

    this.automaton = {
      neighborhood: neighborhoodName ?? "square",
      elements: [],
      kinds: [],
      rules: [],
    };
  }

  public element(
    name: string,
    color: Color,
    extensions: KindBlueprint[] = []
  ): ElementBlueprint {
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

    this.automaton.elements.push(element);

    return new ElementBlueprint(this.context, this.automaton, element);
  }

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

  public create(): Automaton {
    this.assertNotCreated();
    this.assertNotEmpty();

    const automaton = this.automaton;

    this.context.created = true;

    return automaton;
  }
}
