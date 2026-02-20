import type { Automaton, Color } from "@/automaton/types";
import type { BlueprintContext } from "@/blueprint/types";

export abstract class BaseBlueprint {
  protected abstract automaton: Automaton;
  protected abstract context: BlueprintContext;

  protected assertNotCreated(): void {
    if (this.context.created) {
      throw new Error("Blueprint methods can no longer be called after create");
    }
  }

  protected assertUniqueName(name: string): void {
    if (name.trim().length === 0) {
      throw new Error(`Name cannot be empty`);
    }

    if (this.automaton.elements.some((element) => element.name === name)) {
      throw new Error(`Element with name "${name}" already exists`);
    }
    if (this.automaton.kinds.some((kind) => kind.name === name)) {
      throw new Error(`Kind with name "${name}" already exists`);
    }
  }

  protected assertUniqueColor(color: Color): void {
    if (color.trim().length === 0) {
      throw new Error(`Color cannot be empty`);
    }

    if (this.automaton.elements.some((element) => element.color === color)) {
      throw new Error(`Element with color "${color}" already exists`);
    }
  }

  protected assertNotEmpty(): void {
    if (!this.automaton.elements.length) {
      throw new Error("No elements found in automaton, at least one element is required");
    }
  }
}
