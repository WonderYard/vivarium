import type { Automaton, Neighborhood, Point } from "@/automaton/types";
import { acceptMap, maxCount } from "@/common/constants";
import { parseColor } from "@/common/utils";

type Json = Record<string, unknown>;

const isObject = (value: unknown): value is Json =>
  typeof value === "object" && value !== null && !Array.isArray(value);

// The checks the builder makes on its arguments and assertAutomaton makes on a whole automaton,
// so both agree on what is valid.

/** A non-blank string. */
export const isName = (value: unknown): value is string =>
  typeof value === "string" && value.trim().length > 0;

/** A whole number of neighbors a cell can have: 0 to 8 for square (the default), 0 to 4 for cross. */
export const isCount = (value: unknown, neighborhood?: Neighborhood): boolean =>
  Number.isInteger(value) && (value as number) >= 0 && (value as number) <= maxCount(neighborhood);

/** A point of the neighborhood, the cell itself included: no diagonals in a cross. */
export const isNeighbor = (value: unknown, neighborhood: Neighborhood): value is Point =>
  isObject(value) &&
  [value.x, value.y].every((v) => v === -1 || v === 0 || v === 1) &&
  (neighborhood === "square" || value.x === 0 || value.y === 0);

/** The whole of a chance ratio: a finite number greater than 0. */
export const isWhole = (value: unknown): boolean =>
  typeof value === "number" && Number.isFinite(value) && value > 0;

/** The part of a chance ratio: a number in [0, whole]. */
export const isPart = (value: unknown, whole: number): boolean =>
  typeof value === "number" && value >= 0 && value <= whole;

// Short enough for an error message: numbers as they are (NaN too), short JSON, or else the type.
const show = (value: unknown): string => {
  if (typeof value === "number" || value === undefined) return String(value);

  const json = JSON.stringify(value);
  if (json !== undefined && json.length <= 40) return json;

  return Array.isArray(value)
    ? "an array"
    : typeof value === "object"
      ? "an object"
      : `a ${typeof value}`;
};

function check(
  condition: unknown,
  path: string,
  value: unknown,
  expected: string,
): asserts condition {
  if (!condition) {
    throw new Error(`${path} is ${show(value)}, expected ${expected}.`);
  }
}

// Checks that value is an array of objects and returns them with their paths.
const objects = (value: unknown, path: string): [Json, string][] => {
  check(Array.isArray(value), path, value, "an array");

  return value.map((item: unknown, i) => {
    check(isObject(item), `${path}[${i}]`, item, "an object");
    return [item, `${path}[${i}]`];
  });
};

/**
 * Throws unless the automaton is one the builder could have made, apart from its ids and colors:
 * ids only need to be unique among elements and kinds, and elements may share a color.
 * The error names the first bad field by its path, e.g. `rules[0].when[1].count[2]`.
 *
 * Automata reach `setup` and `setAutomaton` from outside the builder too (an editor, a file),
 * so these check them before anything reads them.
 */
export function assertAutomaton(automaton: unknown): asserts automaton is Automaton {
  check(isObject(automaton), "the automaton", automaton, "an object");

  const { neighborhood } = automaton;
  check(
    neighborhood === "square" || neighborhood === "cross",
    "neighborhood",
    neighborhood,
    '"square" or "cross"',
  );
  check(typeof automaton.wrapping === "boolean", "wrapping", automaton.wrapping, "a boolean");

  const elements = objects(automaton.elements, "elements");
  check(elements.length > 0, "elements", automaton.elements, "at least one element");
  const kinds = objects(automaton.kinds, "kinds");

  const elementIds = new Set<unknown>();
  const kindIds = new Set<unknown>();
  const names = new Set<unknown>();

  // Like the builder: ids and names are unique across elements and kinds.
  const identify = (ref: Json, path: string, type: "element" | "kind", ids: Set<unknown>) => {
    check(ref.type === type, `${path}.type`, ref.type, `"${type}"`);
    check(
      Number.isInteger(ref.id) && !elementIds.has(ref.id) && !kindIds.has(ref.id),
      `${path}.id`,
      ref.id,
      "an integer id no other element or kind has",
    );
    check(
      isName(ref.name) && !names.has(ref.name),
      `${path}.name`,
      ref.name,
      "a non-blank name no other element or kind has",
    );
    ids.add(ref.id);
    names.add(ref.name);
  };

  elements.forEach(([element, path]) => identify(element, path, "element", elementIds));
  kinds.forEach(([kind, path]) => identify(kind, path, "kind", kindIds));

  for (const [element, path] of elements) {
    check(
      typeof element.color === "string" && parseColor(element.color) !== undefined,
      `${path}.color`,
      element.color,
      "a CSS named color or a hex value",
    );
    check(Array.isArray(element.extensions), `${path}.extensions`, element.extensions, "an array");
    element.extensions.forEach((id: unknown, i) =>
      check(kindIds.has(id), `${path}.extensions[${i}]`, id, "a kind id"),
    );
  }

  const neighbor = `a ${neighborhood} neighbor`;
  const isRef = (value: unknown) => elementIds.has(value) || kindIds.has(value);

  for (const [rule, path] of objects(automaton.rules, "rules")) {
    check(
      rule.fromType === "element" || rule.fromType === "kind",
      `${path}.fromType`,
      rule.fromType,
      '"element" or "kind"',
    );
    check(
      (rule.fromType === "element" ? elementIds : kindIds).has(rule.fromId),
      `${path}.fromId`,
      rule.fromId,
      rule.fromType === "element" ? "an element id" : "a kind id",
    );
    check(
      elementIds.has(rule.to) || isNeighbor(rule.to, neighborhood),
      `${path}.to`,
      rule.to,
      `an element id or ${neighbor}`,
    );
    check(
      Object.values<unknown>(acceptMap).includes(rule.accept),
      `${path}.accept`,
      rule.accept,
      "0, 1, 2 or 3 (all, any, one or none)",
    );

    for (const [condition, conditionPath] of objects(rule.when, `${path}.when`)) {
      check(Number.isInteger(condition.id), `${conditionPath}.id`, condition.id, "an integer");

      if (condition.type === "count") {
        check(
          isRef(condition.check) || isNeighbor(condition.check, neighborhood),
          `${conditionPath}.check`,
          condition.check,
          `an element id, a kind id or ${neighbor}`,
        );
        check(
          Array.isArray(condition.count),
          `${conditionPath}.count`,
          condition.count,
          "an array",
        );
        condition.count.forEach((count: unknown, i) =>
          check(
            isCount(count, neighborhood),
            `${conditionPath}.count[${i}]`,
            count,
            `a whole number in [0, ${maxCount(neighborhood)}] for a ${neighborhood} neighborhood`,
          ),
        );
      } else if (condition.type === "is") {
        check(
          isNeighbor(condition.compare, neighborhood),
          `${conditionPath}.compare`,
          condition.compare,
          neighbor,
        );
        check(
          isRef(condition.with) || isNeighbor(condition.with, neighborhood),
          `${conditionPath}.with`,
          condition.with,
          `an element id, a kind id or ${neighbor}`,
        );
      } else if (condition.type === "chance") {
        const { ratio } = condition;
        check(isObject(ratio), `${conditionPath}.ratio`, ratio, "an object");
        check(
          isWhole(ratio.whole),
          `${conditionPath}.ratio.whole`,
          ratio.whole,
          "a finite number greater than 0",
        );
        const whole = ratio.whole as number;
        check(
          isPart(ratio.part, whole),
          `${conditionPath}.ratio.part`,
          ratio.part,
          `a number in [0, ${whole}]`,
        );
      } else {
        check(false, `${conditionPath}.type`, condition.type, '"count", "is" or "chance"');
      }
    }
  }
}
