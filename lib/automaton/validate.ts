import type { Automaton, Point } from "@/automaton/types";
import { acceptMap } from "@/common/constants";
import { parseColor } from "@/common/utils";

type Json = Record<string, unknown>;

function check(condition: unknown, path: string, expected: string): asserts condition {
  if (!condition) {
    throw new Error(`${path} is not ${expected}.`);
  }
}

const isObject = (value: unknown): value is Json =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const isString = (value: unknown): value is string => typeof value === "string";

const isPoint = (value: unknown): boolean =>
  isObject(value) && Number.isInteger(value.x) && Number.isInteger(value.y);

const isRef = (value: unknown): boolean => Number.isInteger(value) || isPoint(value);

// Checks that value is an array of objects and returns them with their paths.
const objects = (value: unknown, path: string): [Json, string][] => {
  check(Array.isArray(value), path, "an array");

  return value.map((item: unknown, i) => {
    check(isObject(item), `${path}[${i}]`, "an object");
    return [item, `${path}[${i}]`];
  });
};

/**
 * Throws if an id is shared, or if a rule, condition or extension references an id the automaton
 * doesn't have or a point that isn't a neighbor. The builder never produces these, but automata
 * assembled outside it (or loaded with `fromPng`) can. `undefined` targets are left to the compiler.
 */
export const assertReferences = (automaton: Automaton): void => {
  const elementIds = new Set(automaton.elements.map((element) => element.id));
  const kindIds = new Set(automaton.kinds.map((kind) => kind.id));
  const refIds = new Set([...elementIds, ...kindIds]);

  if (refIds.size !== automaton.elements.length + automaton.kinds.length) {
    throw new Error("Element and kind ids must be unique.");
  }

  const assertRef = (
    ref: number | Point | undefined,
    where: string,
    ids: Set<number>,
    expected: string,
  ): void => {
    if (typeof ref === "number" && !ids.has(ref)) {
      throw new Error(`${where} references ${ref}, which is not ${expected}.`);
    }

    if (typeof ref === "object" && ![ref.x, ref.y].every((v) => v === -1 || v === 0 || v === 1)) {
      throw new Error(`${where} references (${ref.x}, ${ref.y}), which is not a neighbor.`);
    }
  };

  automaton.elements.forEach((element) => {
    element.extensions.forEach((kindId) =>
      assertRef(kindId, `Element "${element.name}"`, kindIds, "a kind"),
    );
  });

  automaton.rules.forEach((rule, i) => {
    const where = `Rule ${i}`;

    if (rule.fromType === "kind") {
      assertRef(rule.fromId, where, kindIds, "a kind");
    } else {
      assertRef(rule.fromId, where, elementIds, "an element");
    }
    assertRef(rule.to, where, elementIds, "an element");

    rule.when.forEach((condition, j) => {
      const where = `Condition ${j} of rule ${i}`;

      if (condition.type === "count") {
        assertRef(condition.check, where, refIds, "an element or a kind");
      } else if (condition.type === "is") {
        assertRef(condition.compare, where, new Set(), "a neighbor");
        assertRef(condition.with, where, refIds, "an element or a kind");
      }
    });
  });
};

/** Checks the whole shape of an untrusted automaton, then its references, before anything reads it. */
export function assertAutomaton(automaton: unknown): asserts automaton is Automaton {
  check(isObject(automaton), "the automaton", "an object");
  check(
    automaton.neighborhood === "square" || automaton.neighborhood === "cross",
    "neighborhood",
    '"square" or "cross"',
  );
  check(typeof automaton.wrapping === "boolean", "wrapping", "a boolean");

  for (const [element, path] of objects(automaton.elements, "elements")) {
    check(element.type === "element", `${path}.type`, '"element"');
    check(Number.isInteger(element.id), `${path}.id`, "an integer");
    check(isString(element.name), `${path}.name`, "a string");
    check(
      isString(element.color) && parseColor(element.color) !== undefined,
      `${path}.color`,
      "a valid color",
    );
    check(
      Array.isArray(element.extensions) && element.extensions.every(Number.isInteger),
      `${path}.extensions`,
      "an array of integers",
    );
  }

  for (const [kind, path] of objects(automaton.kinds, "kinds")) {
    check(kind.type === "kind", `${path}.type`, '"kind"');
    check(Number.isInteger(kind.id), `${path}.id`, "an integer");
    check(isString(kind.name), `${path}.name`, "a string");
  }

  for (const [rule, path] of objects(automaton.rules, "rules")) {
    check(
      rule.fromType === "element" || rule.fromType === "kind",
      `${path}.fromType`,
      '"element" or "kind"',
    );
    check(Number.isInteger(rule.fromId), `${path}.fromId`, "an integer");
    check(isRef(rule.to), `${path}.to`, "an id or a point");
    check(
      Object.values<unknown>(acceptMap).includes(rule.accept),
      `${path}.accept`,
      "a valid accept strategy",
    );

    for (const [condition, conditionPath] of objects(rule.when, `${path}.when`)) {
      check(Number.isInteger(condition.id), `${conditionPath}.id`, "an integer");

      if (condition.type === "count") {
        check(isRef(condition.check), `${conditionPath}.check`, "an id or a point");
        check(
          Array.isArray(condition.count) && condition.count.every(Number.isFinite),
          `${conditionPath}.count`,
          "an array of numbers",
        );
      } else if (condition.type === "is") {
        check(isPoint(condition.compare), `${conditionPath}.compare`, "a point");
        check(isRef(condition.with), `${conditionPath}.with`, "an id or a point");
      } else {
        check(condition.type === "chance", `${conditionPath}.type`, '"count", "is" or "chance"');

        const { ratio } = condition;
        check(
          isObject(ratio) &&
            typeof ratio.part === "number" &&
            typeof ratio.whole === "number" &&
            Number.isFinite(ratio.whole) &&
            ratio.whole > 0 &&
            ratio.part >= 0 &&
            ratio.part <= ratio.whole,
          `${conditionPath}.ratio`,
          "a ratio with 0 <= part <= whole and whole > 0",
        );
      }
    }
  }

  assertReferences(automaton as Automaton);
}
