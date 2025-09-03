import { expect, test } from "vitest";
import { RefBlueprint } from "@/blueprint/ref-blueprint";
import { Accept } from "@/common/constants";
import { vivarium } from "@/vivarium/vivarium";

test("throws on an empty elements list", () => {
  const vi = vivarium();

  expect(() => vi.create()).toThrow(/^No elements found/);
});

test("throws on building after closing with create", () => {
  const vi = vivarium();
  const alive = vi.element("alive", "#ffffff");
  const rule = alive.to(alive).count(alive);

  vi.create();

  expect(() => {
    rule.accept("none");
  }).toThrow(/^Blueprint methods can no longer be called after create/);

  expect(() => rule.count(alive)).toThrow(
    /^Blueprint methods can no longer be called after create/
  );

  expect(() => alive.to(alive)).toThrow(
    /^Blueprint methods can no longer be called after create/
  );

  expect(() => vi.create()).toThrow(
    /^Blueprint methods can no longer be called after create/
  );
});

test("creates an element", () => {
  const vi = vivarium();
  const alive = vi.element("alive", "#ffffff");

  const automaton = vi.create();

  expect(automaton.elements).toHaveLength(1);
  expect(automaton.elements[0].name).toBe("alive");
  expect(automaton.elements[0].color).toBe("#ffffff");
  expect(automaton.rules).toHaveLength(0);
  expect(automaton.elements[0].extensions).toHaveLength(0);

  expect(alive.type).toBe("element");
  expect(alive).toBeInstanceOf(RefBlueprint);
  expect(alive.name).toBe("alive");
});

test("throws on duplicate element name", () => {
  const vi = vivarium();
  vi.element("alive", "#ffffff");

  expect(() => vi.element("alive", "#000000")).toThrow(
    /^Element with name "alive" already exists/
  );
});

test("creates a kind", () => {
  const vi = vivarium();
  vi.element("alive", "#ffffff");

  vi.kind("flammable");

  const automaton = vi.create();

  expect(automaton.kinds).toHaveLength(1);
  expect(automaton.kinds[0].name).toBe("flammable");
  expect(automaton.rules).toHaveLength(0);

  expect(automaton.kinds[0]).not.toHaveProperty("color");
  expect(automaton.kinds[0]).not.toHaveProperty("extensions");
});

test("throws on duplicate kind name", () => {
  const vi = vivarium();
  vi.kind("flammable");

  expect(() => vi.kind("flammable")).toThrow(
    /^Kind with name "flammable" already exists/
  );
});

test("throws on element and kind with same name", () => {
  const vi = vivarium();
  vi.element("alive", "#ffffff");

  expect(() => vi.kind("alive")).toThrow(
    /^Element with name "alive" already exists/
  );
});

test("creates a rule on an element", () => {
  const vi = vivarium();
  const alive = vi.element("alive", "#ffffff");

  alive.to(alive);

  const automaton = vi.create();

  expect(automaton.rules).toHaveLength(1);
  const rule = automaton.rules[0];

  expect(rule.to).toBe(
    automaton.elements.find((element) => element.name === "alive")?.id
  );
  expect(rule.when).toStrictEqual([]);
  expect(rule.accept).toBe(Accept.ALL);
});

test("creates a rule on a kind", () => {
  const vi = vivarium();
  const alive = vi.element("alive", "#ffffff");

  const flammable = vi.kind("flammable");

  flammable.to(alive);

  const automaton = vi.create();

  expect(automaton.rules).toHaveLength(1);
  const rule = automaton.rules[0];

  expect(rule.to).toBe(
    automaton.elements.find((element) => element.name === "alive")?.id
  );
  expect(rule.when).toStrictEqual([]);
  expect(rule.accept).toBe(Accept.ALL);
});

test("creates a count condition", () => {
  const vi = vivarium();
  const alive = vi.element("alive", "#ffffff");

  alive.to(alive).count(alive);

  const automaton = vi.create();

  const checkCondition = automaton.rules[0].when[0];

  expect(checkCondition).toHaveProperty("check");

  if (typeof checkCondition === "object" && checkCondition.type === "count") {
    expect(checkCondition.check).toBe(
      automaton.elements.find((element) => element.name === "alive")?.id
    );
    expect(checkCondition.count).toStrictEqual([1, 2, 3, 4, 5, 6, 7, 8]);
  }
});

test("creates a count condition with default cross count", () => {
  const vi = vivarium("cross");
  const alive = vi.element("alive", "#ffffff");

  alive.to(alive).count(alive);

  const automaton = vi.create();

  const checkCondition = automaton.rules[0].when[0];

  if (typeof checkCondition === "object" && checkCondition.type === "count") {
    expect(checkCondition.count).toStrictEqual([1, 2, 3, 4]);
  }
});

test("creates a count condition with set cross count", () => {
  const vi = vivarium("cross");
  const alive = vi.element("alive", "#ffffff");

  alive.to(alive).count(alive, [3]);

  const automaton = vi.create();

  const checkCondition = automaton.rules[0].when[0];

  if (typeof checkCondition === "object" && checkCondition.type === "count") {
    expect(checkCondition.count).toStrictEqual([3]);
  }
});
