import { expect, test } from "vitest";
import { RefBlueprint } from "@/blueprint/ref-blueprint";
import { Accept, Cross, Square } from "@/common/constants";
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
    /^Blueprint methods can no longer be called after create/,
  );

  expect(() => alive.to(alive)).toThrow(/^Blueprint methods can no longer be called after create/);

  expect(() => vi.create()).toThrow(/^Blueprint methods can no longer be called after create/);
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

  expect(() => vi.element("alive", "#000000")).toThrow(/^Element with name "alive" already exists/);
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

  expect(() => vi.kind("flammable")).toThrow(/^Kind with name "flammable" already exists/);
});

test("throws on element and kind with same name", () => {
  const vi = vivarium();
  vi.element("alive", "#ffffff");

  expect(() => vi.kind("alive")).toThrow(/^Element with name "alive" already exists/);
});

test("creates a rule on an element", () => {
  const vi = vivarium();
  const alive = vi.element("alive", "#ffffff");

  alive.to(alive);

  const automaton = vi.create();

  expect(automaton.rules).toHaveLength(1);
  const rule = automaton.rules[0];

  expect(rule.to).toBe(automaton.elements.find((element) => element.name === "alive")?.id);
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

  expect(rule.to).toBe(automaton.elements.find((element) => element.name === "alive")?.id);
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
      automaton.elements.find((element) => element.name === "alive")?.id,
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

// ── Vivarium neighborhood defaults ──────────────────────────────────

test("defaults to square neighborhood", () => {
  const vi = vivarium();
  vi.element("a", "#111111");

  const automaton = vi.create();

  expect(automaton.neighborhood).toBe("square");
});

test("creates cross neighborhood when specified", () => {
  const vi = vivarium("cross");
  vi.element("a", "#111111");

  const automaton = vi.create();

  expect(automaton.neighborhood).toBe("cross");
});

test("exposes Square neighbor constants for square neighborhood", () => {
  const vi = vivarium();

  expect(vi.neighbor).toBe(Square);
});

test("exposes Cross neighbor constants for cross neighborhood", () => {
  const vi = vivarium("cross");

  expect(vi.neighbor).toBe(Cross);
});

// ── BaseBlueprint validation ────────────────────────────────────────

test("throws on empty element name", () => {
  const vi = vivarium();

  expect(() => vi.element("", "#ffffff")).toThrow(/^Name cannot be empty/);
});

test("throws on whitespace-only element name", () => {
  const vi = vivarium();

  expect(() => vi.element("   ", "#ffffff")).toThrow(/^Name cannot be empty/);
});

test("throws on empty kind name", () => {
  const vi = vivarium();

  expect(() => vi.kind("")).toThrow(/^Name cannot be empty/);
});

test("throws on empty element color", () => {
  const vi = vivarium();

  expect(() => vi.element("a", "")).toThrow(/^Color cannot be empty/);
});

test("throws on whitespace-only element color", () => {
  const vi = vivarium();

  expect(() => vi.element("a", "   ")).toThrow(/^Color cannot be empty/);
});

test("throws on duplicate element color", () => {
  const vi = vivarium();
  vi.element("a", "#ffffff");

  expect(() => vi.element("b", "#ffffff")).toThrow(/^Element with color "#ffffff" already exists/);
});

test("throws on kind and element with same name (kind first)", () => {
  const vi = vivarium();
  vi.kind("shared");

  expect(() => vi.element("shared", "#ffffff")).toThrow(/^Kind with name "shared" already exists/);
});

// ── Element creation with extensions ────────────────────────────────

test("creates an element with kind extensions", () => {
  const vi = vivarium();
  const flammable = vi.kind("flammable");
  const element = vi.element("wood", "#8b4513", [flammable]);

  const automaton = vi.create();

  expect(automaton.elements[0].extensions).toHaveLength(1);
  expect(automaton.elements[0].extensions[0]).toBe(flammable.id);
  expect(element.id).toBe(automaton.elements[0].id);
});

test("creates an element with multiple kind extensions", () => {
  const vi = vivarium();
  const flammable = vi.kind("flammable");
  const organic = vi.kind("organic");
  vi.element("wood", "#8b4513", [flammable, organic]);

  const automaton = vi.create();

  expect(automaton.elements[0].extensions).toHaveLength(2);
  expect(automaton.elements[0].extensions).toContain(flammable.id);
  expect(automaton.elements[0].extensions).toContain(organic.id);
});

test("creates multiple elements", () => {
  const vi = vivarium();
  vi.element("alive", "#ffffff");
  vi.element("dead", "#000000");
  vi.element("zombie", "#00ff00");

  const automaton = vi.create();

  expect(automaton.elements).toHaveLength(3);
  expect(automaton.elements.map((e) => e.name)).toStrictEqual(["alive", "dead", "zombie"]);
});

// ── Kind properties ─────────────────────────────────────────────────

test("kind has correct type and id", () => {
  const vi = vivarium();
  vi.element("a", "#111111");
  const k = vi.kind("flammable");

  expect(k.type).toBe("kind");
  expect(k).toBeInstanceOf(RefBlueprint);
  expect(typeof k.id).toBe("string");
  expect(k.id.length).toBeGreaterThan(0);
});

test("creates multiple kinds", () => {
  const vi = vivarium();
  vi.element("a", "#111111");
  vi.kind("flammable");
  vi.kind("organic");

  const automaton = vi.create();

  expect(automaton.kinds).toHaveLength(2);
  expect(automaton.kinds.map((k) => k.name)).toStrictEqual(["flammable", "organic"]);
});

// ── Rules: fromId and fromType ──────────────────────────────────────

test("rule from element has correct fromId and fromType", () => {
  const vi = vivarium();
  const alive = vi.element("alive", "#ffffff");
  const dead = vi.element("dead", "#000000");

  alive.to(dead);

  const automaton = vi.create();
  const rule = automaton.rules[0];

  expect(rule.fromId).toBe(alive.id);
  expect(rule.fromType).toBe("element");
});

test("rule from kind has correct fromId and fromType", () => {
  const vi = vivarium();
  const alive = vi.element("alive", "#ffffff");
  const flammable = vi.kind("flammable");

  flammable.to(alive);

  const automaton = vi.create();
  const rule = automaton.rules[0];

  expect(rule.fromId).toBe(flammable.id);
  expect(rule.fromType).toBe("kind");
});

// ── Rules: to with neighbor (Point) ─────────────────────────────────

test("creates a rule with neighbor as target", () => {
  const vi = vivarium();
  const alive = vi.element("alive", "#ffffff");

  alive.to(vi.neighbor.TOP);

  const automaton = vi.create();
  const rule = automaton.rules[0];

  expect(rule.to).toStrictEqual({ x: 0, y: -1 });
});

test("creates a rule with SELF neighbor as target", () => {
  const vi = vivarium();
  const alive = vi.element("alive", "#ffffff");

  alive.to(vi.neighbor.SELF);

  const automaton = vi.create();
  const rule = automaton.rules[0];

  expect(rule.to).toStrictEqual({ x: 0, y: 0 });
});

test("creates a rule with diagonal neighbor as target in square", () => {
  const vi = vivarium();
  const alive = vi.element("alive", "#ffffff");

  alive.to(vi.neighbor.BOTTOM_RIGHT);

  const automaton = vi.create();
  const rule = automaton.rules[0];

  expect(rule.to).toStrictEqual({ x: 1, y: 1 });
});

// ── Multiple rules on same element ──────────────────────────────────

test("creates multiple rules on the same element", () => {
  const vi = vivarium();
  const alive = vi.element("alive", "#ffffff");
  const dead = vi.element("dead", "#000000");

  alive.to(dead);
  alive.to(alive).count(alive, 2, 3);
  dead.to(alive).count(alive, 3);

  const automaton = vi.create();

  expect(automaton.rules).toHaveLength(3);
  expect(automaton.rules[0].fromId).toBe(alive.id);
  expect(automaton.rules[1].fromId).toBe(alive.id);
  expect(automaton.rules[2].fromId).toBe(dead.id);
});

// ── Count condition with multiple individual numbers ────────────────

test("creates a count condition with multiple individual numbers", () => {
  const vi = vivarium();
  const alive = vi.element("alive", "#ffffff");

  alive.to(alive).count(alive, 2, 3);

  const automaton = vi.create();
  const condition = automaton.rules[0].when[0];

  expect(condition.type).toBe("count");
  if (condition.type === "count") {
    expect(condition.count).toStrictEqual([2, 3]);
  }
});

test("creates a count condition with mixed numbers and arrays", () => {
  const vi = vivarium();
  const alive = vi.element("alive", "#ffffff");

  alive.to(alive).count(alive, 1, [2, 3], 5);

  const automaton = vi.create();
  const condition = automaton.rules[0].when[0];

  if (condition.type === "count") {
    expect(condition.count).toStrictEqual([1, 2, 3, 5]);
  }
});

test("creates a count condition checking a neighbor", () => {
  const vi = vivarium();
  const alive = vi.element("alive", "#ffffff");

  alive.to(alive).count(vi.neighbor.TOP, 2);

  const automaton = vi.create();
  const condition = automaton.rules[0].when[0];

  if (condition.type === "count") {
    expect(condition.check).toStrictEqual({ x: 0, y: -1 });
    expect(condition.count).toStrictEqual([2]);
  }
});

test("creates a count condition checking a kind", () => {
  const vi = vivarium();
  const alive = vi.element("alive", "#ffffff");
  const flammable = vi.kind("flammable");

  alive.to(alive).count(flammable, 1, 2);

  const automaton = vi.create();
  const condition = automaton.rules[0].when[0];

  if (condition.type === "count") {
    expect(condition.check).toBe(flammable.id);
    expect(condition.count).toStrictEqual([1, 2]);
  }
});

// ── Is condition ────────────────────────────────────────────────────

test("creates an is condition comparing neighbor with element", () => {
  const vi = vivarium();
  const alive = vi.element("alive", "#ffffff");

  alive.to(alive).is(vi.neighbor.TOP, alive);

  const automaton = vi.create();
  const condition = automaton.rules[0].when[0];

  expect(condition.type).toBe("is");
  if (condition.type === "is") {
    expect(condition.compare).toStrictEqual({ x: 0, y: -1 });
    expect(condition.with).toBe(alive.id);
  }
});

test("creates an is condition comparing neighbor with another neighbor", () => {
  const vi = vivarium();
  const alive = vi.element("alive", "#ffffff");

  alive.to(alive).is(vi.neighbor.LEFT, vi.neighbor.RIGHT);

  const automaton = vi.create();
  const condition = automaton.rules[0].when[0];

  if (condition.type === "is") {
    expect(condition.compare).toStrictEqual({ x: -1, y: 0 });
    expect(condition.with).toStrictEqual({ x: 1, y: 0 });
  }
});

test("creates an is condition with kind", () => {
  const vi = vivarium();
  const alive = vi.element("alive", "#ffffff");
  const flammable = vi.kind("flammable");

  alive.to(alive).is(vi.neighbor.BOTTOM, flammable);

  const automaton = vi.create();
  const condition = automaton.rules[0].when[0];

  if (condition.type === "is") {
    expect(condition.compare).toStrictEqual({ x: 0, y: 1 });
    expect(condition.with).toBe(flammable.id);
  }
});

test("creates an is condition with cross neighborhood", () => {
  const vi = vivarium("cross");
  const alive = vi.element("alive", "#ffffff");

  alive.to(alive).is(vi.neighbor.LEFT, alive);

  const automaton = vi.create();
  const condition = automaton.rules[0].when[0];

  if (condition.type === "is") {
    expect(condition.compare).toStrictEqual({ x: -1, y: 0 });
    expect(condition.with).toBe(alive.id);
  }
});

// ── Chance condition ────────────────────────────────────────────────

test("creates a chance condition with default whole", () => {
  const vi = vivarium();
  const alive = vi.element("alive", "#ffffff");

  alive.to(alive).chance(50);

  const automaton = vi.create();
  const condition = automaton.rules[0].when[0];

  expect(condition.type).toBe("chance");
  if (condition.type === "chance") {
    expect(condition.ratio).toStrictEqual({ part: 50, whole: 100 });
  }
});

test("creates a chance condition with custom whole", () => {
  const vi = vivarium();
  const alive = vi.element("alive", "#ffffff");

  alive.to(alive).chance(1, 10);

  const automaton = vi.create();
  const condition = automaton.rules[0].when[0];

  if (condition.type === "chance") {
    expect(condition.ratio).toStrictEqual({ part: 1, whole: 10 });
  }
});

test("clamps negative part to 0 in chance condition", () => {
  const vi = vivarium();
  const alive = vi.element("alive", "#ffffff");

  alive.to(alive).chance(-5);

  const automaton = vi.create();
  const condition = automaton.rules[0].when[0];

  if (condition.type === "chance") {
    expect(condition.ratio.part).toBe(0);
  }
});

test("clamps zero whole to 1 in chance condition", () => {
  const vi = vivarium();
  const alive = vi.element("alive", "#ffffff");

  alive.to(alive).chance(1, 0);

  const automaton = vi.create();
  const condition = automaton.rules[0].when[0];

  if (condition.type === "chance") {
    expect(condition.ratio.whole).toBe(1);
  }
});

test("clamps negative whole to 1 in chance condition", () => {
  const vi = vivarium();
  const alive = vi.element("alive", "#ffffff");

  alive.to(alive).chance(1, -5);

  const automaton = vi.create();
  const condition = automaton.rules[0].when[0];

  if (condition.type === "chance") {
    expect(condition.ratio.whole).toBe(1);
  }
});

test("clamps part to whole when part exceeds whole in chance condition", () => {
  const vi = vivarium();
  const alive = vi.element("alive", "#ffffff");

  alive.to(alive).chance(200, 100);

  const automaton = vi.create();
  const condition = automaton.rules[0].when[0];

  if (condition.type === "chance") {
    expect(condition.ratio.part).toBe(100);
    expect(condition.ratio.whole).toBe(100);
  }
});

test("floors fractional values in chance condition", () => {
  const vi = vivarium();
  const alive = vi.element("alive", "#ffffff");

  alive.to(alive).chance(3.7, 10.9);

  const automaton = vi.create();
  const condition = automaton.rules[0].when[0];

  if (condition.type === "chance") {
    expect(condition.ratio.part).toBe(3);
    expect(condition.ratio.whole).toBe(10);
  }
});

// ── Accept strategies ───────────────────────────────────────────────

test("sets accept strategy to all", () => {
  const vi = vivarium();
  const alive = vi.element("alive", "#ffffff");

  alive.to(alive).count(alive, 3).accept("all");

  const automaton = vi.create();

  expect(automaton.rules[0].accept).toBe(Accept.ALL);
});

test("sets accept strategy to any", () => {
  const vi = vivarium();
  const alive = vi.element("alive", "#ffffff");

  alive.to(alive).count(alive, 3).accept("any");

  const automaton = vi.create();

  expect(automaton.rules[0].accept).toBe(Accept.ANY);
});

test("sets accept strategy to one", () => {
  const vi = vivarium();
  const alive = vi.element("alive", "#ffffff");

  alive.to(alive).count(alive, 3).accept("one");

  const automaton = vi.create();

  expect(automaton.rules[0].accept).toBe(Accept.ONE);
});

test("sets accept strategy to none", () => {
  const vi = vivarium();
  const alive = vi.element("alive", "#ffffff");

  alive.to(alive).count(alive, 3).accept("none");

  const automaton = vi.create();

  expect(automaton.rules[0].accept).toBe(Accept.NONE);
});

// ── Multiple conditions on same rule (chaining) ─────────────────────

test("chains count and is conditions on the same rule", () => {
  const vi = vivarium();
  const alive = vi.element("alive", "#ffffff");
  const dead = vi.element("dead", "#000000");

  alive.to(alive).count(alive, 2, 3).is(vi.neighbor.TOP, dead);

  const automaton = vi.create();
  const rule = automaton.rules[0];

  expect(rule.when).toHaveLength(2);
  expect(rule.when[0].type).toBe("count");
  expect(rule.when[1].type).toBe("is");
});

test("chains count and chance conditions on the same rule", () => {
  const vi = vivarium();
  const alive = vi.element("alive", "#ffffff");

  alive.to(alive).count(alive, 3).chance(50);

  const automaton = vi.create();
  const rule = automaton.rules[0];

  expect(rule.when).toHaveLength(2);
  expect(rule.when[0].type).toBe("count");
  expect(rule.when[1].type).toBe("chance");
});

test("chains three conditions on the same rule", () => {
  const vi = vivarium();
  const alive = vi.element("alive", "#ffffff");
  const dead = vi.element("dead", "#000000");

  alive.to(alive).count(alive, 2, 3).is(vi.neighbor.LEFT, dead).chance(75);

  const automaton = vi.create();
  const rule = automaton.rules[0];

  expect(rule.when).toHaveLength(3);
  expect(rule.when[0].type).toBe("count");
  expect(rule.when[1].type).toBe("is");
  expect(rule.when[2].type).toBe("chance");
});

test("accept is available after chaining conditions", () => {
  const vi = vivarium();
  const alive = vi.element("alive", "#ffffff");

  alive.to(alive).count(alive, 2).chance(50).accept("any");

  const automaton = vi.create();

  expect(automaton.rules[0].accept).toBe(Accept.ANY);
  expect(automaton.rules[0].when).toHaveLength(2);
});

// ── Conditions have unique ids ──────────────────────────────────────

test("each condition gets a unique id", () => {
  const vi = vivarium();
  const alive = vi.element("alive", "#ffffff");

  alive.to(alive).count(alive, 2).count(alive, 3);

  const automaton = vi.create();
  const ids = automaton.rules[0].when.map((c) => c.id);

  expect(ids[0]).not.toBe(ids[1]);
});

// ── Helpers: not() ──────────────────────────────────────────────────

test("helpers.not excludes a single value for square", () => {
  const vi = vivarium();

  expect(vi.helpers.not(3)).toStrictEqual([0, 1, 2, 4, 5, 6, 7, 8]);
});

test("helpers.not excludes multiple values for square", () => {
  const vi = vivarium();

  expect(vi.helpers.not([2, 3])).toStrictEqual([0, 1, 4, 5, 6, 7, 8]);
});

test("helpers.not excludes a single value for cross", () => {
  const vi = vivarium("cross");

  expect(vi.helpers.not(2)).toStrictEqual([0, 1, 3, 4]);
});

test("helpers.not excludes multiple values for cross", () => {
  const vi = vivarium("cross");

  expect(vi.helpers.not([0, 4])).toStrictEqual([1, 2, 3]);
});

// ── Helpers: between() ──────────────────────────────────────────────

test("helpers.between returns inclusive range", () => {
  const vi = vivarium();

  expect(vi.helpers.between(2, 5)).toStrictEqual([2, 3, 4, 5]);
});

test("helpers.between with min 0", () => {
  const vi = vivarium();

  expect(vi.helpers.between(0, 3)).toStrictEqual([0, 1, 2, 3]);
});

test("helpers.between throws when min is negative", () => {
  const vi = vivarium();

  expect(() => vi.helpers.between(-1, 3)).toThrow(/^min cannot be less than 0/);
});

test("helpers.between throws when max is greater than 8", () => {
  const vi = vivarium();

  expect(() => vi.helpers.between(0, 9)).toThrow(/^max cannot be more than 8/);
});

test("helpers.between throws when max equals min", () => {
  const vi = vivarium();

  expect(() => vi.helpers.between(3, 3)).toThrow(/^max cannot be less or equal than min/);
});

test("helpers.between throws when max is less than min", () => {
  const vi = vivarium();

  expect(() => vi.helpers.between(5, 2)).toThrow(/^max cannot be less or equal than min/);
});

// ── Helpers: even() and odd() ───────────────────────────────────────

test("helpers.even returns even numbers for square", () => {
  const vi = vivarium();

  expect(vi.helpers.even()).toStrictEqual([0, 2, 4, 6, 8]);
});

test("helpers.even returns even numbers for cross", () => {
  const vi = vivarium("cross");

  expect(vi.helpers.even()).toStrictEqual([0, 2, 4]);
});

test("helpers.odd returns odd numbers for square", () => {
  const vi = vivarium();

  expect(vi.helpers.odd()).toStrictEqual([1, 3, 5, 7]);
});

test("helpers.odd returns odd numbers for cross", () => {
  const vi = vivarium("cross");

  expect(vi.helpers.odd()).toStrictEqual([1, 3]);
});

// ── Helpers used in count conditions ────────────────────────────────

test("helpers.not can be used in count condition", () => {
  const vi = vivarium();
  const alive = vi.element("alive", "#ffffff");

  alive.to(alive).count(alive, vi.helpers.not(0));

  const automaton = vi.create();
  const condition = automaton.rules[0].when[0];

  if (condition.type === "count") {
    expect(condition.count).toStrictEqual([1, 2, 3, 4, 5, 6, 7, 8]);
  }
});

test("helpers.between can be used in count condition", () => {
  const vi = vivarium();
  const alive = vi.element("alive", "#ffffff");

  alive.to(alive).count(alive, vi.helpers.between(2, 4));

  const automaton = vi.create();
  const condition = automaton.rules[0].when[0];

  if (condition.type === "count") {
    expect(condition.count).toStrictEqual([2, 3, 4]);
  }
});

test("helpers.even can be used in count condition", () => {
  const vi = vivarium();
  const alive = vi.element("alive", "#ffffff");

  alive.to(alive).count(alive, vi.helpers.even());

  const automaton = vi.create();
  const condition = automaton.rules[0].when[0];

  if (condition.type === "count") {
    expect(condition.count).toStrictEqual([0, 2, 4, 6, 8]);
  }
});

test("helpers.odd can be used in count condition", () => {
  const vi = vivarium();
  const alive = vi.element("alive", "#ffffff");

  alive.to(alive).count(alive, vi.helpers.odd());

  const automaton = vi.create();
  const condition = automaton.rules[0].when[0];

  if (condition.type === "count") {
    expect(condition.count).toStrictEqual([1, 3, 5, 7]);
  }
});

// ── Post-create lockdown on new element/kind ────────────────────────

test("throws on creating element after create", () => {
  const vi = vivarium();
  vi.element("alive", "#ffffff");
  vi.create();

  expect(() => vi.element("dead", "#000000")).toThrow(
    /^Blueprint methods can no longer be called after create/,
  );
});

test("throws on creating kind after create", () => {
  const vi = vivarium();
  vi.element("alive", "#ffffff");
  vi.create();

  expect(() => vi.kind("flammable")).toThrow(
    /^Blueprint methods can no longer be called after create/,
  );
});

test("throws on is condition after create", () => {
  const vi = vivarium();
  const alive = vi.element("alive", "#ffffff");
  const ruleChain = alive.to(alive).count(alive, 3);

  vi.create();

  expect(() => ruleChain.is(Square.TOP, alive)).toThrow(
    /^Blueprint methods can no longer be called after create/,
  );
});

test("throws on chance condition after create", () => {
  const vi = vivarium();
  const alive = vi.element("alive", "#ffffff");
  const ruleChain = alive.to(alive).count(alive, 3);

  vi.create();

  expect(() => ruleChain.chance(50)).toThrow(
    /^Blueprint methods can no longer be called after create/,
  );
});

// ── Real-world automata scenarios ───────────────────────────────────

test("Conway's Game of Life", () => {
  const vi = vivarium();

  const dead = vi.element("dead", "#000000");
  const alive = vi.element("alive", "#ffffff");

  // A dead cell with exactly 3 alive neighbors becomes alive
  dead.to(alive).count(alive, 3);

  // An alive cell with 2 or 3 alive neighbors stays alive
  alive.to(alive).count(alive, 2, 3);

  // An alive cell otherwise dies
  alive.to(dead);

  const automaton = vi.create();

  expect(automaton.neighborhood).toBe("square");
  expect(automaton.elements).toHaveLength(2);
  expect(automaton.rules).toHaveLength(3);

  // Rule 1: dead -> alive when count(alive) == 3
  expect(automaton.rules[0].fromId).toBe(dead.id);
  expect(automaton.rules[0].to).toBe(alive.id);
  expect(automaton.rules[0].when).toHaveLength(1);
  expect(automaton.rules[0].when[0].type).toBe("count");

  // Rule 2: alive -> alive when count(alive) in [2, 3]
  expect(automaton.rules[1].fromId).toBe(alive.id);
  expect(automaton.rules[1].to).toBe(alive.id);
  expect(automaton.rules[1].when).toHaveLength(1);
  if (automaton.rules[1].when[0].type === "count") {
    expect(automaton.rules[1].when[0].count).toStrictEqual([2, 3]);
  }

  // Rule 3: alive -> dead (fallback, no conditions)
  expect(automaton.rules[2].fromId).toBe(alive.id);
  expect(automaton.rules[2].to).toBe(dead.id);
  expect(automaton.rules[2].when).toHaveLength(0);
});

test("Seeds automaton", () => {
  const vi = vivarium();

  const off = vi.element("off", "#000000");
  const on = vi.element("on", "#ffffff");

  // An off cell with exactly 2 on neighbors turns on
  off.to(on).count(on, 2);

  // An on cell always turns off (no conditions)
  on.to(off);

  const automaton = vi.create();

  expect(automaton.elements).toHaveLength(2);
  expect(automaton.rules).toHaveLength(2);

  expect(automaton.rules[0].fromId).toBe(off.id);
  expect(automaton.rules[0].to).toBe(on.id);
  if (automaton.rules[0].when[0].type === "count") {
    expect(automaton.rules[0].when[0].count).toStrictEqual([2]);
  }

  expect(automaton.rules[1].fromId).toBe(on.id);
  expect(automaton.rules[1].to).toBe(off.id);
  expect(automaton.rules[1].when).toHaveLength(0);
});

test("Brian's Brain automaton", () => {
  const vi = vivarium();

  const off = vi.element("off", "#000000");
  const dying = vi.element("dying", "#888888");
  const on = vi.element("on", "#ffffff");

  // An off cell with exactly 2 on neighbors turns on
  off.to(on).count(on, 2);

  // An on cell transitions to dying
  on.to(dying);

  // A dying cell transitions to off
  dying.to(off);

  const automaton = vi.create();

  expect(automaton.elements).toHaveLength(3);
  expect(automaton.rules).toHaveLength(3);

  // off -> on when 2 on neighbors
  expect(automaton.rules[0].fromId).toBe(off.id);
  expect(automaton.rules[0].to).toBe(on.id);

  // on -> dying (unconditional)
  expect(automaton.rules[1].fromId).toBe(on.id);
  expect(automaton.rules[1].to).toBe(dying.id);
  expect(automaton.rules[1].when).toHaveLength(0);

  // dying -> off (unconditional)
  expect(automaton.rules[2].fromId).toBe(dying.id);
  expect(automaton.rules[2].to).toBe(off.id);
  expect(automaton.rules[2].when).toHaveLength(0);
});

test("Wireworld-like automaton with kinds", () => {
  const vi = vivarium();

  const empty = vi.element("empty", "#000000");
  const conductor = vi.kind("conductor");
  const wire = vi.element("wire", "#ff8800", [conductor]);
  const head = vi.element("electron-head", "#0088ff", [conductor]);
  const tail = vi.element("electron-tail", "#ffffff", [conductor]);

  // Electron head becomes tail
  head.to(tail);

  // Electron tail becomes wire
  tail.to(wire);

  // Wire becomes head if 1 or 2 neighbors are electron heads
  wire.to(head).count(head, 1, 2);

  const automaton = vi.create();

  expect(automaton.elements).toHaveLength(4);
  expect(automaton.kinds).toHaveLength(1);
  expect(automaton.rules).toHaveLength(3);

  // Verify conductor kind is linked to wire, head, and tail
  const conductorKind = automaton.kinds.find((k) => k.name === "conductor");
  expect(conductorKind).toBeDefined();

  const wireEl = automaton.elements.find((e) => e.name === "wire");
  const headEl = automaton.elements.find((e) => e.name === "electron-head");
  const tailEl = automaton.elements.find((e) => e.name === "electron-tail");

  expect(wireEl?.extensions).toContain(conductorKind?.id);
  expect(headEl?.extensions).toContain(conductorKind?.id);
  expect(tailEl?.extensions).toContain(conductorKind?.id);

  // Empty has no extensions
  expect(empty.id).toBe(automaton.elements.find((e) => e.name === "empty")?.id);
  const emptyEl = automaton.elements.find((e) => e.name === "empty");
  expect(emptyEl?.extensions).toHaveLength(0);
});

test("Game of Life variant using helpers.not for survival", () => {
  const vi = vivarium();

  const dead = vi.element("dead", "#000000");
  const alive = vi.element("alive", "#ffffff");

  // Birth: exactly 3 alive neighbors
  dead.to(alive).count(alive, 3);

  // Survival: NOT 0 or 1 alive neighbors (i.e., 2-8)
  // Using helpers.not to express "everything except 0 and 1"
  alive.to(alive).count(alive, vi.helpers.not([0, 1]));

  alive.to(dead);

  const automaton = vi.create();
  const survivalCondition = automaton.rules[1].when[0];

  if (survivalCondition.type === "count") {
    expect(survivalCondition.count).toStrictEqual([2, 3, 4, 5, 6, 7, 8]);
  }
});

test("cross neighborhood automaton with is condition", () => {
  const vi = vivarium("cross");

  const a = vi.element("a", "#ff0000");
  const b = vi.element("b", "#00ff00");

  // A cell of type 'a' transitions to 'b' if its top neighbor is 'b'
  a.to(b).is(vi.neighbor.TOP, b);

  const automaton = vi.create();

  expect(automaton.neighborhood).toBe("cross");
  expect(automaton.rules).toHaveLength(1);

  const rule = automaton.rules[0];
  expect(rule.fromId).toBe(a.id);
  expect(rule.to).toBe(b.id);

  const condition = rule.when[0];
  if (condition.type === "is") {
    expect(condition.compare).toStrictEqual({ x: 0, y: -1 });
    expect(condition.with).toBe(b.id);
  }
});

test("stochastic automaton with chance and count", () => {
  const vi = vivarium();

  const empty = vi.element("empty", "#000000");
  const tree = vi.element("tree", "#00ff00");
  const fire = vi.element("fire", "#ff0000");

  // Empty grows a tree with 1% chance
  empty.to(tree).chance(1);

  // A tree catches fire if any neighbor is on fire
  tree.to(fire).count(fire, vi.helpers.between(1, 8)).accept("any");

  // A tree can spontaneously combust with 0.01% chance
  tree.to(fire).chance(1, 10000);

  // Fire burns out
  fire.to(empty);

  const automaton = vi.create();

  expect(automaton.elements).toHaveLength(3);
  expect(automaton.rules).toHaveLength(4);

  // Rule 0: empty -> tree with 1/100 chance
  const chanceCondition = automaton.rules[0].when[0];
  if (chanceCondition.type === "chance") {
    expect(chanceCondition.ratio).toStrictEqual({ part: 1, whole: 100 });
  }

  // Rule 1: tree -> fire with count and accept any
  expect(automaton.rules[1].accept).toBe(Accept.ANY);
  if (automaton.rules[1].when[0].type === "count") {
    expect(automaton.rules[1].when[0].count).toStrictEqual([1, 2, 3, 4, 5, 6, 7, 8]);
  }

  // Rule 2: tree -> fire with 1/10000 chance
  const spontaneousChance = automaton.rules[2].when[0];
  if (spontaneousChance.type === "chance") {
    expect(spontaneousChance.ratio).toStrictEqual({ part: 1, whole: 10000 });
  }

  // Rule 3: fire -> empty (unconditional)
  expect(automaton.rules[3].when).toHaveLength(0);
});

test("rule with none accept strategy ignores all conditions", () => {
  const vi = vivarium();
  const alive = vi.element("alive", "#ffffff");
  const dead = vi.element("dead", "#000000");

  // Even though count is specified, "none" means no condition should match
  alive.to(dead).count(alive, 3).accept("none");

  const automaton = vi.create();
  const rule = automaton.rules[0];

  expect(rule.accept).toBe(Accept.NONE);
  expect(rule.when).toHaveLength(1);
});

test("automaton with only unconditional rules", () => {
  const vi = vivarium();
  const a = vi.element("a", "#ff0000");
  const b = vi.element("b", "#00ff00");

  // Oscillator: a -> b, b -> a
  a.to(b);
  b.to(a);

  const automaton = vi.create();

  expect(automaton.rules).toHaveLength(2);
  expect(automaton.rules[0].when).toHaveLength(0);
  expect(automaton.rules[1].when).toHaveLength(0);
  expect(automaton.rules[0].accept).toBe(Accept.ALL);
  expect(automaton.rules[1].accept).toBe(Accept.ALL);
});
