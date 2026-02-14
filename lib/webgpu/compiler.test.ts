import { describe, expect, test } from "vitest";
import { Accept, Opcode, Square, To } from "@/common/constants";
import { vivarium } from "@/vivarium/vivarium";
import { compileGpuAutomaton } from "./compiler";

// ── Helpers ─────────────────────────────────────────────────────────

// -1 as unsigned 32-bit integer (used in GPU vec2u for neighbor offsets)
const NEG1 = -1 >>> 0;

/**
 * Build an automaton with the vivarium API and compile it to GPU structures.
 * This is the standard pattern for all tests in this file.
 */
const compile = (
  build: (vi: ReturnType<typeof vivarium>) => void,
  neighborhood?: "square" | "cross"
) => {
  const vi = vivarium(neighborhood);
  build(vi);
  return compileGpuAutomaton(vi.create());
};

// ── Basic compilation ───────────────────────────────────────────────

describe("basic compilation", () => {
  test("single element with no rules produces one gpuElement and empty rules/conditions", () => {
    const gpu = compile((vi) => {
      vi.element("dead", "#000000");
    });

    expect(gpu.gpuElements).toHaveLength(1);
    expect(gpu.gpuRules).toHaveLength(0);
    expect(gpu.gpuConditions).toHaveLength(0);

    // element has no rules
    expect(gpu.gpuElements[0].ruleStart).toBe(0);
    expect(gpu.gpuElements[0].ruleEnd).toBe(0);
  });

  test("square neighborhood compiles to 8", () => {
    const gpu = compile((vi) => {
      vi.element("a", "#000000");
    });

    expect(gpu.gpuNeighborhood).toBe(8);
  });

  test("cross neighborhood compiles to 4", () => {
    const gpu = compile((vi) => {
      vi.element("a", "#000000");
    }, "cross");

    expect(gpu.gpuNeighborhood).toBe(4);
  });

  test("element colors are converted to ABGR format", () => {
    const gpu = compile((vi) => {
      vi.element("red", "#ff0000");
    });

    // #ff0000 -> R=0xff, G=0x00, B=0x00 -> ABGR = 0xFF0000FF
    expect(gpu.gpuElements[0].color).toBe(0xff0000ff);
  });

  test("unconditional rule compiles with empty condition range", () => {
    const gpu = compile((vi) => {
      const a = vi.element("a", "#ff0000");
      const b = vi.element("b", "#00ff00");
      a.to(b);
    });

    expect(gpu.gpuRules).toHaveLength(1);
    expect(gpu.gpuConditions).toHaveLength(0);

    expect(gpu.gpuRules[0].conditionsStart).toBe(0);
    expect(gpu.gpuRules[0].conditionsEnd).toBe(0);
  });
});

// ── To targets ──────────────────────────────────────────────────────

describe("rule to targets", () => {
  test("to element compiles with ELEMENT_ID type", () => {
    const gpu = compile((vi) => {
      const a = vi.element("a", "#ff0000");
      const b = vi.element("b", "#00ff00");
      a.to(b);
    });

    expect(gpu.gpuRules[0].toType).toBe(To.ELEMENT_ID);
    expect(gpu.gpuRules[0].toId).toBe(1); // b is index 1
  });

  test("to neighbor compiles with POINT type", () => {
    const gpu = compile((vi) => {
      const a = vi.element("a", "#ff0000");
      vi.element("b", "#00ff00");
      a.to(Square.TOP);
    });

    expect(gpu.gpuRules[0].toType).toBe(To.POINT);
    // Square.TOP = { x: 0, y: -1 } stored as unsigned 32-bit
    expect(gpu.gpuRules[0].toNeighbor.x).toBe(0);
    expect(gpu.gpuRules[0].toNeighbor.y).toBe(NEG1);
  });
});

// ── COUNT_ELEMENT opcode ────────────────────────────────────────────

describe("COUNT_ELEMENT opcode", () => {
  test("counting a specific element uses COUNT_ELEMENT opcode", () => {
    const gpu = compile((vi) => {
      const dead = vi.element("dead", "#000000");
      const alive = vi.element("alive", "#ffffff");
      dead.to(alive).count(alive, 3);
    });

    expect(gpu.gpuConditions).toHaveLength(1);
    expect(gpu.gpuConditions[0].opcode).toBe(Opcode.COUNT_ELEMENT);
    expect(gpu.gpuConditions[0].checkId).toBe(1); // alive is index 1
  });

  test("count values are packed as bit flags", () => {
    const gpu = compile((vi) => {
      const dead = vi.element("dead", "#000000");
      const alive = vi.element("alive", "#ffffff");
      // count [2, 3] -> 0b000001100 = 12
      dead.to(alive).count(alive, 2, 3);
    });

    // Packed: (1 << 2) | (1 << 3) = 4 | 8 = 12
    expect(gpu.gpuConditions[0].countOrWithId).toBe(12);
  });

  test("count of exactly 0 packs to bit 0", () => {
    const gpu = compile((vi) => {
      const a = vi.element("a", "#ff0000");
      const b = vi.element("b", "#00ff00");
      // count [0] -> 0b000000001 = 1
      a.to(b).count(b, 0);
    });

    expect(gpu.gpuConditions[0].countOrWithId).toBe(1);
  });

  test("count of exactly 8 packs to bit 8", () => {
    const gpu = compile((vi) => {
      const a = vi.element("a", "#ff0000");
      const b = vi.element("b", "#00ff00");
      // count [8] -> 1 << 8 = 256
      a.to(b).count(b, 8);
    });

    expect(gpu.gpuConditions[0].countOrWithId).toBe(256);
  });

  test("full range count packs all bits 0-8", () => {
    const gpu = compile((vi) => {
      const a = vi.element("a", "#ff0000");
      const b = vi.element("b", "#00ff00");
      // count [0,1,2,3,4,5,6,7,8] -> 0b111111111 = 511
      a.to(b).count(b, 0, 1, 2, 3, 4, 5, 6, 7, 8);
    });

    expect(gpu.gpuConditions[0].countOrWithId).toBe(511);
  });
});

// ── COUNT_POINT opcode ──────────────────────────────────────────────

describe("COUNT_POINT opcode", () => {
  test("counting by neighbor position uses COUNT_POINT opcode", () => {
    const gpu = compile((vi) => {
      const a = vi.element("a", "#ff0000");
      const b = vi.element("b", "#00ff00");
      // Count whatever is at TOP position
      a.to(b).count(Square.TOP, 2);
    });

    expect(gpu.gpuConditions).toHaveLength(1);
    expect(gpu.gpuConditions[0].opcode).toBe(Opcode.COUNT_POINT);
    // TOP = { x: 0, y: -1 } stored as unsigned 32-bit
    expect(gpu.gpuConditions[0].checkPointOrComparePoint.x).toBe(0);
    expect(gpu.gpuConditions[0].checkPointOrComparePoint.y).toBe(NEG1);
  });

  test("count value for point count is packed as bit flags", () => {
    const gpu = compile((vi) => {
      const a = vi.element("a", "#ff0000");
      const b = vi.element("b", "#00ff00");
      a.to(b).count(Square.TOP_LEFT, 1, 3);
    });

    // Packed: (1 << 1) | (1 << 3) = 2 | 8 = 10
    expect(gpu.gpuConditions[0].countOrWithId).toBe(10);
  });
});

// ── COUNT_KIND opcode ───────────────────────────────────────────────

describe("COUNT_KIND opcode", () => {
  test("counting by kind uses COUNT_KIND opcode with packed element IDs", () => {
    const gpu = compile((vi) => {
      const conductor = vi.kind("conductor");
      const _empty = vi.element("empty", "#000000");
      const wire = vi.element("wire", "#ff8800", [conductor]);
      const head = vi.element("head", "#0088ff", [conductor]);
      vi.element("tail", "#ffffff", [conductor]);

      // wire -> head when counting conductor neighbors
      wire.to(head).count(conductor, 1, 2);
    });

    expect(gpu.gpuConditions).toHaveLength(1);
    expect(gpu.gpuConditions[0].opcode).toBe(Opcode.COUNT_KIND);

    // conductor includes wire(1), head(2), tail(3)
    // packed: (1 << 1) | (1 << 2) | (1 << 3) = 2 | 4 | 8 = 14
    expect(gpu.gpuConditions[0].checkId).toBe(14);

    // count [1, 2] -> (1 << 1) | (1 << 2) = 2 | 4 = 6
    expect(gpu.gpuConditions[0].countOrWithId).toBe(6);
  });
});

// ── IS_ELEMENT opcode ───────────────────────────────────────────────

describe("IS_ELEMENT opcode", () => {
  test("is condition with element uses IS_ELEMENT opcode", () => {
    const gpu = compile((vi) => {
      const a = vi.element("a", "#ff0000");
      const b = vi.element("b", "#00ff00");
      // if TOP neighbor is b, transition a -> b
      a.to(b).is(Square.TOP, b);
    });

    expect(gpu.gpuConditions).toHaveLength(1);
    expect(gpu.gpuConditions[0].opcode).toBe(Opcode.IS_ELEMENT);
    // compare point is TOP = { x: 0, y: -1 } stored as unsigned 32-bit
    expect(gpu.gpuConditions[0].checkPointOrComparePoint.x).toBe(0);
    expect(gpu.gpuConditions[0].checkPointOrComparePoint.y).toBe(NEG1);
    // with id is b = index 1
    expect(gpu.gpuConditions[0].countOrWithId).toBe(1);
  });
});

// ── IS_POINT opcode ─────────────────────────────────────────────────

describe("IS_POINT opcode", () => {
  test("is condition comparing two positions uses IS_POINT opcode", () => {
    const gpu = compile((vi) => {
      const a = vi.element("a", "#ff0000");
      const b = vi.element("b", "#00ff00");
      // if TOP == BOTTOM, transition a -> b
      a.to(b).is(Square.TOP, Square.BOTTOM);
    });

    expect(gpu.gpuConditions).toHaveLength(1);
    expect(gpu.gpuConditions[0].opcode).toBe(Opcode.IS_POINT);
    // compare point is TOP = { x: 0, y: -1 } stored as unsigned 32-bit
    expect(gpu.gpuConditions[0].checkPointOrComparePoint.x).toBe(0);
    expect(gpu.gpuConditions[0].checkPointOrComparePoint.y).toBe(NEG1);
    // with point is BOTTOM = { x: 0, y: 1 }
    expect(gpu.gpuConditions[0].withPoint.x).toBe(0);
    expect(gpu.gpuConditions[0].withPoint.y).toBe(1);
  });
});

// ── IS_KIND opcode ──────────────────────────────────────────────────

describe("IS_KIND opcode", () => {
  test("is condition with kind uses IS_KIND opcode and packed IDs", () => {
    const gpu = compile((vi) => {
      const conductor = vi.kind("conductor");
      const empty = vi.element("empty", "#000000");
      const wire = vi.element("wire", "#ff8800", [conductor]);
      const _head = vi.element("head", "#0088ff", [conductor]);
      vi.element("tail", "#ffffff", [conductor]);

      // empty -> wire if TOP is any conductor
      empty.to(wire).is(Square.TOP, conductor);
    });

    expect(gpu.gpuConditions).toHaveLength(1);
    expect(gpu.gpuConditions[0].opcode).toBe(Opcode.IS_KIND);

    // compare point is TOP = { x: 0, y: -1 } stored as unsigned 32-bit
    expect(gpu.gpuConditions[0].checkPointOrComparePoint.x).toBe(0);
    expect(gpu.gpuConditions[0].checkPointOrComparePoint.y).toBe(NEG1);

    // conductor includes wire(1), head(2), tail(3)
    // packed: (1 << 1) | (1 << 2) | (1 << 3) = 14
    expect(gpu.gpuConditions[0].countOrWithId).toBe(14);
  });
});

// ── CHANCE opcode ───────────────────────────────────────────────────

describe("CHANCE opcode", () => {
  test("chance condition uses CHANCE opcode with ratio", () => {
    const gpu = compile((vi) => {
      const a = vi.element("a", "#ff0000");
      const b = vi.element("b", "#00ff00");
      a.to(b).chance(1, 100);
    });

    expect(gpu.gpuConditions).toHaveLength(1);
    expect(gpu.gpuConditions[0].opcode).toBe(Opcode.CHANCE);
    expect(gpu.gpuConditions[0].chance).toBeCloseTo(0.01);
  });

  test("chance 50/100 compiles to 0.5", () => {
    const gpu = compile((vi) => {
      const a = vi.element("a", "#ff0000");
      const b = vi.element("b", "#00ff00");
      a.to(b).chance(50, 100);
    });

    expect(gpu.gpuConditions[0].chance).toBeCloseTo(0.5);
  });

  test("chance 1/10000 compiles to 0.0001", () => {
    const gpu = compile((vi) => {
      const a = vi.element("a", "#ff0000");
      const b = vi.element("b", "#00ff00");
      a.to(b).chance(1, 10000);
    });

    expect(gpu.gpuConditions[0].chance).toBeCloseTo(0.0001);
  });
});

// ── Accept strategies ───────────────────────────────────────────────

describe("accept strategies", () => {
  test("default accept is ALL", () => {
    const gpu = compile((vi) => {
      const a = vi.element("a", "#ff0000");
      const b = vi.element("b", "#00ff00");
      a.to(b).count(b, 3);
    });

    expect(gpu.gpuRules[0].accept).toBe(Accept.ALL);
  });

  test("accept any compiles correctly", () => {
    const gpu = compile((vi) => {
      const a = vi.element("a", "#ff0000");
      const b = vi.element("b", "#00ff00");
      a.to(b).count(b, 3).accept("any");
    });

    expect(gpu.gpuRules[0].accept).toBe(Accept.ANY);
  });

  test("accept one compiles correctly", () => {
    const gpu = compile((vi) => {
      const a = vi.element("a", "#ff0000");
      const b = vi.element("b", "#00ff00");
      a.to(b).count(b, 3).accept("one");
    });

    expect(gpu.gpuRules[0].accept).toBe(Accept.ONE);
  });

  test("accept none compiles correctly", () => {
    const gpu = compile((vi) => {
      const a = vi.element("a", "#ff0000");
      const b = vi.element("b", "#00ff00");
      a.to(b).count(b, 3).accept("none");
    });

    expect(gpu.gpuRules[0].accept).toBe(Accept.NONE);
  });
});

// ── Rule-element assignment ─────────────────────────────────────────

describe("rule-element assignment", () => {
  test("multiple rules for same element are contiguous", () => {
    const gpu = compile((vi) => {
      const dead = vi.element("dead", "#000000");
      const alive = vi.element("alive", "#ffffff");

      alive.to(alive).count(alive, 2, 3);
      alive.to(dead);
    });

    // dead has no rules
    expect(gpu.gpuElements[0].ruleStart).toBe(0);
    expect(gpu.gpuElements[0].ruleEnd).toBe(0);

    // alive has 2 rules
    expect(gpu.gpuElements[1].ruleStart).toBe(0);
    expect(gpu.gpuElements[1].ruleEnd).toBe(2);
  });

  test("rules assigned to correct elements with multiple elements", () => {
    const gpu = compile((vi) => {
      const dead = vi.element("dead", "#000000");
      const alive = vi.element("alive", "#ffffff");

      dead.to(alive).count(alive, 3);
      alive.to(alive).count(alive, 2, 3);
      alive.to(dead);
    });

    // dead has 1 rule [0, 1)
    expect(gpu.gpuElements[0].ruleStart).toBe(0);
    expect(gpu.gpuElements[0].ruleEnd).toBe(1);

    // alive has 2 rules [1, 3)
    expect(gpu.gpuElements[1].ruleStart).toBe(1);
    expect(gpu.gpuElements[1].ruleEnd).toBe(3);

    expect(gpu.gpuRules).toHaveLength(3);
  });
});

// ── Kind rule merging ───────────────────────────────────────────────

describe("kind rule merging", () => {
  test("kind rules are merged into all extending elements", () => {
    const gpu = compile((vi) => {
      const conductor = vi.kind("conductor");
      vi.element("empty", "#000000");
      const _wire = vi.element("wire", "#ff8800", [conductor]);
      const _head = vi.element("head", "#0088ff", [conductor]);
      const _tail = vi.element("tail", "#ffffff", [conductor]);

      // kind rule: all conductors with no rules just stay themselves
      conductor.to(Square.SELF);
    });

    // empty has no rules
    expect(gpu.gpuElements[0].ruleStart).toBe(0);
    expect(gpu.gpuElements[0].ruleEnd).toBe(0);

    // wire, head, tail each get the kind rule
    expect(gpu.gpuElements[1].ruleEnd - gpu.gpuElements[1].ruleStart).toBe(1);
    expect(gpu.gpuElements[2].ruleEnd - gpu.gpuElements[2].ruleStart).toBe(1);
    expect(gpu.gpuElements[3].ruleEnd - gpu.gpuElements[3].ruleStart).toBe(1);

    // 3 rules total (one per extending element)
    expect(gpu.gpuRules).toHaveLength(3);
  });

  test("element rules come before kind rules for the same element", () => {
    const gpu = compile((vi) => {
      const conductor = vi.kind("conductor");
      vi.element("empty", "#000000");
      const wire = vi.element("wire", "#ff8800", [conductor]);
      const head = vi.element("head", "#0088ff", [conductor]);
      vi.element("tail", "#ffffff", [conductor]);

      // element-specific rule for wire
      wire.to(head).count(head, 1, 2);

      // kind rule: all conductors fallback
      conductor.to(Square.SELF);
    });

    // wire has 2 rules: its own + the kind rule
    const wireStart = gpu.gpuElements[1].ruleStart;
    const wireEnd = gpu.gpuElements[1].ruleEnd;
    expect(wireEnd - wireStart).toBe(2);

    // First rule is the element-specific one (COUNT_ELEMENT for head)
    const wireRule0 = gpu.gpuRules[wireStart];
    expect(wireRule0.toType).toBe(To.ELEMENT_ID);
    expect(wireRule0.toId).toBe(2); // head is index 2

    // Second rule is the kind rule (POINT/SELF)
    const wireRule1 = gpu.gpuRules[wireStart + 1];
    expect(wireRule1.toType).toBe(To.POINT);
  });

  test("kind rule conditions are deduplicated across extending elements", () => {
    const gpu = compile((vi) => {
      const conductor = vi.kind("conductor");
      vi.element("empty", "#000000");
      const _wire = vi.element("wire", "#ff8800", [conductor]);
      const head = vi.element("head", "#0088ff", [conductor]);
      vi.element("tail", "#ffffff", [conductor]);

      // kind rule with conditions
      conductor.to(Square.SELF).count(head, 1, 2);
    });

    // 3 rules (one per extending element), but only 1 condition (shared)
    expect(gpu.gpuRules).toHaveLength(3);
    expect(gpu.gpuConditions).toHaveLength(1);

    // All rules point to the same condition range
    const condStart = gpu.gpuRules[0].conditionsStart;
    const condEnd = gpu.gpuRules[0].conditionsEnd;
    expect(gpu.gpuRules[1].conditionsStart).toBe(condStart);
    expect(gpu.gpuRules[1].conditionsEnd).toBe(condEnd);
    expect(gpu.gpuRules[2].conditionsStart).toBe(condStart);
    expect(gpu.gpuRules[2].conditionsEnd).toBe(condEnd);
  });
});

// ── Multiple conditions per rule ────────────────────────────────────

describe("multiple conditions per rule", () => {
  test("rule with count and chance produces two conditions", () => {
    const gpu = compile((vi) => {
      const a = vi.element("a", "#ff0000");
      const b = vi.element("b", "#00ff00");
      a.to(b).count(b, 3).chance(50);
    });

    expect(gpu.gpuConditions).toHaveLength(2);
    expect(gpu.gpuConditions[0].opcode).toBe(Opcode.COUNT_ELEMENT);
    expect(gpu.gpuConditions[1].opcode).toBe(Opcode.CHANCE);
  });

  test("rule with count and is produces two conditions", () => {
    const gpu = compile((vi) => {
      const a = vi.element("a", "#ff0000");
      const b = vi.element("b", "#00ff00");
      a.to(b).count(b, 3).is(Square.TOP, b);
    });

    expect(gpu.gpuConditions).toHaveLength(2);
    expect(gpu.gpuConditions[0].opcode).toBe(Opcode.COUNT_ELEMENT);
    expect(gpu.gpuConditions[1].opcode).toBe(Opcode.IS_ELEMENT);

    // condition range spans both
    expect(gpu.gpuRules[0].conditionsStart).toBe(0);
    expect(gpu.gpuRules[0].conditionsEnd).toBe(2);
  });
});

// ── Edge cases ──────────────────────────────────────────────────────

describe("edge cases", () => {
  test("element at index 0 used in kind condition packs correctly", () => {
    const gpu = compile((vi) => {
      const k = vi.kind("k");
      const a = vi.element("a", "#ff0000", [k]);
      const b = vi.element("b", "#00ff00");

      b.to(a).count(k, 1);
    });

    expect(gpu.gpuConditions[0].opcode).toBe(Opcode.COUNT_KIND);
    // a is index 0 -> packed = (1 << 0) = 1
    expect(gpu.gpuConditions[0].checkId).toBe(1);
  });

  test("throws when element index > 31 is used in kind condition", () => {
    expect(() => {
      compile((vi) => {
        const k = vi.kind("k");

        // Create 32 elements without the kind extension
        for (let i = 0; i < 32; i++) {
          vi.element(`e${i}`, `#${i.toString(16).padStart(6, "0")}`);
        }

        // Element at index 32 extends kind k
        const e32 = vi.element("e32", "#ffffff", [k]);
        vi.element("trigger", "#aabbcc");

        // This count condition references kind k, which includes element index 32
        e32.to(e32).count(k, 1);
      });
    }).toThrow(/index greater than 31/);
  });

  test("element with no rules has equal ruleStart and ruleEnd", () => {
    const gpu = compile((vi) => {
      const a = vi.element("a", "#ff0000");
      vi.element("b", "#00ff00");
      const c = vi.element("c", "#0000ff");

      // Only a and c have rules
      a.to(c);
      c.to(a);
    });

    // b has no rules
    expect(gpu.gpuElements[1].ruleStart).toBe(gpu.gpuElements[1].ruleEnd);
  });

  test("conditions from separate rules are stored sequentially", () => {
    const gpu = compile((vi) => {
      const a = vi.element("a", "#ff0000");
      const b = vi.element("b", "#00ff00");

      a.to(b).count(b, 3);
      a.to(b).chance(50);
    });

    // Rule 0 has condition at index 0
    expect(gpu.gpuRules[0].conditionsStart).toBe(0);
    expect(gpu.gpuRules[0].conditionsEnd).toBe(1);

    // Rule 1 has condition at index 1
    expect(gpu.gpuRules[1].conditionsStart).toBe(1);
    expect(gpu.gpuRules[1].conditionsEnd).toBe(2);

    expect(gpu.gpuConditions).toHaveLength(2);
  });
});

// ── Real-world automata compilation ─────────────────────────────────

describe("real-world automata compilation", () => {
  test("Conway's Game of Life compiles correctly", () => {
    const gpu = compile((vi) => {
      const dead = vi.element("dead", "#000000");
      const alive = vi.element("alive", "#ffffff");

      dead.to(alive).count(alive, 3);
      alive.to(alive).count(alive, 2, 3);
      alive.to(dead);
    });

    expect(gpu.gpuNeighborhood).toBe(8);
    expect(gpu.gpuElements).toHaveLength(2);
    expect(gpu.gpuRules).toHaveLength(3);
    expect(gpu.gpuConditions).toHaveLength(2);

    // dead: 1 rule [0, 1)
    expect(gpu.gpuElements[0].ruleStart).toBe(0);
    expect(gpu.gpuElements[0].ruleEnd).toBe(1);

    // alive: 2 rules [1, 3)
    expect(gpu.gpuElements[1].ruleStart).toBe(1);
    expect(gpu.gpuElements[1].ruleEnd).toBe(3);

    // Rule 0: dead -> alive, count(alive, 3)
    expect(gpu.gpuRules[0].toType).toBe(To.ELEMENT_ID);
    expect(gpu.gpuRules[0].toId).toBe(1); // alive
    expect(gpu.gpuConditions[0].opcode).toBe(Opcode.COUNT_ELEMENT);
    expect(gpu.gpuConditions[0].checkId).toBe(1); // alive
    expect(gpu.gpuConditions[0].countOrWithId).toBe(1 << 3); // exactly 3

    // Rule 1: alive -> alive, count(alive, 2, 3)
    expect(gpu.gpuRules[1].toId).toBe(1); // alive
    expect(gpu.gpuConditions[1].opcode).toBe(Opcode.COUNT_ELEMENT);
    expect(gpu.gpuConditions[1].checkId).toBe(1); // alive
    expect(gpu.gpuConditions[1].countOrWithId).toBe((1 << 2) | (1 << 3)); // 2 or 3

    // Rule 2: alive -> dead (unconditional)
    expect(gpu.gpuRules[2].toId).toBe(0); // dead
    expect(gpu.gpuRules[2].conditionsStart).toBe(gpu.gpuRules[2].conditionsEnd);
  });

  test("oscillator (unconditional back-and-forth) compiles correctly", () => {
    const gpu = compile((vi) => {
      const a = vi.element("a", "#ff0000");
      const b = vi.element("b", "#00ff00");
      a.to(b);
      b.to(a);
    });

    expect(gpu.gpuElements).toHaveLength(2);
    expect(gpu.gpuRules).toHaveLength(2);
    expect(gpu.gpuConditions).toHaveLength(0);

    // a -> b
    expect(gpu.gpuRules[0].toType).toBe(To.ELEMENT_ID);
    expect(gpu.gpuRules[0].toId).toBe(1);

    // b -> a
    expect(gpu.gpuRules[1].toType).toBe(To.ELEMENT_ID);
    expect(gpu.gpuRules[1].toId).toBe(0);
  });

  test("fire spread automaton compiles chance and count opcodes", () => {
    const gpu = compile((vi) => {
      const empty = vi.element("empty", "#000000");
      const tree = vi.element("tree", "#00ff00");
      const fire = vi.element("fire", "#ff0000");

      empty.to(tree).chance(1);
      tree.to(fire).count(fire, vi.helpers.between(1, 8)).accept("any");
      tree.to(fire).chance(1, 10000);
      fire.to(empty);
    });

    expect(gpu.gpuElements).toHaveLength(3);
    expect(gpu.gpuRules).toHaveLength(4);

    // Rule 0 condition: CHANCE 1/100
    expect(gpu.gpuConditions[0].opcode).toBe(Opcode.CHANCE);
    expect(gpu.gpuConditions[0].chance).toBeCloseTo(0.01);

    // Rule 1 condition: COUNT_ELEMENT for fire
    expect(gpu.gpuConditions[1].opcode).toBe(Opcode.COUNT_ELEMENT);
    expect(gpu.gpuConditions[1].checkId).toBe(2); // fire
    expect(gpu.gpuRules[1].accept).toBe(Accept.ANY);

    // Rule 2 condition: CHANCE 1/10000
    expect(gpu.gpuConditions[2].opcode).toBe(Opcode.CHANCE);
    expect(gpu.gpuConditions[2].chance).toBeCloseTo(0.0001);

    // Rule 3: fire -> empty (unconditional)
    expect(gpu.gpuRules[3].conditionsStart).toBe(gpu.gpuRules[3].conditionsEnd);
  });

  test("Wireworld compiles kind count conditions correctly", () => {
    const gpu = compile((vi) => {
      const conductor = vi.kind("conductor");
      vi.element("empty", "#000000");
      const wire = vi.element("wire", "#ff8800", [conductor]);
      const head = vi.element("head", "#0088ff", [conductor]);
      const tail = vi.element("tail", "#ffffff", [conductor]);

      head.to(tail);
      tail.to(wire);
      wire.to(head).count(head, 1, 2);
    });

    expect(gpu.gpuElements).toHaveLength(4);
    expect(gpu.gpuRules).toHaveLength(3);

    // wire -> head with count condition
    const wireRuleIdx = gpu.gpuElements[1].ruleStart;
    const wireRule = gpu.gpuRules[wireRuleIdx];
    expect(wireRule.toType).toBe(To.ELEMENT_ID);
    expect(wireRule.toId).toBe(2); // head

    // The count condition: counting head (element, not kind)
    const condIdx = wireRule.conditionsStart;
    expect(gpu.gpuConditions[condIdx].opcode).toBe(Opcode.COUNT_ELEMENT);
    expect(gpu.gpuConditions[condIdx].checkId).toBe(2); // head
    expect(gpu.gpuConditions[condIdx].countOrWithId).toBe((1 << 1) | (1 << 2)); // 1 or 2
  });
});
