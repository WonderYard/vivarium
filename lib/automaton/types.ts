import type { Accept } from "@/common/constants";

export type Neighborhood = "square" | "cross";

export type RefType = "element" | "kind";

export type Kind = {
  type: "kind";
  id: string;
  name: string;
};

export type Color = string;

export type Automaton = {
  neighborhood: Neighborhood;
  wrapping: boolean;
  elements: Element[];
  kinds: Kind[];
  rules: Rule[];
};

export type Element = {
  type: "element";
  id: string;
  name: string;
  color: string;
  extensions: string[];
};

/*
 * ADR: optional targets in rules and conditions
 *
 * `Rule.to`, `Count.check`, `Is.compare` and `Is.with` accept `undefined` even though the
 * blueprint builder always sets them. An automaton can also be assembled step by step outside
 * the builder (for example in an editor where a rule exists before its target is picked), and
 * these types let such a partial automaton be stored with the library's own types. Callers must
 * drop incomplete rules and conditions before passing the automaton to `setup`/`setAutomaton`;
 * the compiler throws on any `undefined` it still finds instead of compiling a wrong rule.
 */
export type Rule = {
  fromId: string;
  fromType: RefType;
  to: string | Point | undefined;
  when: Condition[];
  accept: Accept;
};

export type Condition = Count | Is | Chance;

export type Count = {
  type: "count";
  id: string;
  check: string | Point | undefined;
  count: number[];
};

export type Is = {
  type: "is";
  id: string;
  compare: Point | undefined;
  with: string | Point | undefined;
};

export type Ratio = { part: number; whole: number };

export type Chance = {
  type: "chance";
  id: string;
  ratio: Ratio;
};

export type Strategy = "all" | "any" | "one" | "none";

export type Point = {
  x: number;
  y: number;
};
