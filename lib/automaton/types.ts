import type { Accept } from "@/common/constants";

export type Neighborhood = "square" | "cross" | "hexagonal";

export type RefType = "element" | "kind";

export type Kind = {
  type: "kind";
  id: string;
  name: string;
};

export type Color = string;

export type Automaton = {
  neighborhood: Neighborhood;
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
