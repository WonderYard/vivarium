import * as d from "typegpu/data";
import type {
  Automaton,
  Condition,
  Count,
  Is,
  Point,
  Rule,
} from "@/automaton/types";
import {
  GpuCondition,
  GpuElement,
  GpuRule,
  Opcode,
  To,
} from "@/common/constants";
import { colorToABGR } from "@/common/utils";

type GpuRuleType = d.Infer<typeof GpuRule>;
type GpuElementType = d.Infer<typeof GpuElement>;
type GpuConditionType = d.Infer<typeof GpuCondition>;

const assertMaxSupportedId = (elementIds: number[]) => {
  if (elementIds.some((index) => index > 31)) {
    console.warn(
      `Elements with an index greater than 31 and extending one or more kinds are not allowed when they are involved in conditions referencing kinds.
To assign a lower index, move the elements with extensions higher in the definition list or reduce the overall number of elements and kinds.`
    );
    throw new Error(
      "Element in condition involving kinds has an index greater than 31"
    );
  }
};

const compileCheckId = (
  checkId: string,
  gpuCondition: GpuConditionType,
  automaton: Automaton
) => {
  const { elements } = automaton;

  const elementId = elements.findIndex((element) => element.id === checkId);

  if (elementId !== -1) {
    // check element id
    gpuCondition.opcode = Opcode.COUNT_ELEMENT;
    gpuCondition.checkId = elementId;
  } else {
    // check kind id
    const elementIds = elements
      .map((element, index) => ({ ...element, index }))
      .filter((element) =>
        element.extensions.some((extension) => extension === checkId)
      )
      .map((element) => element.index);

    assertMaxSupportedId(elementIds);

    const packedIds = elementIds.reduce((acc, val) => {
      acc |= 1 << val;
      return acc;
    }, 0);

    gpuCondition.checkId = packedIds;
    gpuCondition.opcode = Opcode.COUNT_KIND;
  }
};

const compileCheckPoint = (
  checkPoint: Point,
  gpuCondition: GpuConditionType
) => {
  gpuCondition.opcode = Opcode.COUNT_POINT;
  gpuCondition.checkPointOrComparePoint = d.vec2u(checkPoint.x, checkPoint.y);
};

const compileCount = (
  condition: Count,
  gpuCondition: GpuConditionType,
  automaton: Automaton
) => {
  const { check, count } = condition;

  if (typeof check === "undefined") {
    throw new Error("condition.check is undefined");
  }

  // check
  if (typeof check === "string") {
    compileCheckId(check, gpuCondition, automaton);
  } else {
    compileCheckPoint(check, gpuCondition);
  }

  // count
  gpuCondition.countOrWithId = count.reduce(
    // Packing numbers as positional bits.
    // Example: [2, 3] -> 0b000001100
    (acc, val) => {
      acc |= 1 << val;
      return acc;
    },
    0
  );
};

const compileIs = (
  condition: Is,
  gpuCondition: GpuConditionType,
  automaton: Automaton
) => {
  const comparePoint = condition.compare;

  if (typeof comparePoint === "undefined") {
    throw new Error("condition.compare is undefined");
  }

  if (typeof condition.with === "undefined") {
    throw new Error("condition.with is undefined");
  }

  gpuCondition.checkPointOrComparePoint = d.vec2u(
    comparePoint.x,
    comparePoint.y
  );

  if (typeof condition.with === "string") {
    const _with = condition.with;
    if (automaton.elements.some((element) => element.id === _with)) {
      gpuCondition.opcode = Opcode.IS_ELEMENT;

      const withId = automaton.elements.findIndex(
        (element) => element.id === _with
      );

      gpuCondition.countOrWithId = withId;
    } else {
      const elementIds = automaton.elements
        .map((element, index) => ({ ...element, index }))
        .filter((element) =>
          element.extensions.some((extension) => extension === _with)
        )
        .map((element) => element.index);

      assertMaxSupportedId(elementIds);

      const packedIds = elementIds.reduce((acc, val) => {
        acc |= 1 << val;
        return acc;
      }, 0);

      gpuCondition.countOrWithId = packedIds;
      gpuCondition.opcode = Opcode.IS_KIND;
    }
  } else {
    gpuCondition.opcode = Opcode.IS_POINT;
    gpuCondition.withPoint = d.vec2u(condition.with.x, condition.with.y);
  }
};

export type GpuAutomaton = {
  gpuNeighborhood: number;
  gpuElements: GpuElementType[];
  gpuRules: GpuRuleType[];
  gpuConditions: GpuConditionType[];
};

export const compileGpuAutomaton = (automaton: Automaton): GpuAutomaton => {
  const { elements, rules: _rules } = automaton;

  // First we group element rules by element index
  const elementRulesByElementIndex = _rules
    .filter((rule) => rule.fromType === "element")
    .reduce<Rule[][]>(
      (elementRulesByElementIndex, rule) => {
        const elementIndex = elements.findIndex(
          (element) => element.id === rule.fromId
        );
        elementRulesByElementIndex[elementIndex].push(rule);

        return elementRulesByElementIndex;
      },
      elements.map(() => [])
    );

  // Then, with lower priority because they come after, we loop kind rules
  // and we assign each rule to all the elements that extend that kind.
  const rulesByElementIndex = _rules
    .filter((rule) => rule.fromType === "kind")
    .reduce<Rule[][]>((rulesByElementIndex, rule) => {
      elements.forEach((element, index) => {
        if (element.extensions.includes(rule.fromId)) {
          rulesByElementIndex[index].push(rule);
        }
      });

      return rulesByElementIndex;
    }, elementRulesByElementIndex);

  const rules: (Rule & { conditionsStart: number; conditionsEnd: number })[] =
    [];
  let conditions: Condition[] = [];
  const conditionsMap = new Map<string, [number, number]>();

  rulesByElementIndex.forEach((elementRules) => {
    elementRules.forEach((rule, index) => {
      let conditionsStart: number;
      let conditionsEnd: number;

      const conditionsKey = `${rule.fromId}:${index.toString()}`;

      if (rule.fromType === "kind" && conditionsMap.has(conditionsKey)) {
        [conditionsStart, conditionsEnd] = conditionsMap.get(conditionsKey) as [
          number,
          number,
        ];
      } else {
        conditionsStart = conditions.length;

        if (Array.isArray(rule.when)) {
          conditions = conditions.concat(rule.when);
        } else {
          conditions = conditions.concat([rule.when]);
        }

        conditionsEnd = conditions.length;

        conditionsMap.set(conditionsKey, [conditionsStart, conditionsEnd]);
      }

      rules.push({
        ...rule,
        conditionsStart,
        conditionsEnd,
      });
    });
  });

  const gpuElements: GpuElementType[] = [];
  const gpuRules: GpuRuleType[] = [];
  const gpuConditions: GpuConditionType[] = [];

  let ruleIndex = 0;

  elements.forEach((element, index) => {
    const ruleStart = ruleIndex;
    // Start is inclusive, end is exclusive, so:
    // [n, n] means element has no rule,
    // [n, n + 1] means element has 1 rule.
    // In the first case, next rule would start at n.
    // in the second case, at n + 1.
    const ruleEnd = ruleIndex + rulesByElementIndex[index].length;
    ruleIndex = ruleEnd;

    const gpuElement = GpuElement({
      color: colorToABGR(element.color),
      ruleStart,
      ruleEnd,
    });

    gpuElements.push(gpuElement);
  });

  rules.forEach((rule) => {
    const { conditionsStart, conditionsEnd } = rule;

    const gpuRule = GpuRule({
      toType: To.ELEMENT_ID,
      toId: 0,
      toNeighbor: d.vec2u(),
      conditionsStart,
      conditionsEnd,
      accept: rule.accept,
    });

    gpuRules.push(gpuRule);

    if (typeof rule.to === "undefined") {
      throw new Error("rule.to is undefined");
    }

    if (typeof rule.to === "string") {
      gpuRule.toType = To.ELEMENT_ID;
      gpuRule.toId = elements.findIndex((element) => element.id === rule.to);
    } else {
      // TODO: is there a need to optimize some cases where point is (0, 0)?
      gpuRule.toType = To.POINT;
      gpuRule.toNeighbor = d.vec2u(rule.to.x, rule.to.y);
    }
  });

  conditions.forEach((condition) => {
    const gpuCondition = GpuCondition({
      opcode: Opcode.NOOP,
      checkId: 0,
      countOrWithId: 0,
      checkPointOrComparePoint: d.vec2u(),
      withPoint: d.vec2u(),
      chance: 0,
    });

    gpuConditions.push(gpuCondition);

    if (condition.type === "count") {
      compileCount(condition, gpuCondition, automaton);
    } else if (condition.type === "is") {
      compileIs(condition, gpuCondition, automaton);
    } else {
      gpuCondition.opcode = Opcode.CHANCE;
      gpuCondition.chance = condition.ratio.part / condition.ratio.whole;
    }
  });

  return {
    gpuNeighborhood: automaton.neighborhood === "cross" ? 4 : 8,
    gpuElements,
    gpuRules,
    gpuConditions,
  };
};
