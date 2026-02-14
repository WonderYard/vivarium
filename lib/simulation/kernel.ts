import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { Accept, Opcode } from "@/common/constants";

/**
 * Returns 1 if the element id matches the check id, 0 otherwise.
 */
export const matchesElement = (elementId: number, checkId: number) => {
  "use gpu";
  return std.select(d.u32(0), d.u32(1), elementId === checkId);
};

/**
 * Returns 1 if the element id is contained in the bit-packed IDs, 0 otherwise.
 * Used for kind-based checks where multiple element IDs are packed as bit flags.
 */
export const matchesPackedIds = (elementId: number, packedIds: number) => {
  "use gpu";
  return std.select(
    d.u32(0),
    d.u32(1),
    (packedIds & (d.u32(1) << elementId)) !== d.u32(0)
  );
};

/**
 * Returns 1 if the neighbor count is one of the counts encoded in packedCount, 0 otherwise.
 * packedCount is a 9-bit array of flags where bit N means "count N is accepted".
 * Example: count = [2, 3] -> packedCount = 0b000001100
 */
export const matchesPackedCount = (count: number, packedCount: number) => {
  "use gpu";
  return std.select(
    d.u32(0),
    d.u32(1),
    (packedCount & (d.u32(1) << count)) !== d.u32(0)
  );
};

/**
 * Given 8 neighbor IDs, count how many match checkId and compare against packedCount.
 */
export const checkIdCount = (
  n0: number,
  n1: number,
  n2: number,
  n3: number,
  n4: number,
  n5: number,
  n6: number,
  n7: number,
  checkId: number,
  packedCount: number
) => {
  "use gpu";
  const countMask =
    matchesElement(n0, checkId) +
    matchesElement(n1, checkId) +
    matchesElement(n2, checkId) +
    matchesElement(n3, checkId) +
    matchesElement(n4, checkId) +
    matchesElement(n5, checkId) +
    matchesElement(n6, checkId) +
    matchesElement(n7, checkId);
  return matchesPackedCount(countMask, packedCount);
};

/**
 * Given 8 neighbor IDs, count how many are in the bit-packed kind set
 * and compare against packedCount.
 */
export const checkIdsCount = (
  n0: number,
  n1: number,
  n2: number,
  n3: number,
  n4: number,
  n5: number,
  n6: number,
  n7: number,
  packedIds: number,
  packedCount: number
) => {
  "use gpu";
  const countMask =
    matchesPackedIds(n0, packedIds) +
    matchesPackedIds(n1, packedIds) +
    matchesPackedIds(n2, packedIds) +
    matchesPackedIds(n3, packedIds) +
    matchesPackedIds(n4, packedIds) +
    matchesPackedIds(n5, packedIds) +
    matchesPackedIds(n6, packedIds) +
    matchesPackedIds(n7, packedIds);
  return matchesPackedCount(countMask, packedCount);
};

/**
 * Evaluate a single condition given the opcode, the 8 neighbor IDs,
 * the compare/with IDs (already resolved by the caller), and the condition data.
 */
export const evaluateCondition = (
  opcode: number,
  n0: number,
  n1: number,
  n2: number,
  n3: number,
  n4: number,
  n5: number,
  n6: number,
  n7: number,
  checkId: number,
  countOrWithId: number,
  compareId: number,
  withId: number
) => {
  "use gpu";

  if (opcode === Opcode.COUNT_ELEMENT) {
    return checkIdCount(n0, n1, n2, n3, n4, n5, n6, n7, checkId, countOrWithId);
  }

  if (opcode === Opcode.COUNT_POINT) {
    // compareId is the id at the check point, already resolved by the caller
    return checkIdCount(
      n0, n1, n2, n3, n4, n5, n6, n7,
      compareId,
      countOrWithId
    );
  }

  if (opcode === Opcode.COUNT_KIND) {
    return checkIdsCount(
      n0, n1, n2, n3, n4, n5, n6, n7,
      checkId,
      countOrWithId
    );
  }

  if (opcode === Opcode.IS_ELEMENT) {
    return matchesElement(compareId, countOrWithId);
  }

  if (opcode === Opcode.IS_POINT) {
    return matchesElement(compareId, withId);
  }

  if (opcode === Opcode.IS_KIND) {
    return matchesPackedIds(compareId, countOrWithId);
  }

  return d.u32(0);
};

/**
 * Check whether the accept strategy is satisfied given the number of passing
 * conditions and the total number of conditions.
 */
export const accepted = (
  accept: number,
  passing: number,
  conditionsCount: number
) => {
  "use gpu";
  return (
    (accept === Accept.ALL && passing === conditionsCount) ||
    (accept === Accept.ANY && passing >= d.u32(1)) ||
    (accept === Accept.ONE && passing === d.u32(1)) ||
    (accept === Accept.NONE && passing === d.u32(0))
  );
};
