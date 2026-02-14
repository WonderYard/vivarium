import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";

/**
 * Returns 1 if the element id matches the check id, 0 otherwise.
 * Used by both GPU (via "use gpu" functions) and CPU (via simulation).
 */
export const matchesElement = tgpu
  .fn([d.u32, d.u32], d.u32)((elementId, checkId) => {
    return std.select(d.u32(0), d.u32(1), elementId === checkId);
  })
  .$name("matchesElement");

/**
 * Returns 1 if the element id is contained in the bit-packed IDs, 0 otherwise.
 * Used for kind-based checks where multiple element IDs are packed as bit flags.
 */
export const matchesPackedIds = tgpu
  .fn([d.u32, d.u32], d.u32)((elementId, packedIds) => {
    return std.select(
      d.u32(0),
      d.u32(1),
      (packedIds & (d.u32(1) << elementId)) !== d.u32(0)
    );
  })
  .$name("matchesPackedIds");

/**
 * Returns 1 if the neighbor count is one of the counts encoded in packedCount, 0 otherwise.
 * packedCount is a 9-bit array of flags where bit N means "count N is accepted".
 * Example: count = [2, 3] -> packedCount = 0b000001100
 */
export const matchesPackedCount = tgpu
  .fn([d.u32, d.u32], d.u32)((count, packedCount) => {
    return std.select(
      d.u32(0),
      d.u32(1),
      (packedCount & (d.u32(1) << count)) !== d.u32(0)
    );
  })
  .$name("matchesPackedCount");
