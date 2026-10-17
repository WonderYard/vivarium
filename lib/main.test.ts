import { expect, test } from "vitest";

test("importing the library doesn't require WebGPU", async () => {
  expect(globalThis.navigator?.gpu).toBeUndefined();
  await expect(import("@/main")).resolves.toHaveProperty("setup");
});
