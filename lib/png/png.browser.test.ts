import { fromPng, toPng } from "@/png/png";
import { vivarium } from "@/vivarium/vivarium";
import { setup } from "@/webgpu/setup";
import { expect, test } from "vitest";

const createAutomaton = () => {
  const v = vivarium("cross", { wrapping: true });
  const dead = v.element("死んだ", "black");
  const alive = v.element("alive", "#f00");
  dead.to(alive).is(v.neighbor.TOP, alive);
  return v.create();
};

const width = 8;
const height = 8;
const grid = Array.from({ length: width * height }, (_, i) => ((i * 7) % 3 === 0 ? 1 : 0));

test("round-trips the automaton and the grid", async () => {
  const automaton = createAutomaton();
  const png = await toPng({ automaton, grid, width, height });

  expect(png.type).toBe("image/png");
  await expect(fromPng(png)).resolves.toEqual({ automaton, grid, width, height });
});

test("stays a regular image showing the grid", async () => {
  const png = await toPng({ automaton: createAutomaton(), grid, width, height });
  const bitmap = await createImageBitmap(png);
  const context = new OffscreenCanvas(width, height).getContext("2d")!;
  context.drawImage(bitmap, 0, 0);

  expect([...context.getImageData(0, 0, 1, 1).data]).toEqual([255, 0, 0, 255]);
  expect([...context.getImageData(1, 0, 1, 1).data]).toEqual([0, 0, 0, 255]);
});

test("rejects a PNG without an automaton", async () => {
  const canvas = new OffscreenCanvas(width, height);
  canvas.getContext("2d")!.fillRect(0, 0, width, height);

  await expect(fromPng(await canvas.convertToBlob())).rejects.toThrow("no vivarium automaton");
});

test("rejects pixels painted with a color that matches no element", async () => {
  const automaton = createAutomaton();
  const red = new Uint8Array(await (await toPng({ automaton, grid, width, height })).arrayBuffer());
  automaton.elements[1].color = "blue";
  const blue = new Uint8Array(
    await (await toPng({ automaton, grid, width, height })).arrayBuffer(),
  );

  // The automaton chunk sits right after IHDR (33 bytes): graft the red automaton onto blue pixels.
  const chunkOf = (png: Uint8Array<ArrayBuffer>) =>
    png.subarray(33, 33 + new DataView(png.buffer).getUint32(33) + 12);
  const tampered = new Blob([
    blue.subarray(0, 33),
    chunkOf(red),
    blue.subarray(33 + chunkOf(blue).length),
  ]);

  await expect(fromPng(tampered)).rejects.toThrow("Pixel at (0, 0) has color #0000ffff");
});

test("rejects elements sharing a color", async () => {
  const automaton = createAutomaton();
  automaton.elements[1].color = "#000000";

  await expect(toPng({ automaton, grid, width, height })).rejects.toThrow("distinct colors");
});

test("saves a running simulation", async () => {
  const automaton = createAutomaton();
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const simulation = await setup({ canvas, automaton, initialGrid: grid });

  simulation.update();
  const expected = await simulation.readGrid();
  const loaded = await fromPng(await simulation.toPng());
  simulation.destroy();

  expect(loaded).toEqual({ automaton, grid: expected, width, height });
});
