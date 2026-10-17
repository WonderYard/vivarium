import type { Automaton } from "@/automaton/types";
import { acceptMap } from "@/common/constants";
import { colorToABGR, parseColor } from "@/common/utils";
import { assertReferences } from "@/webgpu/compiler";

// The automaton travels in an iTXt chunk (UTF-8, unlike tEXt), so names can use any script.
const KEYWORD = "vivarium";

// IHDR is always the first chunk: 8 (signature) + 4 (length) + 4 (type) + 13 (data) + 4 (crc).
const IHDR_END = 33;

const utf8 = (text: string) => new TextEncoder().encode(text);

// Type, keyword, then null separator, compression flag and method (none),
// empty language tag and empty translated keyword: five zero bytes.
const CHUNK_PREFIX = new Uint8Array([...utf8("iTXt"), ...utf8(KEYWORD), 0, 0, 0, 0, 0]);

const CRC_TABLE = Array.from({ length: 256 }, (_, n) => {
  let c = n;
  for (let k = 0; k < 8; k++) {
    c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  }
  return c >>> 0;
});

const crc32 = (bytes: Uint8Array): number => {
  let c = 0xffffffff;
  for (const byte of bytes) {
    c = CRC_TABLE[(c ^ byte) & 0xff] ^ (c >>> 8);
  }
  return (c ^ 0xffffffff) >>> 0;
};

const textChunk = (text: string): Uint8Array<ArrayBuffer> => {
  const content = utf8(text);
  const chunk = new Uint8Array(4 + CHUNK_PREFIX.length + content.length + 4);
  const view = new DataView(chunk.buffer);

  chunk.set(CHUNK_PREFIX, 4);
  chunk.set(content, 4 + CHUNK_PREFIX.length);
  // The length counts only the data, the crc covers type and data.
  view.setUint32(0, chunk.length - 12);
  view.setUint32(chunk.length - 4, crc32(chunk.subarray(4, chunk.length - 4)));

  return chunk;
};

const readText = (png: Uint8Array): string | undefined => {
  const view = new DataView(png.buffer, png.byteOffset, png.byteLength);

  for (let offset = 8; offset + 8 <= png.length; offset += view.getUint32(offset) + 12) {
    const body = png.subarray(offset + 4, offset + 8 + view.getUint32(offset));

    if (CHUNK_PREFIX.every((byte, i) => body[i] === byte)) {
      return new TextDecoder().decode(body.subarray(CHUNK_PREFIX.length));
    }
  }

  return undefined;
};

type Json = Record<string, unknown>;

const MALFORMED = "This PNG has a malformed vivarium automaton in it";

function check(condition: unknown, path: string, expected: string): asserts condition {
  if (!condition) {
    throw new Error(`${MALFORMED}: ${path} is not ${expected}.`);
  }
}

const isObject = (value: unknown): value is Json =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const isString = (value: unknown): value is string => typeof value === "string";

const isPoint = (value: unknown): boolean =>
  isObject(value) && Number.isInteger(value.x) && Number.isInteger(value.y);

const isRef = (value: unknown): boolean => isString(value) || isPoint(value);

// Checks that value is an array of objects and returns them with their paths.
const objects = (value: unknown, path: string): [Json, string][] => {
  check(Array.isArray(value), path, "an array");

  return value.map((item: unknown, i) => {
    check(isObject(item), `${path}[${i}]`, "an object");
    return [item, `${path}[${i}]`];
  });
};

// The file is untrusted input: check the whole shape before anything reads it.
const parseAutomaton = (text: string): Automaton => {
  let automaton: unknown;

  try {
    automaton = JSON.parse(text);
  } catch {
    throw new Error(`${MALFORMED}: it is not valid JSON.`);
  }

  check(isObject(automaton), "the automaton", "an object");
  check(
    automaton.neighborhood === "square" || automaton.neighborhood === "cross",
    "neighborhood",
    '"square" or "cross"',
  );
  check(typeof automaton.wrapping === "boolean", "wrapping", "a boolean");

  for (const [element, path] of objects(automaton.elements, "elements")) {
    check(element.type === "element", `${path}.type`, '"element"');
    check(isString(element.id), `${path}.id`, "a string");
    check(isString(element.name), `${path}.name`, "a string");
    check(
      isString(element.color) && parseColor(element.color) !== undefined,
      `${path}.color`,
      "a valid color",
    );
    check(
      Array.isArray(element.extensions) && element.extensions.every(isString),
      `${path}.extensions`,
      "an array of strings",
    );
  }

  for (const [kind, path] of objects(automaton.kinds, "kinds")) {
    check(kind.type === "kind", `${path}.type`, '"kind"');
    check(isString(kind.id), `${path}.id`, "a string");
    check(isString(kind.name), `${path}.name`, "a string");
  }

  for (const [rule, path] of objects(automaton.rules, "rules")) {
    check(
      rule.fromType === "element" || rule.fromType === "kind",
      `${path}.fromType`,
      '"element" or "kind"',
    );
    check(isString(rule.fromId), `${path}.fromId`, "a string");
    check(isRef(rule.to), `${path}.to`, "an id or a point");
    check(
      Object.values<unknown>(acceptMap).includes(rule.accept),
      `${path}.accept`,
      "a valid accept strategy",
    );

    for (const [condition, conditionPath] of objects(rule.when, `${path}.when`)) {
      check(isString(condition.id), `${conditionPath}.id`, "a string");

      if (condition.type === "count") {
        check(isRef(condition.check), `${conditionPath}.check`, "an id or a point");
        check(
          Array.isArray(condition.count) && condition.count.every(Number.isFinite),
          `${conditionPath}.count`,
          "an array of numbers",
        );
      } else if (condition.type === "is") {
        check(isPoint(condition.compare), `${conditionPath}.compare`, "a point");
        check(isRef(condition.with), `${conditionPath}.with`, "an id or a point");
      } else {
        check(condition.type === "chance", `${conditionPath}.type`, '"count", "is" or "chance"');

        const { ratio } = condition;
        check(
          isObject(ratio) &&
            typeof ratio.part === "number" &&
            typeof ratio.whole === "number" &&
            Number.isFinite(ratio.whole) &&
            ratio.whole > 0 &&
            ratio.part >= 0 &&
            ratio.part <= ratio.whole,
          `${conditionPath}.ratio`,
          "a ratio with 0 <= part <= whole and whole > 0",
        );
      }
    }
  }

  try {
    assertReferences(automaton as Automaton);
  } catch (error) {
    throw new Error(`${MALFORMED}: ${(error as Error).message}`, { cause: error });
  }

  return automaton as Automaton;
};

// ABGR, as unsigned so it compares equal to the pixels read through a Uint32Array.
const elementColors = (automaton: Automaton) =>
  automaton.elements.map((element) => colorToABGR(element.color) >>> 0);

/**
 * Encodes an automaton and a grid as a PNG image: each cell becomes one pixel with its element's color,
 * and the automaton is stored inside the file, so {@link fromPng} can bring both back.
 * The image stays a regular PNG that any viewer can open.
 *
 * @param options.automaton - The automaton whose element colors paint the grid. Its elements must have distinct colors.
 * @param options.grid - A flat array of element indices (row-major order), like the one returned by `readGrid`.
 * @param options.width - The grid width in cells.
 * @param options.height - The grid height in cells.
 * @returns A promise of the PNG as a `Blob`.
 */
export const toPng = async ({
  automaton,
  grid,
  width,
  height,
}: {
  automaton: Automaton;
  grid: number[];
  width: number;
  height: number;
}): Promise<Blob> => {
  if (grid.length !== width * height) {
    throw new Error(
      `Grid length ${grid.length} does not match the expected length of ${width * height} (width ${width} x height ${height}).`,
    );
  }

  const colors = elementColors(automaton);

  // Two elements with the same color would be indistinguishable when reading the pixels back.
  if (new Set(colors).size !== colors.length) {
    throw new Error("Elements must have distinct colors to be saved as a PNG.");
  }

  const image = new ImageData(width, height);
  // ImageData bytes are RGBA, which a little-endian Uint32Array reads as ABGR.
  const pixels = new Uint32Array(image.data.buffer);

  grid.forEach((elementIndex, i) => {
    const color = colors[elementIndex];

    if (color === undefined) {
      throw new Error(
        `Element index ${elementIndex} at position ${i} is invalid. Expected an integer in range [0, ${colors.length - 1}].`,
      );
    }

    pixels[i] = color;
  });

  const canvas = new OffscreenCanvas(width, height);
  canvas.getContext("2d")!.putImageData(image, 0, 0);
  const png = new Uint8Array(await (await canvas.convertToBlob()).arrayBuffer());

  return new Blob(
    [png.subarray(0, IHDR_END), textChunk(JSON.stringify(automaton)), png.subarray(IHDR_END)],
    { type: "image/png" },
  );
};

/**
 * Decodes a PNG made by {@link toPng} back into its automaton and grid.
 * The grid is read from the pixels, so the image can be edited with any image editor
 * as long as it only uses the elements' colors and keeps the file's metadata.
 *
 * @param png - The PNG file, e.g. from an `<input type="file">`, a drop event or `fetch(...).then((r) => r.blob())`.
 * @returns A promise of the `automaton`, the flat `grid` of element indices (row-major order), and its `width` and `height`.
 * Pass them all to {@link setup}, with `grid` as `initialGrid`; the canvas can be any size, the grid is stretched over it.
 * To load them into a running simulation of the same size, call `setAutomaton` before `writeGrid`.
 */
export const fromPng = async (
  png: Blob,
): Promise<{ automaton: Automaton; grid: number[]; width: number; height: number }> => {
  const text = readText(new Uint8Array(await png.arrayBuffer()));

  if (text === undefined) {
    throw new Error("This PNG has no vivarium automaton in it.");
  }

  const automaton = parseAutomaton(text);

  // Skip color management, which would shift the colors away from the elements' ones.
  const bitmap = await createImageBitmap(png, {
    colorSpaceConversion: "none",
    premultiplyAlpha: "none",
  });
  const { width, height } = bitmap;
  const context = new OffscreenCanvas(width, height).getContext("2d")!;
  context.drawImage(bitmap, 0, 0);
  bitmap.close();

  const pixels = new Uint32Array(context.getImageData(0, 0, width, height).data.buffer);
  const indices = new Map(elementColors(automaton).map((color, index) => [color, index]));

  const grid = Array.from(pixels, (pixel, i) => {
    const index = indices.get(pixel);

    if (index === undefined) {
      const hex = [pixel, pixel >>> 8, pixel >>> 16, pixel >>> 24]
        .map((byte) => (byte & 0xff).toString(16).padStart(2, "0"))
        .join("");
      throw new Error(
        `Pixel at (${i % width}, ${Math.floor(i / width)}) has color #${hex}, which matches no element.`,
      );
    }

    return index;
  });

  return { automaton, grid, width, height };
};
