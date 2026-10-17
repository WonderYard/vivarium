import type { Automaton } from "@/automaton/types";
import {
  Accept,
  GpuCondition,
  GpuElement,
  GpuRule,
  neighborhoodPoints,
  Opcode,
  To,
} from "@/common/constants";
import { compileGpuAutomaton } from "@/webgpu/compiler";
import * as d from "typegpu/data";

type Layout = {
  stride: number;
  fields: { name: string; type: string; offset: number; size: number }[];
};

const enumNames = (e: object): string[] =>
  Object.keys(e).filter((key) => Number.isNaN(Number(key)));

// Offsets come from TypeGPU so the byte view follows the structs if they change.
const layoutOf = (schema: d.AnyWgslStruct): Layout => ({
  stride: d.sizeOf(schema),
  fields: Object.entries(schema.propTypes).map(([name, type]) => ({
    name,
    type: type.type,
    offset: d.memoryLayoutOf(schema, (s) => (s as Record<string, unknown>)[name]).offset,
    size: d.sizeOf(type as d.AnyWgslData),
  })),
});

// Every field is built from 4-byte scalars (u32, f32, vec2u), so this covers all of them.
const toBytes = (layout: Layout, value: Record<string, unknown>): number[] => {
  const view = new DataView(new ArrayBuffer(layout.stride));
  layout.fields.forEach(({ name, type, offset }) => {
    const v = value[name];
    const parts = typeof v === "number" ? [v] : Array.from(v as ArrayLike<number>);
    parts.forEach((part, i) => {
      if (type.includes("f")) {
        view.setFloat32(offset + i * 4, part, true);
      } else {
        view.setUint32(offset + i * 4, part >>> 0, true);
      }
    });
  });
  return Array.from(new Uint8Array(view.buffer));
};

const buildData = (automaton: Automaton, title: string) => {
  const gpu = compileGpuAutomaton(automaton);
  const { gpuElements, gpuRules, gpuConditions, gpuKindWords, gpuKindBits, gpuRuleSources } = gpu;

  const layouts = {
    elements: layoutOf(GpuElement),
    rules: layoutOf(GpuRule),
    conditions: layoutOf(GpuCondition),
    kindBits: { stride: 4, fields: [{ name: "word", type: "u32", offset: 0, size: 4 }] },
  };
  const buffer = (layout: Layout, rows: Record<string, unknown>[]) => ({
    ...layout,
    rows: rows.map((row) => toBytes(layout, row)),
  });
  // setup allocates at least one entry per array, so empty arrays still take one stride.
  const arrayBytes = (layout: Layout, count: number) => layout.stride * Math.max(count, 1);

  return {
    title,
    neighborhood: automaton.neighborhood,
    wrapping: automaton.wrapping,
    sourceRuleCount: automaton.rules.length,
    opcodes: enumNames(Opcode),
    accepts: enumNames(Accept),
    tos: enumNames(To),
    neighbors: Object.entries(neighborhoodPoints).map(
      ([name, p]) => [`${p.x},${p.y}`, name] as [string, string],
    ),
    kinds: automaton.kinds.map((kind) => kind.name),
    kindWords: gpuKindWords,
    kindBits: gpuKindBits,
    elements: gpuElements.map((e, i) => ({
      name: automaton.elements[i].name,
      color: e.color,
      ruleStart: e.ruleStart,
      ruleEnd: e.ruleEnd,
    })),
    rules: gpuRules.map((r, i) => {
      const source = gpuRuleSources[i];
      const rule = automaton.rules[source];
      return {
        toType: r.toType,
        toId: r.toId,
        toNeighbor: Array.from(r.toNeighbor),
        accept: r.accept,
        conditionsStart: r.conditionsStart,
        conditionsEnd: r.conditionsEnd,
        source,
        via:
          rule.fromType === "kind"
            ? automaton.kinds.find((kind) => kind.id === rule.fromId)?.name
            : undefined,
      };
    }),
    conditions: gpuConditions.map((c) => ({
      opcode: c.opcode,
      checkId: c.checkId,
      countOrWithId: c.countOrWithId,
      checkPoint: Array.from(c.checkPointOrComparePoint),
      withPoint: Array.from(c.withPoint),
      chance: c.chance,
    })),
    bindings: [
      ["neighborhood", "uniform u32", 4, `${gpu.gpuNeighborhood} (${automaton.neighborhood})`],
      [
        "elements",
        `array<GpuElement, ${gpuElements.length}>`,
        arrayBytes(layouts.elements, gpuElements.length),
        "",
      ],
      [
        "rules",
        `array<GpuRule, ${gpuRules.length}>`,
        arrayBytes(layouts.rules, gpuRules.length),
        "",
      ],
      [
        "conditions",
        `array<GpuCondition, ${gpuConditions.length}>`,
        arrayBytes(layouts.conditions, gpuConditions.length),
        "",
      ],
      ["kindWords", "uniform u32", 4, String(gpuKindWords)],
      [
        "kindBits",
        `array<u32, ${gpuKindBits.length}>`,
        arrayBytes(layouts.kindBits, gpuKindBits.length),
        "",
      ],
    ] as [string, string, number, string][],
    buffers: {
      elements: buffer(layouts.elements, gpuElements),
      rules: buffer(layouts.rules, gpuRules),
      conditions: buffer(layouts.conditions, gpuConditions),
      kindBits: buffer(
        layouts.kindBits,
        gpuKindBits.map((word) => ({ word })),
      ),
    },
  };
};

type InspectData = ReturnType<typeof buildData>;

// Runs in the generated page through toString(), so it must not reference anything outside its body.
const render = (data: InspectData): void => {
  const $ = (id: string) => document.getElementById(id) as HTMLElement;
  const esc = (s: string) => s.replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
  const h = (tag: string, cls: string, html: string) => {
    const n = document.createElement(tag);
    n.className = cls;
    n.innerHTML = html;
    return n;
  };
  const hex32 = (n: number) => `0x${(n >>> 0).toString(16).toUpperCase().padStart(8, "0")}`;
  const fmtBytes = (n: number) => (n >= 1024 ? `${(n / 1024).toLocaleString()} KiB` : `${n} B`);
  const neighbors = new Map(data.neighbors);
  const pointName = ([x, y]: number[]) =>
    neighbors.get(`${x | 0},${y | 0}`) ?? `(${x | 0}, ${y | 0})`;
  const elName = (i: number) => esc(data.elements[i]?.name ?? `#${i}?`);
  const kindName = (i: number) => esc(data.kinds[i] ?? `#${i}?`);
  const cssColor = (abgr: number) =>
    `#${[abgr & 255, (abgr >>> 8) & 255, (abgr >>> 16) & 255].map((v) => v.toString(16).padStart(2, "0")).join("")}`;
  const hasKind = (element: number, kind: number) =>
    (data.kindBits[element * data.kindWords + (kind >>> 5)] >>> (kind & 31)) & 1;
  const maxCount = data.neighborhood === "cross" ? 4 : 8;
  const bitsHtml = (packed: number) =>
    `<span class="bits">${Array.from({ length: maxCount + 1 }, (_, i) => `<i class="${(packed >>> i) & 1 ? "on" : ""}">${i}</i>`).join("")}</span>`;
  const percent = (chance: number) => `${+(chance * 100).toPrecision(3)}%`;

  const decodeCondition = (c: InspectData["conditions"][number]) => {
    switch (data.opcodes[c.opcode]) {
      case "COUNT_ELEMENT":
        return `count <b>${elName(c.checkId)}</b> in ${bitsHtml(c.countOrWithId)}`;
      case "COUNT_KIND":
        return `count kind <b>${kindName(c.checkId)}</b> in ${bitsHtml(c.countOrWithId)}`;
      case "COUNT_POINT":
        return `count same as <b>${pointName(c.checkPoint)}</b> in ${bitsHtml(c.countOrWithId)}`;
      case "IS_ELEMENT":
        return `<b>${pointName(c.checkPoint)}</b> is <b>${elName(c.countOrWithId)}</b>`;
      case "IS_KIND":
        return `<b>${pointName(c.checkPoint)}</b> is kind <b>${kindName(c.countOrWithId)}</b>`;
      case "IS_POINT":
        return `<b>${pointName(c.checkPoint)}</b> equals <b>${pointName(c.withPoint)}</b>`;
      case "CHANCE":
        return `chance <b>${percent(c.chance)}</b>`;
      default:
        return "noop";
    }
  };

  // ---- header ----
  const meta: [string, string | number][] = [
    ["neighborhood", data.neighborhood],
    ["wrapping", String(data.wrapping)],
    ["elements", data.elements.length],
    ["kinds", data.kinds.length],
    ["rules", `${data.rules.length} (${data.sourceRuleCount} in source)`],
    ["conditions", data.conditions.length],
  ];
  meta.forEach(([k, v]) => $("meta").append(h("span", "chip", `<b>${k}</b> ${v}`)));

  // ---- bind group ----
  const total = data.bindings.reduce((sum, b) => sum + b[2], 0);
  $("group-size").textContent = fmtBytes(total);
  data.bindings.forEach(([name, type, size, value], i) => {
    $("bar").append(h("span", "", ""));
    ($("bar").lastChild as HTMLElement).style.flex = String(Math.max(size, total * 0.012));
    $("bindings").append(
      h(
        "div",
        "binding",
        `<span class="name">@${i} ${name}${value ? ` = ${value}` : ""}</span><span class="type">${esc(type)}</span><span class="size">${fmtBytes(size)}</span>`,
      ),
    );
  });

  // ---- pointer map ----
  const relations: Record<string, Set<string>> = {};
  const link = (a: string, b: string) => {
    (relations[a] ??= new Set([a])).add(b);
    (relations[b] ??= new Set([b])).add(a);
  };
  data.elements.forEach((e, ei) => {
    for (let r = e.ruleStart; r < e.ruleEnd; r++) {
      link(`e${ei}`, `r${r}`);
      for (let c = data.rules[r].conditionsStart; c < data.rules[r].conditionsEnd; c++) {
        link(`e${ei}`, `c${c}`);
        link(`r${r}`, `c${c}`);
      }
    }
  });

  const row = (key: string, html: string) => {
    const n = h("div", "row", html);
    n.dataset.key = key;
    n.tabIndex = 0;
    return n;
  };
  const laneTitle = (lane: string, title: string, count: number, size: number) => {
    $(lane).append(
      h(
        "div",
        "lane-title",
        `<span class="eyebrow">${title}</span><span class="mono muted">${count} × ${size} B</span>`,
      ),
    );
    if (count === 0) $(lane).append(h("p", "sub", `No ${title}.`));
  };
  laneTitle("lane-e", "elements", data.elements.length, data.buffers.elements.stride);
  laneTitle("lane-r", "rules", data.rules.length, data.buffers.rules.stride);
  laneTitle("lane-c", "conditions", data.conditions.length, data.buffers.conditions.stride);

  data.elements.forEach((e, i) => {
    const kinds = data.kinds.filter((_, k) => hasKind(i, k)).map(esc);
    $("lane-e").append(
      row(
        `e${i}`,
        `<div class="row-top"><span class="idx">${i}</span><span class="swatch" style="background:${cssColor(e.color)}"></span><span class="row-name">${elName(i)}</span>${kinds.length ? `<span class="tag">${kinds.join(", ")}</span>` : ""}</div>
        <div class="row-fields"><span>color <b>${hex32(e.color)}</b></span><span>rules <b>[${e.ruleStart}, ${e.ruleEnd})</b></span>${e.ruleStart === e.ruleEnd ? `<span class="tag warn">no rules</span>` : ""}</div>`,
      ),
    );
  });

  data.rules.forEach((r, i) => {
    const owner = data.elements.findIndex((e) => i >= e.ruleStart && i < e.ruleEnd);
    const target =
      data.tos[r.toType] === "ELEMENT_ID"
        ? `→ <b>${elName(r.toId)}</b>`
        : `→ copy <b>${pointName(r.toNeighbor)}</b>`;
    const firstUse = data.rules.findIndex((x) => x.source === r.source);
    $("lane-r").append(
      row(
        `r${i}`,
        `<div class="row-top"><span class="idx">${i}</span><span class="row-name">${elName(owner)} ${target}</span></div>
        <div class="row-fields"><span>accept <b>${data.accepts[r.accept]}</b></span><span>conds <b>[${r.conditionsStart}, ${r.conditionsEnd})</b></span><span>source <b>#${r.source}</b></span></div>
        <div class="row-fields">${r.via ? `<span class="tag">inherited via ${esc(r.via)}</span>` : ""}${firstUse < i ? `<span class="tag warn">reuses conditions of rule ${firstUse}</span>` : ""}${r.conditionsStart === r.conditionsEnd ? `<span class="tag dim">unconditional</span>` : ""}</div>`,
      ),
    );
  });

  data.conditions.forEach((c, i) => {
    const users = data.rules.filter((r) => i >= r.conditionsStart && i < r.conditionsEnd).length;
    $("lane-c").append(
      row(
        `c${i}`,
        `<div class="row-top"><span class="idx">${i}</span><span class="op">${data.opcodes[c.opcode]}</span><span class="tag dim">opcode ${c.opcode}</span>${users > 1 ? `<span class="tag">shared by ${users} rules</span>` : ""}</div>
        <div class="row-fields decoded">${decodeCondition(c)}</div>`,
      ),
    );
  });

  const svg = $("links");
  const map = $("map");
  const band = (a: Element, b0: Element, b1: Element, dashed: boolean, key: string) => {
    const m = map.getBoundingClientRect();
    const ra = a.getBoundingClientRect();
    const r0 = b0.getBoundingClientRect();
    const r1 = b1.getBoundingClientRect();
    const x0 = ra.right - m.left;
    const x1 = r0.left - m.left;
    const mx = (x0 + x1) / 2;
    const ay0 = ra.top - m.top + 6;
    const ay1 = ra.bottom - m.top - 6;
    const by0 = r0.top - m.top + 2;
    const by1 = r1.bottom - m.top - 2;
    const p = document.createElementNS("http://www.w3.org/2000/svg", "path");
    p.setAttribute(
      "d",
      `M${x0},${ay0} C${mx},${ay0} ${mx},${by0} ${x1},${by0} L${x1},${by1} C${mx},${by1} ${mx},${ay1} ${x0},${ay1} Z`,
    );
    p.setAttribute("class", dashed ? "band reused" : "band");
    p.dataset.key = key;
    svg.append(p);
  };
  const drawLinks = () => {
    svg.innerHTML = "";
    const at = (key: string) => map.querySelector(`[data-key="${key}"]`) as Element;
    data.elements.forEach((e, i) => {
      if (e.ruleEnd > e.ruleStart)
        band(at(`e${i}`), at(`r${e.ruleStart}`), at(`r${e.ruleEnd - 1}`), false, `e${i}`);
    });
    data.rules.forEach((r, i) => {
      const reused = data.rules.findIndex((x) => x.source === r.source) < i;
      if (r.conditionsEnd > r.conditionsStart)
        band(
          at(`r${i}`),
          at(`c${r.conditionsStart}`),
          at(`c${r.conditionsEnd - 1}`),
          reused,
          `r${i}`,
        );
    });
  };
  const hover = (key: string | null) => {
    const hot = key ? (relations[key] ?? new Set([key])) : null;
    map.classList.toggle("hovering", !!hot);
    map
      .querySelectorAll<HTMLElement>(".row")
      .forEach((n) => n.classList.toggle("hot", !!hot && hot.has(n.dataset.key as string)));
    svg
      .querySelectorAll<SVGElement>("path")
      .forEach((p) => p.classList.toggle("hot", !!hot && hot.has(p.dataset.key as string)));
  };
  map.querySelectorAll<HTMLElement>(".row").forEach((n) => {
    n.addEventListener("mouseenter", () => hover(n.dataset.key as string));
    n.addEventListener("focus", () => hover(n.dataset.key as string));
    n.addEventListener("mouseleave", () => hover(null));
    n.addEventListener("blur", () => hover(null));
  });
  new ResizeObserver(drawLinks).observe(map);

  // ---- kind bits ----
  $("kinds").innerHTML =
    `<tr><th>element</th>${data.kinds.map((k, i) => `<th>bit ${i} · ${esc(k)}</th>`).join("")}<th>words</th></tr>` +
    data.elements
      .map(
        (e, i) =>
          `<tr><td><span class="swatch" style="background:${cssColor(e.color)}"></span> ${elName(i)}</td>${data.kinds.map((_, k) => `<td class="bit"><span class="dot ${hasKind(i, k) ? "on" : ""}"></span></td>`).join("")}<td>${data.kindBits
            .slice(i * data.kindWords, (i + 1) * data.kindWords)
            .map(hex32)
            .join(" ")}</td></tr>`,
      )
      .join("");

  // ---- raw bytes ----
  type Bytes = InspectData["buffers"]["elements"];
  type Field = Bytes["fields"][number];
  const describe = (name: string, buffer: Bytes, i: number, f: Field) => {
    if (f.type === "pad")
      return `<b>${name}[${i}]</b> · ${f.size} B of alignment padding, never read`;
    const view = new DataView(new Uint8Array(buffer.rows[i]).buffer);
    const values = Array.from({ length: f.size / 4 }, (_, k) =>
      f.type.includes("f")
        ? view.getFloat32(f.offset + k * 4, true)
        : view.getUint32(f.offset + k * 4, true),
    );
    const v = values[0];
    let shown =
      values.length > 1
        ? `(${values.join(", ")}) = ${pointName(values)}`
        : f.type.includes("f")
          ? String(v)
          : `${v} (${hex32(v)})`;
    if (f.name === "opcode") shown += ` = ${data.opcodes[v]}`;
    if (f.name === "accept") shown += ` = ${data.accepts[v]}`;
    if (f.name === "toType") shown += ` = ${data.tos[v]}`;
    if (f.name === "toId") shown += ` = ${elName(v)}`;
    if (f.name === "color") shown += ` = ${cssColor(v)}`;
    if (f.name === "chance") shown += ` = ${percent(v)}`;
    return `<b>${name}[${i}].${f.name}</b> · ${f.type} · ${shown}`;
  };
  const withPadding = (buffer: Bytes) => {
    const fields: Field[] = [];
    let cursor = 0;
    [...buffer.fields, { name: "", type: "end", offset: buffer.stride, size: 0 }]
      .sort((a, b) => a.offset - b.offset)
      .forEach((f) => {
        if (f.offset > cursor)
          fields.push({ name: "pad", type: "pad", offset: cursor, size: f.offset - cursor });
        if (f.type !== "end") fields.push(f);
        cursor = f.offset + f.size;
      });
    return fields;
  };
  const renderHex = (name: keyof InspectData["buffers"]) => {
    const buffer = data.buffers[name];
    const root = $("hex");
    root.innerHTML = buffer.rows.length
      ? ""
      : `<p class="sub">Empty. setup uploads one zeroed entry instead.</p>`;
    $("readout").textContent = "Hover a field to decode it.";
    const fields = withPadding(buffer);
    buffer.rows.forEach((bytes, i) => {
      const line = h("div", "hex-row", `<span class="ri">[${i}]</span>`);
      fields.forEach((f) => {
        const cell = h(
          "div",
          `field t-${f.type}`,
          `${i === 0 ? `<span class="fl" style="max-width:${f.size * 24}px">${f.name}</span>` : ""}<div class="bytes">${bytes
            .slice(f.offset, f.offset + f.size)
            .map((b) => `<span>${b.toString(16).padStart(2, "0")}</span>`)
            .join("")}</div>`,
        );
        cell.title = f.name;
        cell.addEventListener(
          "mouseenter",
          () => ($("readout").innerHTML = describe(name, buffer, i, f)),
        );
        line.append(cell);
      });
      root.append(line);
    });
  };
  (Object.keys(data.buffers) as (keyof InspectData["buffers"])[]).forEach((name, i) => {
    const buffer = data.buffers[name];
    const b = h("button", "", `${name} · ${buffer.rows.length * buffer.stride} B`);
    b.setAttribute("role", "tab");
    b.setAttribute("aria-selected", String(i === 0));
    b.addEventListener("click", () => {
      $("tabs")
        .querySelectorAll("button")
        .forEach((x) => x.setAttribute("aria-selected", String(x === b)));
      renderHex(name);
    });
    $("tabs").append(b);
  });
  renderHex("elements");
};

const escapeHtml = (s: string) => s.replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);

const STYLE = `
:root {
  --bg: #f3f4f7; --panel: #ffffff; --ink: #161a24; --muted: #5d6475; --line: #dde0e8;
  --accent: #3446c9; --accent-soft: #e4e7fb; --band: rgba(52, 70, 201, .10); --band-hot: rgba(52, 70, 201, .32);
  --f-u32: #dbe6ff; --f-vec: #d9f2e6; --f-f32: #fbe6cf; --f-pad: #eceef2; --f-pad-stripe: #d8dbe3;
  --warn: #9a5b00; --warn-soft: #fff1d9;
  --font-ui: system-ui, -apple-system, "Segoe UI", sans-serif;
  --font-data: ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0f1218; --panel: #171b24; --ink: #e6e9f1; --muted: #9aa2b4; --line: #2a303d;
    --accent: #8f9cff; --accent-soft: #232a4a; --band: rgba(143, 156, 255, .10); --band-hot: rgba(143, 156, 255, .34);
    --f-u32: #22305a; --f-vec: #1d3d31; --f-f32: #4a3418; --f-pad: #1e222c; --f-pad-stripe: #2b303c;
    --warn: #f0b552; --warn-soft: #3a2b12;
    color-scheme: dark;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); font: 15px/1.5 var(--font-ui); padding: 28px 16px 64px; }
.wrap { max-width: 1180px; margin: 0 auto; display: grid; gap: 28px; }
code, .mono { font-family: var(--font-data); font-size: 12px; font-variant-numeric: tabular-nums; }
.muted { color: var(--muted); }
h1, h2 { margin: 0; }
h1 { font-size: 26px; font-weight: 700; }
h2 { font-size: 17px; font-weight: 600; }
.eyebrow { font-family: var(--font-data); font-size: 11px; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }
.sub { color: var(--muted); margin: 0; max-width: 70ch; }
header, section { display: grid; gap: 12px; }
.meta { display: flex; flex-wrap: wrap; gap: 8px; }
.chip { font-family: var(--font-data); font-size: 11.5px; border: 1px solid var(--line); background: var(--panel); padding: 3px 9px; border-radius: 6px; }
.chip b { color: var(--muted); font-weight: 400; }
.sec-head { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: 6px 16px; }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 14px; display: grid; gap: 10px; min-width: 0; }
.bar { display: flex; height: 10px; border-radius: 3px; overflow: hidden; gap: 2px; }
.bar span { background: var(--accent); opacity: .85; }
.bar span:nth-child(2n) { opacity: .55; }
.binding { display: grid; grid-template-columns: 1fr auto auto; gap: 12px; align-items: baseline; padding: 4px 6px; border-radius: 5px; }
.binding:hover { background: var(--accent-soft); }
.binding .name { font-family: var(--font-data); font-size: 12px; }
.binding .type { font-family: var(--font-data); font-size: 11px; color: var(--muted); }
.binding .size { font-family: var(--font-data); font-size: 12px; text-align: right; min-width: 7ch; }
.scroll { overflow-x: auto; }
.map { position: relative; display: grid; grid-template-columns: minmax(220px, 1fr) minmax(260px, 1.15fr) minmax(300px, 1.3fr); column-gap: 64px; min-width: 900px; }
.map svg { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; overflow: visible; }
.band { fill: var(--band); stroke: var(--accent); stroke-opacity: .25; }
.band.reused { stroke-opacity: .9; stroke-width: 1.5; stroke-dasharray: 4 3; }
.band.hot { fill: var(--band-hot); }
.lane { display: grid; gap: 6px; align-content: start; position: relative; z-index: 1; }
.lane-title { display: flex; justify-content: space-between; align-items: baseline; padding: 0 2px 4px; }
.row { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px; display: grid; gap: 4px; transition: border-color .12s, box-shadow .12s, opacity .12s; }
.row:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.map.hovering .row { opacity: .38; }
.map.hovering .row.hot { opacity: 1; border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent); }
.row-top { display: flex; align-items: center; gap: 8px; }
.idx { font-family: var(--font-data); font-size: 11px; color: var(--muted); min-width: 2.2ch; }
.swatch { display: inline-block; vertical-align: -2px; width: 14px; height: 14px; border-radius: 3px; border: 1px solid rgba(0,0,0,.15); flex: none; }
.row-name { font-weight: 600; font-size: 14px; }
.row-fields { display: flex; flex-wrap: wrap; align-items: center; gap: 4px 12px; font-family: var(--font-data); font-size: 11px; color: var(--muted); }
.row-fields b { color: var(--ink); font-weight: 500; }
.row-fields.decoded { color: var(--ink); font-size: 11.5px; }
.tag { font-family: var(--font-data); font-size: 10.5px; padding: 1px 6px; border-radius: 4px; background: var(--accent-soft); color: var(--accent); }
.tag.warn { background: var(--warn-soft); color: var(--warn); }
.tag.dim { background: var(--f-pad); color: var(--muted); }
.op { font-family: var(--font-data); font-size: 11.5px; font-weight: 600; }
.bits { display: inline-flex; gap: 2px; }
.bits i { font-style: normal; width: 16px; height: 16px; border-radius: 3px; font-size: 9px; display: grid; place-items: center; background: var(--f-pad); color: var(--muted); }
.bits i.on { background: var(--accent); color: var(--panel); }
.legend { display: flex; flex-wrap: wrap; gap: 14px; font-size: 12.5px; color: var(--muted); align-items: center; }
.legend span { display: inline-flex; gap: 6px; align-items: center; }
.legend i { display: inline-block; width: 22px; height: 10px; border-radius: 2px; }
table { border-collapse: collapse; width: 100%; font-family: var(--font-data); font-size: 12px; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--line); white-space: nowrap; }
th { font-weight: 500; color: var(--muted); font-size: 11px; }
td.bit { text-align: center; }
.dot { display: inline-block; width: 12px; height: 12px; border-radius: 3px; background: var(--f-pad); }
.dot.on { background: var(--accent); }
.tabs { display: flex; flex-wrap: wrap; gap: 6px; }
.tabs button { font: 500 12.5px var(--font-ui); border: 1px solid var(--line); background: var(--panel); color: var(--ink); padding: 5px 12px; border-radius: 999px; cursor: pointer; }
.tabs button[aria-selected="true"] { background: var(--ink); color: var(--bg); border-color: var(--ink); }
.tabs button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.hex { display: grid; gap: 4px; width: max-content; }
.hex-row { display: flex; }
.hex-row .ri { width: 44px; font-family: var(--font-data); font-size: 11px; color: var(--muted); align-self: end; padding-bottom: 4px; }
.field { display: flex; flex-direction: column; justify-content: end; border-right: 2px solid var(--panel); }
.field .fl { font-family: var(--font-data); font-size: 9.5px; color: var(--muted); padding: 0 3px 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.field .bytes { display: flex; }
.field .bytes span { width: 24px; text-align: center; font-family: var(--font-data); font-size: 11px; padding: 4px 0; }
.t-u32 .bytes { background: var(--f-u32); } .t-vec2u .bytes { background: var(--f-vec); } .t-f32 .bytes { background: var(--f-f32); }
.t-pad .bytes { background: repeating-linear-gradient(135deg, var(--f-pad) 0 4px, var(--f-pad-stripe) 4px 6px); color: var(--muted); }
.field:hover .bytes { outline: 1px solid var(--accent); outline-offset: -1px; }
.readout { font-family: var(--font-data); font-size: 12px; color: var(--muted); min-height: 1.6em; }
.readout b { color: var(--ink); font-weight: 500; }
@media (prefers-reduced-motion: reduce) { .row { transition: none; } }
`;

/**
 * Renders the GPU buffers an automaton compiles to as a self-contained HTML page: the bind group,
 * a map of how elements point to rules and rules to conditions, kind membership bits and the raw
 * bytes of every buffer. Needs no WebGPU, so it also runs in Node.
 *
 * @param automaton - The automaton to compile, as returned by `vi.create()`.
 * @param options.title - The name shown in the page heading. Defaults to "automaton".
 * @returns The HTML document as a string, ready to save to a file or open in a new tab.
 * @throws If the automaton doesn't compile.
 */
export const inspect = (
  automaton: Automaton,
  { title = "automaton" }: { title?: string } = {},
): string => {
  // < is escaped so a name containing "</script>" can't close the data script.
  const json = JSON.stringify(buildData(automaton, title)).replace(/</g, "\\u003c");
  const name = escapeHtml(title);

  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${name} · buffers</title>
<style>${STYLE}</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>${name} · compiled GPU buffers</h1>
    <p class="sub">What the compiler uploads to the <code>automatonLayout</code> bind group, decoded back into the names you wrote. Hover any row to follow its pointers.</p>
    <div class="meta" id="meta"></div>
  </header>
  <section>
    <div class="sec-head"><h2>Bind group</h2><span class="eyebrow">automatonLayout · <span id="group-size"></span></span></div>
    <div class="card"><div class="bar" id="bar"></div><div id="bindings"></div></div>
  </section>
  <section>
    <div class="sec-head">
      <h2>Pointer map</h2>
      <div class="legend"><span><i style="background:var(--band-hot)"></i>[start, end) range</span><span><i style="border-top:2px dashed var(--accent);height:0"></i>reused condition range</span></div>
    </div>
    <div class="scroll">
      <div class="map" id="map">
        <svg id="links" aria-hidden="true"></svg>
        <div class="lane" id="lane-e"></div>
        <div class="lane" id="lane-r"></div>
        <div class="lane" id="lane-c"></div>
      </div>
    </div>
  </section>
  <section>
    <div class="sec-head"><h2>Kind bits</h2><span class="eyebrow">one row of kindWords words per element</span></div>
    <div class="card scroll"><table id="kinds"></table></div>
  </section>
  <section>
    <div class="sec-head"><h2>Raw bytes</h2><span class="eyebrow">little-endian · one row per array entry</span></div>
    <div class="tabs" role="tablist" id="tabs"></div>
    <div class="legend">
      <span><i style="background:var(--f-u32)"></i>u32</span>
      <span><i style="background:var(--f-vec)"></i>vec2u</span>
      <span><i style="background:var(--f-f32)"></i>f32</span>
      <span><i style="background:repeating-linear-gradient(135deg,var(--f-pad) 0 3px,var(--f-pad-stripe) 3px 5px)"></i>alignment padding</span>
    </div>
    <div class="card">
      <div class="readout" id="readout"></div>
      <div class="scroll"><div class="hex" id="hex"></div></div>
    </div>
  </section>
</div>
<script type="application/json" id="data">${json}</script>
<script>(${render.toString()})(JSON.parse(document.getElementById("data").textContent));</script>
</body>
</html>
`;
};
