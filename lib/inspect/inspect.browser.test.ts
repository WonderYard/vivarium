import { inspect } from "@/inspect/inspect";
import { vivarium } from "@/vivarium/vivarium";
import { expect, test } from "vitest";

test("renders every buffer row of the compiled automaton", async () => {
  const vi = vivarium();
  const flammable = vi.kind("flammable");
  const empty = vi.element("empty", "#1a1a2e");
  vi.element("tree", "#16a34a", [flammable]);
  // A name that would break the page if it weren't escaped.
  vi.element("</script><b>grass", "#84cc16", [flammable]);
  const fire = vi.element("fire", "#ef4444");
  flammable.to(fire).count(fire).accept("any");
  fire.to(empty);
  empty.to(fire).is(vi.neighbor.TOP, fire).chance(1, 50);

  const errors: unknown[] = [];
  const frame = document.createElement("iframe");
  frame.srcdoc = inspect(vi.create(), { title: "forest" });
  const loaded = new Promise((resolve) => (frame.onload = resolve));
  document.body.append(frame);
  await loaded;
  frame.contentWindow!.addEventListener("error", (e) => errors.push(e.error));
  const doc = frame.contentDocument!;

  expect(errors).toEqual([]);
  expect(doc.querySelectorAll("#lane-e .row")).toHaveLength(4);
  // the kind rule is copied into tree and grass
  expect(doc.querySelectorAll("#lane-r .row")).toHaveLength(4);
  expect(doc.querySelectorAll("#lane-c .row")).toHaveLength(3);
  expect(doc.querySelector("#bindings")!.textContent).toContain("array<GpuRule, 4>");
  expect(doc.querySelector("#lane-e")!.textContent).toContain("</script><b>grass");
  expect(doc.querySelector("#lane-r")!.textContent).toContain("reuses conditions of rule 1");
  // GpuCondition is 40 bytes with 8 bytes of padding
  (doc.querySelectorAll("#tabs button")[2] as HTMLElement).click();
  expect(doc.querySelectorAll("#hex .hex-row")).toHaveLength(3);
  expect(doc.querySelectorAll("#hex .hex-row:first-child .bytes span")).toHaveLength(40);
  expect(doc.querySelectorAll("#hex .hex-row:first-child .t-pad")).toHaveLength(2);
});
