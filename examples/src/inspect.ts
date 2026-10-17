import type { Automaton } from "@wonderyard/vivarium";
import { inspect } from "@wonderyard/vivarium/inspect";

// Opens the compiled GPU buffers of the automaton in a new tab when the "Inspect" button is clicked.
export const inspectOnClick = (automaton: Automaton, title: string) => {
  document.getElementById("inspect-button")?.addEventListener("click", () => {
    const html = new Blob([inspect(automaton, { title })], { type: "text/html" });
    window.open(URL.createObjectURL(html));
  });
};
