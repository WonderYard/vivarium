import { resolve } from "node:path";
import tailwindcss from "@tailwindcss/vite";
import typegpu from "unplugin-typegpu/vite";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [tailwindcss(), typegpu({})],
  resolve: {
    tsconfigPaths: true,
  },
  build: {
    rolldownOptions: {
      input: {
        main: resolve(import.meta.dirname, "index.html"),
        life: resolve(import.meta.dirname, "life/index.html"),
        "brians-brain": resolve(import.meta.dirname, "brians-brain/index.html"),
        wireworld: resolve(import.meta.dirname, "wireworld/index.html"),
        "forest-fire": resolve(import.meta.dirname, "forest-fire/index.html"),
        "langtons-ant": resolve(import.meta.dirname, "langtons-ant/index.html"),
      },
    },
  },
});
