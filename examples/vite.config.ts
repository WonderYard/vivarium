import { resolve } from "node:path";
import tailwindcss from "@tailwindcss/vite";
import typegpu from "unplugin-typegpu/vite";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [tailwindcss(), typegpu({})],
  resolve: {
    alias: {
      "@wonderyard/vivarium": resolve(__dirname, "../lib/main.ts"),
      "@/": `${resolve(__dirname, "../lib")}/`,
    },
  },
  build: {
    rollupOptions: {
      input: {
        main: resolve(__dirname, "index.html"),
        life: resolve(__dirname, "life/index.html"),
        "brians-brain": resolve(__dirname, "brians-brain/index.html"),
        wireworld: resolve(__dirname, "wireworld/index.html"),
        "forest-fire": resolve(__dirname, "forest-fire/index.html"),
      },
    },
  },
});
