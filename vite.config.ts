/// <reference types="vitest/config" />

import { extname, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import basicSsl from "@vitejs/plugin-basic-ssl";
import { playwright } from "@vitest/browser-playwright";
import { glob } from "glob";
import typegpu from "unplugin-typegpu/vite";
import { defineConfig } from "vite";
import dts from "vite-plugin-dts";

export default defineConfig({
  build: {
    lib: {
      entry: resolve(__dirname, "lib/main.ts"),
      formats: ["es"],
    },
    rollupOptions: {
      input: Object.fromEntries(
        glob
          .sync("lib/**/*.ts", {
            ignore: ["lib/**/*.d.ts", "lib/**/*.test.*"],
          })
          .map((file) => [
            relative("lib", file.slice(0, file.length - extname(file).length)),
            fileURLToPath(new URL(file, import.meta.url)),
          ]),
      ),
      output: {
        assetFileNames: "assets/[name][extname]",
        entryFileNames: "[name].js",
      },
    },
  },
  plugins: [
    process.env.ENABLE_BASIC_SSL === "true" && basicSsl(),
    dts({ include: ["lib"] }),
    typegpu({}),
  ],

  resolve: {
    alias: {
      "@/": new URL("./lib/", import.meta.url).pathname,
    },
  },

  test: {
    projects: [
      {
        extends: true,
        test: {
          name: "unit",
          include: ["lib/**/*.test.ts"],
          exclude: ["lib/**/*.browser.test.ts"],
        },
      },
      {
        extends: true,
        test: {
          name: "browser",
          include: ["lib/**/*.browser.test.ts"],
          browser: {
            enabled: true,
            headless: true,
            provider: playwright({
              launchOptions: {
                args: ["--enable-unsafe-webgpu"],
              },
            }),
            instances: [{ browser: "chromium" }],
          },
        },
      },
    ],
  },
});
