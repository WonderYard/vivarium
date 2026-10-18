/// <reference types="vitest/config" />

import { extname, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { playwright } from "@vitest/browser-playwright";
import { glob } from "glob";
import typegpu from "unplugin-typegpu/vite";
import { defineConfig } from "vite";
import dts from "vite-plugin-dts";

export default defineConfig({
  build: {
    lib: {
      entry: resolve(import.meta.dirname, "lib/main.ts"),
      formats: ["es"],
    },
    rolldownOptions: {
      external: [/^typegpu(\/|$)/, /^@typegpu\//],
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
  plugins: [dts({ include: ["lib"], exclude: ["lib/**/*.test.ts"] }), typegpu({})],

  resolve: {
    tsconfigPaths: true,
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
            // Tests read back GPU buffers and draw nothing to the page, so failure screenshots are blank.
            screenshotFailures: false,
            provider: playwright({
              launchOptions: {
                args: ["--enable-unsafe-webgpu", "--use-angle=swiftshader"],
              },
            }),
            instances: [{ browser: "chromium" }],
          },
        },
      },
    ],
  },
});
