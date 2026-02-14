import { defineConfig } from "vitest/config";
import typegpu from "unplugin-typegpu/vite";
import { playwright } from "@vitest/browser-playwright";

export default defineConfig({
  plugins: [typegpu({})],
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
            provider: playwright({
              launchOptions: {
                args: [
                  "--enable-unsafe-webgpu",
                  "--enable-features=Vulkan",
                ],
              },
            }),
            instances: [
              {
                browser: "chromium",
              },
            ],
          },
        },
      },
    ],
  },
});
