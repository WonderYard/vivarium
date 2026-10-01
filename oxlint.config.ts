import typegpu from "eslint-plugin-typegpu";
import { defineConfig } from "oxlint";

export default defineConfig({
  plugins: ["eslint", "typescript", "unicorn", "oxc", "import", "jsdoc", "promise", "vitest"],
  jsPlugins: ["eslint-plugin-typegpu"],
  rules: {
    "vitest/no-conditional-expect": "off",
    ...typegpu.configs.recommended.rules,
  },
});
