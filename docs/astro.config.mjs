// @ts-check

import starlight from "@astrojs/starlight";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "astro/config";

// https://astro.build/config
export default defineConfig({
  integrations: [
    starlight({
      title: "Vivarium",
      logo: {
        src: "./src/assets/glider.png",
        alt: "Glider logo",
      },
      favicon: "/favicon.png",
      customCss: ["./src/styles/global.css"],
      social: [
        {
          icon: "github",
          label: "GitHub",
          href: "https://github.com/wonderyard/vivarium",
        },
      ],
      sidebar: [
        {
          slug: "learn/intro",
        },
        {
          slug: "guides/start",
        },
        {
          label: "Building blocks",
          autogenerate: { directory: "concepts" },
        },
        {
          label: "Reference",
          collapsed: true,
          autogenerate: { directory: "reference" },
        },
      ],
    }),
  ],

  vite: {
    ssr: {
      noExternal: ["nanoid"],
    },
    plugins: [tailwindcss()],
  },
});
