# vivarium

## Work in progress

### Install

```shell
pnpm add @wonderyard/vivarium
```

### Import

```TypeScript
import { vivarium } from "@wonderyard/vivarium";
```

### Use

```TypeScript
const vi = vivarium();

const space = vi.element("space", "black");
const cat = vi.element("cat", "orange");

space.to(cat).count(cat, 3);
cat.to(cat).count(cat, [2, 3]);
cat.to(space);
```
