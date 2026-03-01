# Vivarium Studio — Google Stitch Design Prompt

> **Purpose**: This prompt describes the complete UX/UI design for **Vivarium Studio**, a single-page web application for creating and simulating cellular automata powered by the [vivarium](https://github.com/WonderYard/vivarium) library. The design targets implementation with **React**, **WebGPU**, **DaisyUI**, and **shadcn/ui**.

---

## Design System & Foundations

The application uses a **dark theme** (DaisyUI `dark` or `night` theme) with vivid accent colors. The palette centers on a deep navy/charcoal background (`#0f172a`) with emerald (`#34d399`) as the primary accent, sky blue (`#38bdf8`) as secondary, and amber (`#fbbf24`) for warnings and highlights. Text is near-white (`#e2e8f0`). All panels use DaisyUI `card` with `bg-base-200` and subtle `border-base-300` borders. Typography uses a monospace font for code/DSL tokens and a clean sans-serif (Inter) for UI labels.

Spacing follows a 4px grid. Border radius is `rounded-lg` throughout. Elevation uses `shadow-lg` on floating panels and `shadow-sm` on inline cards. Icons come from Lucide React. Transitions use `transition-all duration-200` for responsive feel.

---

## 1. Main View — App Shell

The main view is a **single-page layout** divided into four regions arranged in a responsive shell. The overall structure uses a CSS Grid with resizable panels (shadcn `ResizablePanelGroup`).

```
┌──────────────────────────────────────────────────────────────┐
│  TOP BAR (Navbar)                                            │
├────────────┬─────────────────────────────────┬───────────────┤
│            │                                 │               │
│  LEFT      │       CANVAS (center)           │  RIGHT        │
│  PANEL     │                                 │  PANEL        │
│            │                                 │               │
│  Elements  │   [WebGPU simulation grid]      │  Rule         │
│  & Kinds   │                                 │  Builder      │
│            │                                 │               │
│            ├─────────────────────────────────┤               │
│            │  CANVAS TOOLBAR (floating)      │               │
├────────────┴─────────────────────────────────┴───────────────┤
│  BOTTOM BAR (Simulation Controls + Timeline)                 │
└──────────────────────────────────────────────────────────────┘
```

### 1.1 Top Bar (Navbar)

A slim DaisyUI `navbar` spanning the full width. Height: 48px.

| Section | Contents |
|---------|----------|
| **Left** | App logo/icon + "Vivarium Studio" wordmark. A DaisyUI `dropdown` for project operations: **New**, **Open**, **Save**, **Export**, **Import**. |
| **Center** | Editable project name displayed as a DaisyUI `input input-ghost` that becomes editable on click. Below or beside it, a small DaisyUI `badge` showing the current neighborhood type ("square" or "cross") and wrapping status ("toroidal" / "bounded"). |
| **Right** | A DaisyUI `btn btn-ghost` **Undo** (↩) and **Redo** (↪). A `btn btn-ghost` for **Settings** (gear icon) that opens a shadcn `Sheet` from the right. A DaisyUI `btn btn-primary` **Share** button. |

### 1.2 Left Panel — Elements & Kinds

Width: 280px, collapsible to 48px icon rail via a toggle chevron. Uses a shadcn `ResizablePanel` with min/max constraints. When collapsed, only colored element icons are visible as a vertical strip.

#### 1.2.1 Elements Section

**Header row**: "Elements" label + a DaisyUI `btn btn-xs btn-circle btn-primary` with a `+` icon to add a new element.

**Element list**: A vertical scrollable list (shadcn `ScrollArea`) of element cards. Each element card is a horizontal row:

- A **color swatch** (24×24 rounded square) showing the element's color. Clicking it opens a shadcn `Popover` containing a compact color picker (hue ring + saturation/brightness square + hex input field).
- The **element name** as an inline-editable DaisyUI `input input-xs input-ghost`. The name validates for uniqueness in real time; duplicate names show a red border with a DaisyUI `tooltip` error.
- **Kind badges**: Zero or more DaisyUI `badge badge-sm badge-outline` showing each kind this element belongs to. A small `+` button at the end opens a shadcn `Popover` with a checkbox list of all defined kinds to toggle membership.
- A drag handle (⠿ grip icon) on the far left for reordering via drag-and-drop (this sets the element's index in the automaton).
- A **context menu** (right-click or `⋮` button) → shadcn `DropdownMenu` with: **Duplicate**, **Delete**, **Set as Default** (for canvas painting).

**Adding a new element**: Clicking `+` appends a new row at the bottom of the list with a generated name ("element-1", "element-2", ...) and a random color not already in use. The name field is immediately focused for editing.

#### 1.2.2 Kinds Section

Below the elements list, separated by a DaisyUI `divider`.

**Header row**: "Kinds" label + `btn btn-xs btn-circle btn-secondary` with `+`.

**Kind list**: Similar vertical list. Each kind row:

- A small **icon/glyph** (auto-assigned geometric shape: circle, diamond, triangle, square, star, hexagon — cycling) rendered in the secondary accent color.
- **Kind name** as inline-editable text.
- A **member count** DaisyUI `badge badge-sm` showing how many elements belong to this kind (e.g., "4 members").
- Expanding the row (DaisyUI `collapse`) reveals a horizontal strip of the member element color swatches for quick visual reference.
- Context menu: **Rename**, **Delete**, **Select All Members on Canvas**.

#### 1.2.3 Library Section (Reusable Components)

Below Kinds, separated by another `divider`. Header: "Library" + `+` button.

This section stores **saved patterns** — rectangular regions captured from the canvas. Each library item displays:

- A **thumbnail** (small preview of the pattern, rendered as a tiny pixelated grid).
- A **name** (editable).
- Dimensions label: e.g., "8×8", "16×32".
- Actions: **Place on Canvas** (enters stamp placement mode), **Edit** (re-opens the pattern in a mini canvas editor dialog), **Export** (downloads as `.vivarium-pattern` JSON), **Delete**.

Items are displayed in a 2-column grid of small DaisyUI `card card-compact` thumbnails within a `ScrollArea`. Drag-and-drop from the library onto the main canvas stamps the pattern at the drop location.

---

### 1.3 Center — Canvas & Drawing Area

The canvas is the core workspace, taking all remaining horizontal space. It renders the cellular automaton grid via WebGPU. The canvas supports pan (middle-click drag or two-finger drag) and zoom (scroll wheel or pinch). A subtle crosshair cursor replaces the default pointer when a drawing tool is active. Grid lines are optionally visible (toggle in canvas toolbar), rendered as thin 1px lines at `opacity-20`.

**Canvas coordinate badge**: A small floating DaisyUI `badge` in the bottom-right corner of the canvas showing the current cursor position as `(x, y)` and the grid dimensions as `W×H`.

#### 1.3.1 Canvas Toolbar (Floating)

A floating horizontal toolbar anchored to the top-center of the canvas area, styled as a DaisyUI `card bg-base-200/90 backdrop-blur` with `shadow-lg`. It contains icon-only DaisyUI `btn btn-sm btn-ghost` buttons organized into groups separated by vertical DaisyUI `divider divider-horizontal` separators:

**Group 1 — Selection & Navigation:**
| Tool | Icon | Behavior |
|------|------|----------|
| **Select** (default) | Pointer arrow | Click to select individual cells; drag to select rectangular regions. Selected region shows a dashed animated border (marching ants). |
| **Pan** | Hand/grab | Click-drag to pan the canvas viewport. |
| **Zoom to Fit** | Maximize icon | Resets zoom and pan so the full grid fills the canvas. |

**Group 2 — Drawing Tools:**
| Tool | Icon | Behavior |
|------|------|----------|
| **Pencil** | Pencil icon | Click or drag to paint cells with the currently selected element. Brush size controlled by a small DaisyUI `range range-xs` slider that appears in a popover on long-press/right-click of the pencil button. |
| **Eraser** | Eraser icon | Paints with the first element (index 0, treated as "empty/default"). |
| **Fill (Bucket)** | Paint bucket | Flood-fills a contiguous region of same-element cells with the selected element. |
| **Line** | Diagonal line | Click start point, drag to end point; draws a line of the selected element using Bresenham's algorithm. Preview line shown while dragging. |
| **Rectangle** | Rectangle outline | Click-drag to draw a filled or outline rectangle. A toggle in the tool options popover switches between filled/outline mode. |
| **Stamp** | Stamp/clone icon | When a library pattern is selected, this tool places it on the canvas at the click position. The stamp preview follows the cursor as a semi-transparent overlay. |

**Group 3 — Canvas Operations:**
| Tool | Icon | Behavior |
|------|------|----------|
| **Capture Selection** | Scissors + save icon | Active only when a rectangular selection exists. Captures the selected region and opens a shadcn `Dialog` to name it and save it to the Library. |
| **Clear Canvas** | Trash icon | Opens a DaisyUI `modal` confirmation: "Clear all cells? This cannot be undone." with **Cancel** and **Clear** (btn-error) buttons. Resets all cells to element index 0. |
| **Toggle Grid** | Grid icon | Toggles grid line visibility. Active state shown with `btn-active` class. |
| **Randomize** | Dice icon | Opens a shadcn `Popover` with sliders for each element's probability weight (DaisyUI `range` inputs). A **Fill** button randomizes the entire grid according to the weights. |

**Group 4 — Active Element Indicator:**

At the far right of the toolbar, the **currently selected painting element** is shown as a larger color swatch (32×32) with the element name below it. Clicking it opens a shadcn `Popover` grid showing all defined elements as clickable color swatches (quick element picker). This duplicates the ability to select from the left panel but provides canvas-local convenience.

#### 1.3.2 Canvas Interaction Flows

- **Painting**: Select an element (left panel click or toolbar picker), choose Pencil tool, click/drag on canvas. Each cell under the cursor calls `writeCellAt(x, y, elementIndex)`.
- **Rectangular Selection**: With Select tool, click-drag creates a selection rectangle. The selected region can then be:
  - **Copied** (`Ctrl+C`) to an internal clipboard.
  - **Cut** (`Ctrl+X`) — copies then fills the region with element 0.
  - **Pasted** (`Ctrl+V`) — enters stamp mode with the clipboard contents.
  - **Captured to Library** — via the toolbar button.
  - **Deleted** (`Delete` key) — fills with element 0.
- **Zoom**: Scroll wheel zooms toward cursor position. `Ctrl+0` resets zoom. Zoom level shown in the coordinate badge (e.g., "150%").
- **Pan**: Middle-click-drag or Space+left-click-drag. Also activated by Pan tool.

---

### 1.4 Right Panel — Rule Builder

Width: 340px, collapsible like the left panel. This is the core innovation: a visual, interactive rule builder that maps directly to vivarium's DSL but uses drag-and-drop tokens, condition cards, and visual connectors.

#### 1.4.1 Panel Header

"Rules" label + neighborhood indicator (`badge` showing "square 8-neighbor" or "cross 4-neighbor"). A DaisyUI `btn btn-xs btn-primary` "**+ New Rule**" button. A DaisyUI `dropdown` gear icon for panel settings: **Collapse All**, **Expand All**, **Sort by Element**, **Validate All Rules**.

#### 1.4.2 Rule Cards

Rules are displayed as a vertical list of **rule cards** in a `ScrollArea`. Each rule card is a DaisyUI `card card-compact bg-base-300` with a left border colored to match the "from" element. Cards can be reordered via drag-and-drop (grip handle on the left).

**A rule card has the following anatomy:**

```
┌─[color bar]──────────────────────────────────────────┐
│  ⠿  [FROM token]  →  [TO token]           ⋮  ▼/▲   │
│                                                      │
│  ┌─ Conditions ─────────────────────────────────┐    │
│  │  [condition pill 1]                          │    │
│  │  [condition pill 2]                          │    │
│  │  [+ Add Condition]                           │    │
│  └──────────────────────────────────────────────┘    │
│                                                      │
│  Strategy: (● all) (○ any) (○ one) (○ none)         │
└──────────────────────────────────────────────────────┘
```

**FROM token**: A styled DaisyUI `badge badge-lg` showing the element or kind name with its color swatch. This is a **drop target** — you can drag an element or kind from the left panel and drop it here to set the "from" source. Clicking the token opens a shadcn `Popover` with a selectable list of all elements and kinds.

**Arrow (→)**: A static styled arrow divider between from and to.

**TO token**: Same badge style as FROM. Drop target for elements. It can also accept a **neighbor position** token (see §1.4.4). Clicking opens a popover listing all elements plus a "Neighbor Position" submenu showing the spatial grid selector.

**Condition pills**: Each condition is rendered as a horizontal **pill** (DaisyUI `badge badge-lg` variant with a distinct background):

- **Count condition** — pill colored blue:
  ```
  [🔢 count] [element-token ▼] [number-chips: 1 2 3 ...]
  ```
  - The element/kind reference is a small inline drop target (drag from left panel or click to pick).
  - The count values are shown as a row of small toggleable **number chips** (DaisyUI `btn btn-xs` for each number 0 through max-neighbors). Clicked chips toggle on/off (active = `btn-primary`, inactive = `btn-ghost`). This directly maps to `count(element, ...values)`.
  - A shadcn `Popover` on a `⚙` icon offers helper shortcuts: **Between** (two number inputs), **Not** (invert selection), **Even**, **Odd**, **Any** (all non-zero). These correspond to `vi.helpers.between()`, `vi.helpers.not()`, `vi.helpers.even()`, `vi.helpers.odd()`.

- **Is condition** — pill colored purple:
  ```
  [📍 is] [position-token ▼] [= element-token ▼]
  ```
  - The **position token** is a visual mini-grid selector: a tiny 3×3 (square) or 3×3-cross (cross) grid rendered inline. The user clicks a cell in the mini-grid to select the neighbor position (TOP_LEFT, TOP, TOP_RIGHT, LEFT, RIGHT, BOTTOM_LEFT, BOTTOM, BOTTOM_RIGHT for square; TOP, LEFT, RIGHT, BOTTOM for cross). The selected cell is highlighted with the primary accent. SELF is always dimmed/unselectable.
  - The **element reference** is a drop target / clickable picker for elements and kinds.

- **Chance condition** — pill colored amber:
  ```
  [🎲 chance] [numerator input] / [denominator input]
  ```
  - Two small DaisyUI `input input-xs` number fields for the fraction numerator and denominator. A real-time label shows the computed percentage (e.g., "0.1%") and a tiny DaisyUI `progress` bar fills proportionally for visual feedback.

**"+ Add Condition" button**: A DaisyUI `btn btn-xs btn-ghost btn-dashed` that opens a shadcn `DropdownMenu` with three choices: **Count**, **Is**, **Chance**. Selecting one appends a new condition pill to the card with default values.

**Strategy radio group**: A row of DaisyUI `radio radio-xs` buttons labeled **all**, **any**, **one**, **none**. Defaults to "all". Maps directly to the `accept()` strategy. Visible only when two or more conditions exist; otherwise hidden (single-condition rules don't need a strategy).

**Card actions** (the `⋮` menu): shadcn `DropdownMenu` with: **Duplicate Rule**, **Delete Rule**, **Move to Top**, **Move to Bottom**, **Disable/Enable** (disabled rules are shown at 50% opacity with a strikethrough).

**Collapse/Expand** (`▼/▲`): Toggles the condition section visibility. Collapsed cards show only the FROM → TO header with a summary like "2 conditions, strategy: any".

#### 1.4.3 Creating a New Rule

Clicking "+ New Rule" appends a new empty rule card at the bottom with placeholder tokens:

```
[? select element] → [? select element]
```

Both FROM and TO tokens pulse gently with a `animate-pulse` ring to indicate they need to be filled. The user can:
1. **Click** either token to open the element/kind picker popover.
2. **Drag** an element from the left panel and **drop** it onto a token.
3. **Use a keyboard shortcut**: with the rule card focused, type the first letters of an element name to filter and select (shadcn `Command` palette style filtering inside the popover).

#### 1.4.4 Neighbor Position Tokens (for `to` with spatial rules)

When the TO target of a rule is a neighbor direction (used in Langton's Ant-style automata), the TO token shows a directional arrow icon with the position name. Selecting "Neighbor Position" from the TO popover opens an inline mini-grid where the user clicks the desired direction. The selected direction is shown as an arrow icon (↑ ↗ → ↘ ↓ ↙ ← ↖) with the position label.

#### 1.4.5 Rule Validation

Rules are validated in real time:
- Missing FROM or TO: the token border turns red with a DaisyUI `tooltip` saying "Required".
- Count condition with no values selected: the number chips area has a red border.
- Referencing a deleted element/kind: the token shows a red "⚠ missing" badge.
- A small validation summary appears above the rule list: "✓ 12 rules valid" (green) or "⚠ 2 rules have errors" (amber, clickable to scroll to the first error).

#### 1.4.6 Drag-and-Drop Interactions

The rule builder supports these drag-and-drop flows:

| Source | Target | Action |
|--------|--------|--------|
| Element from left panel | FROM token | Sets rule source element |
| Element from left panel | TO token | Sets rule target element |
| Kind from left panel | FROM token | Sets rule source kind |
| Kind from left panel | Count condition reference | Sets count reference |
| Element from left panel | Is condition reference | Sets is reference |
| Element from left panel | Active element indicator | Sets painting element |
| Library item | Canvas | Stamps pattern at drop position |
| Rule card (grip) | Another rule card | Reorders rules |
| Condition pill (grip) | Another position in same card | Reorders conditions |
| Condition pill | Different rule card | Moves condition to another rule |

Drop targets highlight with a dashed border and `bg-primary/10` background when a compatible drag is hovering over them.

---

### 1.5 Bottom Bar — Simulation Controls & Timeline

A fixed-height bar (56px) at the bottom spanning the full width, styled as a DaisyUI `card bg-base-200` with a top border.

**Layout** (flex row, items centered):

```
[ ⏮ | ⏪ | ▶/⏸ | ⏩ | ⏭ ]   Step: 1,247   |   Speed: [━━━━○━━━━]   |   Grid: 256×256   |   FPS: 60   |   [ ⚙ Config ]
```

| Control | Component | Behavior |
|---------|-----------|----------|
| **Step Back** (⏮) | `btn btn-sm btn-ghost` | Rewinds one step (if history is enabled). Disabled during playback. |
| **Slow** (⏪) | `btn btn-sm btn-ghost` | Halves the current simulation speed. |
| **Play/Pause** (▶/⏸) | `btn btn-sm btn-primary` | Toggles simulation loop. Icon toggles between play and pause. When paused, the button has a subtle breathing glow animation. |
| **Fast** (⏩) | `btn btn-sm btn-ghost` | Doubles the current simulation speed. |
| **Step Forward** (⏭) | `btn btn-sm btn-ghost` | Advances exactly one simulation step. Only works when paused. Calls `update()` + `draw()` once. |
| **Step Counter** | DaisyUI `badge badge-ghost` | Shows the current generation number, formatted with commas. |
| **Speed Slider** | DaisyUI `range range-xs range-primary` | Logarithmic slider from 1 step/sec to max FPS. Current speed shown as a tooltip on hover. |
| **Grid Size** | `badge badge-ghost` | Shows current grid dimensions. Clicking opens a shadcn `Popover` to resize (select from power-of-2 presets: 32, 64, 128, 256, 512; or type a custom power-of-2 value). Resizing prompts a confirmation since it clears the grid. |
| **FPS Counter** | `badge badge-ghost` | Live frames-per-second counter with color coding: green (>30), yellow (15–30), red (<15). |
| **Config** (⚙) | `btn btn-sm btn-ghost` | Opens the simulation config popover/sheet. |

**Simulation Config (shadcn `Sheet` from bottom or `Popover`):**

| Setting | Control | Maps To |
|---------|---------|---------|
| **Neighborhood** | DaisyUI `toggle` labeled "Square (8) / Cross (4)" | `vivarium("square")` vs `vivarium("cross")` |
| **Wrapping** | DaisyUI `toggle` labeled "Toroidal Wrapping" | `{ wrapping: true }` option |
| **History Size** | DaisyUI `range` + number input (0 = disabled, max 1000) | How many past states to keep for step-back |
| **Auto-pause on Error** | DaisyUI `toggle` | Pause simulation if a rule validation error occurs |

Changing neighborhood or wrapping requires re-creating the automaton. A warning DaisyUI `alert alert-warning` appears: "Changing neighborhood type will reset all rules. Continue?" with **Cancel** and **Continue** buttons.

---

## 2. Subviews & Dialogs

### 2.1 New Project Dialog

Opened from Top Bar → New. A shadcn `Dialog` centered on screen.

**Contents:**
- **Project Name**: DaisyUI `input` with placeholder "My Automaton".
- **Neighborhood**: A visual toggle between two illustrated options — a 3×3 grid labeled "Square (8 neighbors)" and a cross-shaped grid labeled "Cross (4 neighbors)". Each option is a DaisyUI `card` that becomes `card-bordered border-primary` when selected.
- **Grid Size**: A row of preset DaisyUI `btn btn-sm` buttons: 64, 128, 256, 512. A "Custom" button reveals a number input (must be power of 2).
- **Wrapping**: DaisyUI `toggle` with label "Enable toroidal wrapping (edges connect)".
- **Template**: An optional shadcn `Select` dropdown with presets: "Blank", "Game of Life", "Forest Fire", "Brian's Brain", "Wireworld", "Langton's Ant". Selecting a template pre-populates elements, kinds, and rules.
- **Footer**: **Cancel** (`btn btn-ghost`) and **Create** (`btn btn-primary`).

### 2.2 Element Color Picker (Popover)

A shadcn `Popover` appearing when clicking an element's color swatch.

**Contents:**
- A **hue strip** (horizontal gradient bar, 360°). Click or drag to set hue.
- A **saturation/brightness square** (2D picker). Click or drag to set saturation and brightness.
- A **hex input** (`input input-sm`) for typing exact values like `#34d399`.
- A **preset palette**: Two rows of 12 common colors as small clickable swatches (the colors from DaisyUI's theme: primary, secondary, accent, info, success, warning, error, plus neutrals and extra vivid colors).
- A **"used colors" row** showing colors already assigned to other elements (dimmed, with a "⚠ already in use" tooltip if clicked).
- The element preview updates live as the user adjusts the color.

### 2.3 Capture to Library Dialog

Opened when the user selects a region on the canvas and clicks "Capture Selection".

A shadcn `Dialog` with:
- A **preview** of the captured region rendered as a pixel grid at display scale.
- **Name**: DaisyUI `input` with auto-generated default based on contents (e.g., "pattern-3" or "glider").
- **Description**: DaisyUI `textarea textarea-sm` (optional).
- **Tags**: A DaisyUI `input` with auto-complete for existing tags. Tags are shown as `badge badge-sm` with `×` remove buttons.
- **Footer**: **Cancel** and **Save to Library** (`btn btn-primary`).

### 2.4 Library Item Detail (Modal)

Double-clicking a library item or choosing "Edit" opens a shadcn `Dialog` (larger, 600×500px).

**Contents:**
- A **mini canvas** showing the pattern at a comfortable zoom level. The user can edit individual cells with pencil/eraser tools (simplified toolbar: pencil, eraser, and element picker only).
- **Metadata panel** on the right: name, description, tags, dimensions, creation date.
- **Actions**: **Save Changes**, **Export Pattern**, **Duplicate**, **Delete**.

### 2.5 Export Dialog

Opened from Top Bar → Export. A shadcn `Dialog`.

**Export Options (DaisyUI `tabs` or shadcn `Tabs`):**

| Tab | Description | Format |
|-----|-------------|--------|
| **Project** | Exports the entire project (elements, kinds, rules, grid state, library) | `.vivarium` JSON file |
| **Rules Only** | Exports just the automaton definition (elements, kinds, rules — no grid state) | `.vivarium-rules` JSON |
| **Grid Snapshot** | Exports the current grid state as data | `.vivarium-grid` JSON |
| **Image** | Exports the current canvas as a PNG image | `.png` |
| **Library** | Exports the entire component library | `.vivarium-library` JSON |

Each tab shows a preview of what will be exported and a **Download** (`btn btn-primary`) button. The filename is editable.

### 2.6 Import Dialog

Opened from Top Bar → Import. A shadcn `Dialog`.

**Contents:**
- A **drag-and-drop zone** (dashed border area, `border-2 border-dashed rounded-lg`) with icon and text "Drop a `.vivarium`, `.vivarium-rules`, `.vivarium-grid`, `.vivarium-library`, or `.vivarium-pattern` file here, or click to browse."
- File type is auto-detected from the JSON structure.
- After file selection, a **preview panel** shows what will be imported:
  - For projects: element count, kind count, rule count, grid dimensions.
  - For rules: list of elements and rules with a diff against current (new items in green, conflicts in amber).
  - For patterns: a visual preview of the pattern.
- **Merge vs Replace** options (DaisyUI `radio` group):
  - **Replace**: Completely replaces current data of that type.
  - **Merge**: Adds new items, skips duplicates (by name). Conflicts highlighted for user resolution.
- **Footer**: **Cancel** and **Import** (`btn btn-primary`).

### 2.7 Settings Sheet

Opened from the gear icon in the Top Bar. A shadcn `Sheet` sliding in from the right.

**Sections (shadcn `Accordion`):**

**Display:**
- Grid line visibility: DaisyUI `toggle`
- Grid line color: color swatch (mini picker)
- Canvas background color: color swatch
- Cell gap (0–2px): DaisyUI `range range-xs`
- Show cell coordinates on hover: DaisyUI `toggle`

**Performance:**
- Target FPS: DaisyUI `select` (30, 60, 120, uncapped)
- History buffer size: number input
- GPU adapter info: read-only text showing the WebGPU adapter

**Keyboard Shortcuts:**
- A two-column table showing all shortcuts with the ability to remap them. Each row: action name | DaisyUI `kbd` showing current binding | "Edit" button.

**About:**
- Version number, vivarium library version, links.

### 2.8 Keyboard Shortcuts Quick Reference

Triggered by pressing `?` or from Settings. A shadcn `Dialog` showing a formatted table:

| Category | Shortcut | Action |
|----------|----------|--------|
| **Tools** | `V` | Select tool |
| | `B` | Pencil (brush) |
| | `E` | Eraser |
| | `G` | Fill (bucket) |
| | `L` | Line tool |
| | `R` | Rectangle tool |
| | `S` | Stamp tool |
| | `H` | Pan (hand) |
| **Canvas** | `Ctrl+0` | Zoom to fit |
| | `Ctrl++` / `Ctrl+-` | Zoom in / out |
| | `Ctrl+G` | Toggle grid lines |
| **Editing** | `Ctrl+C` | Copy selection |
| | `Ctrl+X` | Cut selection |
| | `Ctrl+V` | Paste (enter stamp mode) |
| | `Delete` | Clear selection |
| | `Ctrl+Z` | Undo |
| | `Ctrl+Shift+Z` | Redo |
| **Simulation** | `Space` | Play / Pause |
| | `.` | Step forward (when paused) |
| | `,` | Step backward (when paused) |
| | `[` / `]` | Decrease / Increase speed |
| **File** | `Ctrl+N` | New project |
| | `Ctrl+O` | Open / Import |
| | `Ctrl+S` | Save / Export project |

---

## 3. User Interaction Flows

### 3.1 Flow: Creating a New Automaton from Scratch

1. User opens the app → sees the New Project dialog (or a blank canvas with "Get Started" watermark).
2. User sets project name, chooses neighborhood type (square/cross), grid size, and wrapping.
3. Clicks **Create** → empty canvas appears, left panel has no elements.
4. User clicks `+` in the Elements section → first element created (auto-named "element-0", random color).
5. User renames it to "space", picks a dark blue color.
6. Repeats for more elements (e.g., "alive" → green).
7. Optionally creates kinds and assigns elements to them.
8. Opens Rule Builder (right panel), clicks **+ New Rule**.
9. Drags "space" from left panel onto the FROM token.
10. Drags "alive" onto the TO token.
11. Clicks **+ Add Condition** → selects **Count**.
12. In the count pill, drags "alive" onto the reference slot.
13. Clicks number chip **3** to activate it.
14. Rule now reads: `space → alive when count(alive, 3)`.
15. Creates more rules similarly.
16. Switches to Pencil tool, selects "alive" element, draws initial pattern on canvas.
17. Presses **Play** → simulation runs.

### 3.2 Flow: Building a Rule with Multiple Conditions

1. User has a rule card open (e.g., `tree → fire`).
2. Clicks **+ Add Condition** → **Chance** → chance pill appears with inputs `1 / 100000`.
3. Clicks **+ Add Condition** → **Count** → count pill appears.
4. Drags "fire" element onto the count reference.
5. Clicks "Any" helper shortcut → all non-zero chips activate (equivalent to `count(fire)`).
6. Now two conditions exist. The strategy radio group becomes visible.
7. User selects **any** strategy → rule reads: `tree → fire when chance(1/100000) OR count(fire) [any]`.

### 3.3 Flow: Capturing and Reusing a Pattern

1. User selects the **Select** tool and drags a rectangle around an interesting pattern (e.g., a glider).
2. The selection shows marching ants border.
3. User clicks the **Capture Selection** button in the canvas toolbar.
4. The Capture to Library dialog opens with a preview.
5. User names it "Glider", adds tag "spaceship", clicks **Save to Library**.
6. The pattern appears in the Library section of the left panel as a thumbnail.
7. Later, user selects the **Stamp** tool, clicks the "Glider" library item.
8. The pattern preview follows the cursor on the canvas.
9. User clicks on the canvas to place the glider. Can place multiple copies.
10. User can also drag the library thumbnail directly onto the canvas.

### 3.4 Flow: Exporting and Importing

**Export:**
1. User clicks Top Bar → Export (or `Ctrl+S`).
2. Export dialog opens with tabs.
3. User selects "Project" tab → sees summary (5 elements, 2 kinds, 8 rules, 256×256 grid).
4. Clicks **Download** → file `my-automaton.vivarium` is saved.

**Import:**
1. User clicks Top Bar → Import (or `Ctrl+O`).
2. Drags a `.vivarium` file into the drop zone.
3. Preview shows the project contents.
4. User selects **Replace** and clicks **Import**.
5. The app loads the project: elements populate the left panel, rules populate the right panel, the grid state renders on the canvas.

### 3.5 Flow: Using a Template

1. User opens New Project dialog.
2. Selects "Game of Life" from the Template dropdown.
3. Clicks **Create**.
4. App auto-populates: two elements ("space" dark blue, "alien" green — matching the vivarium library convention), three rules (birth at 3, survival at 2–3, death otherwise).
5. Canvas is blank but ready. User draws an initial pattern and hits Play.

---

## 4. Responsive & Accessibility Considerations

### Responsive Behavior
- **≥1440px**: Full three-panel layout as described.
- **1024–1439px**: Left and right panels are collapsible. Default state: left panel open, right panel collapsed to a tab strip showing "Rules" that expands on click.
- **768–1023px**: Left and right panels become toggleable DaisyUI `drawer` overlays that slide over the canvas. Toggle buttons appear in the top bar.
- **<768px**: Mobile is not a primary target but the app degrades gracefully: single-view navigation with a bottom DaisyUI `btm-nav` to switch between Canvas, Elements, Rules, and Library views.

### Accessibility
- All interactive elements have `aria-label` attributes.
- Drag-and-drop has keyboard alternatives: focus a token, press `Enter` to open the picker, use arrow keys to navigate, `Enter` to select.
- Color swatches include the color name in a `tooltip` for color-blind users.
- Rule cards are focusable and navigable with `Tab`/`Shift+Tab`.
- Simulation controls are accessible via keyboard shortcuts.
- High-contrast mode toggle in Settings that increases borders and uses pattern fills alongside colors for element differentiation on the canvas.

---

## 5. Visual Design Details

### Animation & Micro-interactions
- **Element drag**: 95% scale on pickup, subtle drop shadow, smooth snap-back if dropped outside a valid target.
- **Token drop**: A brief scale-up pulse (1.05×) with a green ring flash on successful drop.
- **Rule card creation**: Slide-down entrance with `animate-slide-in-from-top` (200ms ease-out).
- **Condition pill add**: Fade-in + slide-right entrance (150ms).
- **Play button**: When paused, the play icon has a slow breathing glow (`animate-pulse` with primary color shadow). When playing, it's solid.
- **Canvas cell painting**: Cells update immediately (optimistic UI via `writeCellAt`) with no perceptible delay.
- **Simulation running**: The step counter increments smoothly with a number-flip animation.
- **Error states**: Rule tokens with errors shake briefly (`animate-shake`, 300ms) when the user tries to run the simulation with invalid rules.

### Color Coding for Rule Conditions
- **Count conditions**: Blue-tinted background (`bg-info/20`).
- **Is conditions**: Purple-tinted background (`bg-secondary/20`).
- **Chance conditions**: Amber-tinted background (`bg-warning/20`).
- **Strategy badges**: Ghost-styled DaisyUI badges with the text of the strategy.

### Empty States
- **No elements defined**: Left panel shows a centered illustration (stylized cell grid icon) with "Define your first element to get started" and a prominent "**+ Create Element**" button.
- **No rules defined**: Right panel shows a flow diagram illustration with "Rules define how elements transform" and "**+ Create Rule**" button.
- **Empty library**: Library section shows "Capture patterns from the canvas to build your library" with a small instructional illustration.
- **Blank canvas**: A subtle watermark in the center: the Vivarium logo at 5% opacity with "Draw your initial state" below it. The watermark disappears once any cell is painted.

---

## 6. Data Model (JSON Schema for Export/Import)

The `.vivarium` project file follows this structure (for implementation reference). The `grid` field is a flat array of element indices with length `width × height`, stored in row-major order. The example below is abbreviated for brevity:

```json
{
  "version": "1.0",
  "name": "My Automaton",
  "neighborhood": "square",
  "wrapping": false,
  "gridSize": { "width": 256, "height": 256 },
  "elements": [
    { "id": "el_0", "name": "space", "color": "#04153b", "kinds": [] },
    { "id": "el_1", "name": "alien", "color": "#34d399", "kinds": ["k_0"] }
  ],
  "kinds": [
    { "id": "k_0", "name": "living" }
  ],
  "rules": [
    {
      "from": { "type": "element", "id": "el_0" },
      "to": { "type": "element", "id": "el_1" },
      "conditions": [
        { "type": "count", "reference": { "type": "element", "id": "el_1" }, "values": [3] }
      ],
      "accept": "all"
    }
  ],
  "grid": [0, 0, 1, 0, 1, 0, 0, 1, 0],
  "library": [
    {
      "id": "lib_0",
      "name": "Glider",
      "description": "A classic spaceship",
      "tags": ["spaceship"],
      "width": 3,
      "height": 3,
      "cells": [0, 1, 0, 0, 0, 1, 1, 1, 1]
    }
  ]
}
```

---

## 7. Technology Implementation Notes

- **Rendering**: The canvas grid is rendered entirely via WebGPU compute and render pipelines using vivarium's built-in `setup()` function. Cell painting uses `writeCellAt()` for individual cells and `writeGrid()` for bulk operations (paste, clear, randomize, import grid).
- **State Management**: React state (via Zustand or similar) manages UI state (selected tool, selected element, panel visibility). The automaton state lives on the GPU; `readGrid()` is called only when needed (export, capture to library).
- **Rule Compilation**: When the user modifies rules in the Rule Builder, the app reconstructs the automaton by calling the vivarium DSL programmatically (`vivarium()` → `.element()` → `.kind()` → `.to().count()` etc. → `.create()`). The new automaton is hot-swapped via `setAutomaton()` without resetting the grid.
- **Drag-and-Drop**: Implemented with `@dnd-kit/core` for accessible, performant DnD interactions.
- **Canvas Interaction Layer**: A transparent HTML overlay div sits above the WebGPU canvas to handle mouse/touch events for drawing tools. The overlay translates pixel coordinates to grid cell coordinates accounting for zoom and pan transforms.
- **DaisyUI**: All standard UI components (buttons, inputs, cards, badges, toggles, modals, tooltips, navbars, drawers) use DaisyUI classes on Tailwind CSS.
- **shadcn/ui**: Complex interactive components use shadcn: `ResizablePanelGroup` (panel layout), `Command` (search/filter in pickers), `Popover` (color picker, element picker, tool options), `Sheet` (settings), `Dialog` (modals), `DropdownMenu` (context menus), `Tabs` (export dialog), `Accordion` (settings sections), `ScrollArea` (scrollable lists).
