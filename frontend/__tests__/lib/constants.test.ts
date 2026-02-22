import { describe, it, expect } from "vitest";
import {
  PORT_TYPE_COLORS,
  PORT_TYPE_BADGE_CLASSES,
  CATEGORY_COLORS,
  CATEGORY_LABELS,
  CATEGORY_COLORS_HEX,
  ALL_CATEGORIES,
} from "@/lib/constants";
import type { PortType, ModuleCategory } from "@/lib/types";

const ALL_PORT_TYPES: PortType[] = [
  "TEXT",
  "TOKEN_IDS",
  "TENSOR",
  "AD_TENSOR",
  "SCALAR",
  "INT",
];

describe("PORT_TYPE_COLORS", () => {
  it("has a color for every port type", () => {
    for (const pt of ALL_PORT_TYPES) {
      expect(PORT_TYPE_COLORS[pt]).toBeDefined();
      expect(PORT_TYPE_COLORS[pt]).toMatch(/^#[0-9a-f]{6}$/i);
    }
  });

  it("has no extra keys", () => {
    expect(Object.keys(PORT_TYPE_COLORS).sort()).toEqual(
      ALL_PORT_TYPES.slice().sort()
    );
  });
});

describe("PORT_TYPE_BADGE_CLASSES", () => {
  it("has classes for every port type", () => {
    for (const pt of ALL_PORT_TYPES) {
      expect(PORT_TYPE_BADGE_CLASSES[pt]).toBeDefined();
      expect(typeof PORT_TYPE_BADGE_CLASSES[pt]).toBe("string");
      expect(PORT_TYPE_BADGE_CLASSES[pt].length).toBeGreaterThan(0);
    }
  });
});

describe("CATEGORY_COLORS", () => {
  it("has a color class for every category", () => {
    for (const cat of ALL_CATEGORIES) {
      expect(CATEGORY_COLORS[cat]).toBeDefined();
      expect(typeof CATEGORY_COLORS[cat]).toBe("string");
    }
  });
});

describe("CATEGORY_LABELS", () => {
  it("has a label for every category", () => {
    for (const cat of ALL_CATEGORIES) {
      expect(CATEGORY_LABELS[cat]).toBeDefined();
      expect(typeof CATEGORY_LABELS[cat]).toBe("string");
      expect(CATEGORY_LABELS[cat].length).toBeGreaterThan(0);
    }
  });
});

describe("CATEGORY_COLORS_HEX", () => {
  it("has a hex color for every category", () => {
    for (const cat of ALL_CATEGORIES) {
      expect(CATEGORY_COLORS_HEX[cat]).toBeDefined();
      expect(CATEGORY_COLORS_HEX[cat]).toMatch(/^#[0-9a-f]{6}$/i);
    }
  });
});

describe("ALL_CATEGORIES", () => {
  it("contains all expected categories", () => {
    const expected: ModuleCategory[] = [
      "input",
      "preprocessing",
      "embedding",
      "normalization",
      "attention",
      "feedforward",
      "linear",
      "transformer",
      "math",
      "loss",
      "training",
    ];
    expect(ALL_CATEGORIES).toEqual(expected);
  });

  it("has no gaps - every category has colors and labels", () => {
    for (const cat of ALL_CATEGORIES) {
      expect(CATEGORY_COLORS[cat]).toBeTruthy();
      expect(CATEGORY_LABELS[cat]).toBeTruthy();
      expect(CATEGORY_COLORS_HEX[cat]).toBeTruthy();
    }
  });
});
