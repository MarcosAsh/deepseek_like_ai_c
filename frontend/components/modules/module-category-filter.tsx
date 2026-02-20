"use client";

import { Button } from "@/components/ui/button";
import { CATEGORY_LABELS, ALL_CATEGORIES } from "@/lib/constants";
import { cn } from "@/lib/utils";

export function ModuleCategoryFilter({
  selected,
  onSelect,
  categories,
}: {
  selected: string | null;
  onSelect: (category: string | null) => void;
  categories: string[];
}) {
  // Only show categories that exist in the modules
  const available = ALL_CATEGORIES.filter((c) => categories.includes(c));

  return (
    <div className="flex gap-2 overflow-x-auto pb-2">
      <Button
        variant={selected === null ? "default" : "outline"}
        size="sm"
        className="shrink-0"
        onClick={() => onSelect(null)}
      >
        All
      </Button>
      {available.map((category) => (
        <Button
          key={category}
          variant={selected === category ? "default" : "outline"}
          size="sm"
          onClick={() => onSelect(category === selected ? null : category)}
          className={cn("shrink-0", selected === category && "shadow-sm")}
        >
          {CATEGORY_LABELS[category] || category}
        </Button>
      ))}
    </div>
  );
}
