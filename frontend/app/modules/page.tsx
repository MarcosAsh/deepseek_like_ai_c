"use client";

import { useQuery } from "@tanstack/react-query";
import { fetchModules } from "@/lib/api";
import { ModuleCard } from "@/components/modules/module-card";
import { ModuleCategoryFilter } from "@/components/modules/module-category-filter";
import { Skeleton } from "@/components/ui/skeleton";
import { Input } from "@/components/ui/input";
import { Search } from "lucide-react";
import { useState, useMemo } from "react";

export default function ModulesPage() {
  const [category, setCategory] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const { data, isLoading, error } = useQuery({
    queryKey: ["modules"],
    queryFn: fetchModules,
  });

  const modules = useMemo(() => data?.modules ?? [], [data]);

  const categories = useMemo(
    () => [...new Set(modules.map((m) => m.category))],
    [modules]
  );

  const filtered = useMemo(() => {
    let result = modules;
    if (category) {
      result = result.filter((m) => m.category === category);
    }
    if (search.trim()) {
      const q = search.toLowerCase();
      result = result.filter(
        (m) =>
          m.type.toLowerCase().includes(q) ||
          m.description.toLowerCase().includes(q) ||
          m.category.toLowerCase().includes(q)
      );
    }
    return result;
  }, [modules, category, search]);

  return (
    <div className="container mx-auto max-w-7xl px-4 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">Module Catalog</h1>
        <p className="text-muted-foreground">
          Explore all {modules.length} neural network modules. Click any module
          to view its documentation, source code, and run it interactively.
        </p>
      </div>

      {isLoading ? (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {Array.from({ length: 9 }).map((_, i) => (
            <Skeleton key={i} className="h-48 rounded-lg" />
          ))}
        </div>
      ) : error ? (
        <div className="text-center py-12">
          <p className="text-destructive text-lg mb-2">Failed to load modules</p>
          <p className="text-muted-foreground text-sm">
            Make sure the backend server is running on{" "}
            {process.env.NEXT_PUBLIC_API_URL || "http://localhost:8080"}
          </p>
        </div>
      ) : (
        <>
          <div className="mb-6 space-y-4">
            <div className="relative max-w-sm">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search modules..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="pl-9"
              />
            </div>
            <ModuleCategoryFilter
              selected={category}
              onSelect={setCategory}
              categories={categories}
            />
          </div>
          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
            {filtered.map((module) => (
              <ModuleCard key={module.type} module={module} />
            ))}
          </div>
          {filtered.length === 0 && (
            <p className="text-center text-muted-foreground py-12">
              No modules found{search ? ` for "${search}"` : " in this category"}.
            </p>
          )}
        </>
      )}
    </div>
  );
}
