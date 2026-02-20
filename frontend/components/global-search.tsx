"use client";

import { useEffect, useState, useMemo } from "react";
import { useRouter } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { fetchModules } from "@/lib/api";
import { CONCEPTS, TUTORIALS } from "@/lib/docs-content";
import {
  CommandDialog,
  CommandInput,
  CommandList,
  CommandEmpty,
  CommandGroup,
  CommandItem,
  CommandSeparator,
} from "@/components/ui/command";
import {
  Blocks,
  BookOpen,
  GraduationCap,
  GitFork,
  FileText,
} from "lucide-react";

export function GlobalSearch() {
  const [open, setOpen] = useState(false);
  const router = useRouter();

  const { data } = useQuery({
    queryKey: ["modules"],
    queryFn: fetchModules,
    enabled: open,
  });

  const modules = data?.modules ?? [];

  // Cmd+K shortcut
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        setOpen((o) => !o);
      }
    }
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, []);

  function navigate(path: string) {
    router.push(path);
    setOpen(false);
  }

  return (
    <CommandDialog
      open={open}
      onOpenChange={setOpen}
      title="Search"
      description="Search modules, concepts, tutorials, and pages"
    >
      <CommandInput placeholder="Search modules, docs, tutorials..." />
      <CommandList>
        <CommandEmpty>No results found.</CommandEmpty>

        <CommandGroup heading="Pages">
          <CommandItem onSelect={() => navigate("/modules")}>
            <Blocks className="mr-2 h-4 w-4" />
            Module Catalog
          </CommandItem>
          <CommandItem onSelect={() => navigate("/graph")}>
            <GitFork className="mr-2 h-4 w-4" />
            Graph Editor
          </CommandItem>
          <CommandItem onSelect={() => navigate("/docs")}>
            <BookOpen className="mr-2 h-4 w-4" />
            Documentation
          </CommandItem>
        </CommandGroup>

        <CommandSeparator />

        {modules.length > 0 && (
          <CommandGroup heading="Modules">
            {modules.map((mod) => (
              <CommandItem
                key={mod.type}
                onSelect={() => navigate(`/modules/${mod.type}`)}
                value={`${mod.type} ${mod.description} ${mod.category}`}
              >
                <Blocks className="mr-2 h-4 w-4" />
                <div className="flex flex-col">
                  <span className="text-sm">{mod.type}</span>
                  <span className="text-xs text-muted-foreground truncate max-w-[300px]">
                    {mod.description}
                  </span>
                </div>
              </CommandItem>
            ))}
          </CommandGroup>
        )}

        <CommandSeparator />

        <CommandGroup heading="Concepts">
          {CONCEPTS.map((concept) => (
            <CommandItem
              key={concept.slug}
              onSelect={() => navigate(`/docs/concepts/${concept.slug}`)}
              value={`${concept.title} ${concept.description}`}
            >
              <BookOpen className="mr-2 h-4 w-4" />
              {concept.title}
            </CommandItem>
          ))}
        </CommandGroup>

        <CommandSeparator />

        <CommandGroup heading="Tutorials">
          {TUTORIALS.map((tutorial) => (
            <CommandItem
              key={tutorial.slug}
              onSelect={() => navigate(`/docs/tutorials/${tutorial.slug}`)}
              value={`${tutorial.title} ${tutorial.description}`}
            >
              <GraduationCap className="mr-2 h-4 w-4" />
              {tutorial.title}
            </CommandItem>
          ))}
        </CommandGroup>
      </CommandList>
    </CommandDialog>
  );
}
