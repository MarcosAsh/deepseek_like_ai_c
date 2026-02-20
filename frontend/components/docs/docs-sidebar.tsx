"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { CONCEPTS, TUTORIALS } from "@/lib/docs-content";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";
import { Button } from "@/components/ui/button";
import { BookOpen, GraduationCap, Menu } from "lucide-react";
import { useState } from "react";

function SidebarContent({ onNavigate }: { onNavigate?: () => void }) {
  const pathname = usePathname();

  return (
    <div className="p-4 space-y-6">
      <div>
        <Link
          href="/docs"
          onClick={onNavigate}
          className={cn(
            "text-sm font-semibold mb-3 block",
            pathname === "/docs"
              ? "text-primary"
              : "text-muted-foreground hover:text-foreground"
          )}
        >
          Documentation Home
        </Link>
      </div>

      <div>
        <div className="flex items-center gap-2 mb-2">
          <BookOpen className="h-4 w-4 text-muted-foreground" />
          <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Concepts
          </h3>
        </div>
        <nav className="space-y-0.5">
          {CONCEPTS.map((doc) => (
            <Link
              key={doc.slug}
              href={`/docs/concepts/${doc.slug}`}
              onClick={onNavigate}
              className={cn(
                "block px-2 py-1.5 rounded text-sm transition-colors",
                pathname === `/docs/concepts/${doc.slug}`
                  ? "bg-accent text-accent-foreground font-medium"
                  : "text-muted-foreground hover:text-foreground hover:bg-accent/50"
              )}
            >
              {doc.title}
            </Link>
          ))}
        </nav>
      </div>

      <div>
        <div className="flex items-center gap-2 mb-2">
          <GraduationCap className="h-4 w-4 text-muted-foreground" />
          <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Tutorials
          </h3>
        </div>
        <nav className="space-y-0.5">
          {TUTORIALS.map((doc, i) => (
            <Link
              key={doc.slug}
              href={`/docs/tutorials/${doc.slug}`}
              onClick={onNavigate}
              className={cn(
                "block px-2 py-1.5 rounded text-sm transition-colors",
                pathname === `/docs/tutorials/${doc.slug}`
                  ? "bg-accent text-accent-foreground font-medium"
                  : "text-muted-foreground hover:text-foreground hover:bg-accent/50"
              )}
            >
              <span className="text-muted-foreground mr-1">
                {i + 1}.
              </span>
              {doc.title}
            </Link>
          ))}
        </nav>
      </div>

      <div>
        <Link
          href="/docs/api"
          onClick={onNavigate}
          className={cn(
            "block px-2 py-1.5 rounded text-sm transition-colors",
            pathname === "/docs/api"
              ? "bg-accent text-accent-foreground font-medium"
              : "text-muted-foreground hover:text-foreground hover:bg-accent/50"
          )}
        >
          API Reference
        </Link>
      </div>
    </div>
  );
}

export function DocsSidebar() {
  const [drawerOpen, setDrawerOpen] = useState(false);

  return (
    <>
      {/* Desktop sidebar */}
      <aside className="w-64 border-r bg-muted/30 hidden md:block">
        <ScrollArea className="h-[calc(100vh-3.5rem)]">
          <SidebarContent />
        </ScrollArea>
      </aside>

      {/* Mobile floating button + drawer */}
      <div className="md:hidden fixed bottom-4 left-4 z-40">
        <Sheet open={drawerOpen} onOpenChange={setDrawerOpen}>
          <SheetTrigger asChild>
            <Button size="icon" className="rounded-full shadow-lg h-12 w-12">
              <Menu className="h-5 w-5" />
            </Button>
          </SheetTrigger>
          <SheetContent side="left" className="w-72 p-0">
            <ScrollArea className="h-full pt-8">
              <SidebarContent onNavigate={() => setDrawerOpen(false)} />
            </ScrollArea>
          </SheetContent>
        </Sheet>
      </div>
    </>
  );
}
