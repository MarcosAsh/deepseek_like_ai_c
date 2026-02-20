import { Brain } from "lucide-react";

export function Footer() {
  return (
    <footer className="border-t py-6 md:py-8">
      <div className="container flex flex-col items-center gap-4 px-4 mx-auto max-w-7xl md:flex-row md:justify-between">
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Brain className="h-4 w-4" />
          <span>LLMs Unlocked | Making AI Accessible</span>
        </div>
        <p className="text-sm text-muted-foreground">
          Built from scratch in C++ with ~50k lines of code
        </p>
      </div>
    </footer>
  );
}
