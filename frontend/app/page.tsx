import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { HealthIndicator } from "@/components/health-indicator";
import { Blocks, GitFork, BookOpen } from "lucide-react";

const features = [
  {
    icon: Blocks,
    title: "Module Explorer",
    description:
      "Browse 18 neural network modules, from embeddings to attention to MoE. View source code, read documentation, and run each module interactively.",
    href: "/modules",
    cta: "Browse Modules",
  },
  {
    icon: GitFork,
    title: "Graph Editor",
    description:
      "Drag and connect modules into computation graphs. Wire ports, configure parameters, and execute end-to-end pipelines visually.",
    href: "/graph",
    cta: "Open Editor",
  },
  {
    icon: BookOpen,
    title: "Learn",
    description:
      "Step-by-step tutorials from tokenization to training. Interactive demos, mathematical foundations, and concept explanations.",
    href: "/docs",
    cta: "Start Learning",
  },
];

export default function HomePage() {
  return (
    <div className="container mx-auto max-w-7xl px-4">
      {/* Hero */}
      <section className="flex flex-col items-center text-center py-20 md:py-32 gap-6">
        <h1 className="text-4xl md:text-6xl font-bold tracking-tight">
          LLMs{" "}
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-purple-500 via-blue-500 to-purple-500 animate-gradient">
            Unlocked
          </span>
        </h1>
        <p className="text-lg md:text-xl text-muted-foreground max-w-2xl">
          Making AI accessible. Explore, modify, and connect LLM components
          visually. Built from scratch in C++.
        </p>
        <div className="flex gap-3 mt-2">
          <Button asChild size="lg">
            <Link href="/modules">Explore Modules</Link>
          </Button>
          <Button asChild size="lg" variant="outline">
            <Link href="/graph">Open Graph Editor</Link>
          </Button>
        </div>
        <HealthIndicator />
      </section>

      {/* Feature Cards */}
      <section className="grid gap-6 md:grid-cols-3 pb-20">
        {features.map((feature) => (
          <Link key={feature.href} href={feature.href}>
            <Card className="h-full hover:border-primary/50 transition-colors group">
              <CardHeader>
                <feature.icon className="h-8 w-8 mb-2 text-purple-500" />
                <CardTitle>{feature.title}</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground mb-4">
                  {feature.description}
                </p>
                <span className="text-sm font-medium text-primary group-hover:underline">
                  {feature.cta} &rarr;
                </span>
              </CardContent>
            </Card>
          </Link>
        ))}
      </section>
    </div>
  );
}
