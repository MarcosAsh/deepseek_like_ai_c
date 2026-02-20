import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { CONCEPTS, TUTORIALS } from "@/lib/docs-content";
import { BookOpen, GraduationCap, Code2 } from "lucide-react";

export default function DocsPage() {
  return (
    <div className="max-w-4xl mx-auto px-6 py-8">
      <h1 className="text-3xl font-bold mb-2">Documentation</h1>
      <p className="text-muted-foreground mb-8">
        Learn how Large Language Models work from the ground up. Explore
        concepts, follow tutorials, and reference the API.
      </p>

      {/* Concepts */}
      <section className="mb-10">
        <div className="flex items-center gap-2 mb-4">
          <BookOpen className="h-5 w-5 text-purple-500" />
          <h2 className="text-xl font-semibold">Concepts</h2>
        </div>
        <div className="grid gap-3 md:grid-cols-2">
          {CONCEPTS.map((doc) => (
            <Link key={doc.slug} href={`/docs/concepts/${doc.slug}`}>
              <Card className="h-full hover:border-primary/50 transition-colors">
                <CardHeader className="pb-2">
                  <CardTitle className="text-base">{doc.title}</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground">
                    {doc.description}
                  </p>
                </CardContent>
              </Card>
            </Link>
          ))}
        </div>
      </section>

      {/* Tutorials */}
      <section className="mb-10">
        <div className="flex items-center gap-2 mb-4">
          <GraduationCap className="h-5 w-5 text-blue-500" />
          <h2 className="text-xl font-semibold">Tutorials</h2>
        </div>
        <div className="space-y-2">
          {TUTORIALS.map((doc, i) => (
            <Link key={doc.slug} href={`/docs/tutorials/${doc.slug}`}>
              <Card className="hover:border-primary/50 transition-colors">
                <CardContent className="py-3 px-4 flex items-center gap-4">
                  <span className="text-2xl font-bold text-muted-foreground w-8">
                    {i + 1}
                  </span>
                  <div>
                    <p className="font-medium">{doc.title}</p>
                    <p className="text-sm text-muted-foreground">
                      {doc.description}
                    </p>
                  </div>
                </CardContent>
              </Card>
            </Link>
          ))}
        </div>
      </section>

      {/* API Reference */}
      <section>
        <div className="flex items-center gap-2 mb-4">
          <Code2 className="h-5 w-5 text-green-500" />
          <h2 className="text-xl font-semibold">API Reference</h2>
        </div>
        <Link href="/docs/api">
          <Card className="hover:border-primary/50 transition-colors">
            <CardContent className="py-4 px-4">
              <p className="font-medium">REST API Documentation</p>
              <p className="text-sm text-muted-foreground">
                Complete reference for all backend API endpoints: health, modules,
                execute, presets.
              </p>
            </CardContent>
          </Card>
        </Link>
      </section>
    </div>
  );
}
