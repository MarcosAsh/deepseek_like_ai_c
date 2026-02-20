import { notFound } from "next/navigation";
import { getTutorialBySlug, TUTORIALS } from "@/lib/docs-content";
import Link from "next/link";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";

export function generateStaticParams() {
  return TUTORIALS.map((t) => ({ slug: t.slug }));
}

export default async function TutorialPage({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  const doc = getTutorialBySlug(slug);
  if (!doc) notFound();

  const currentIndex = TUTORIALS.findIndex((t) => t.slug === slug);
  const prev = currentIndex > 0 ? TUTORIALS[currentIndex - 1] : null;
  const next =
    currentIndex < TUTORIALS.length - 1 ? TUTORIALS[currentIndex + 1] : null;

  return (
    <article className="max-w-3xl mx-auto px-6 py-8">
      <div
        className="prose dark:prose-invert max-w-none prose-headings:scroll-mt-20 prose-pre:bg-muted prose-pre:text-foreground prose-code:before:content-none prose-code:after:content-none"
        dangerouslySetInnerHTML={{ __html: doc.content }}
      />

      {/* Navigation */}
      <nav className="flex items-center justify-between mt-12 pt-6 border-t">
        {prev ? (
          <Button variant="ghost" asChild>
            <Link href={`/docs/tutorials/${prev.slug}`}>
              <ChevronLeft className="h-4 w-4 mr-1" />
              {prev.title}
            </Link>
          </Button>
        ) : (
          <div />
        )}
        {next ? (
          <Button variant="ghost" asChild>
            <Link href={`/docs/tutorials/${next.slug}`}>
              {next.title}
              <ChevronRight className="h-4 w-4 ml-1" />
            </Link>
          </Button>
        ) : (
          <div />
        )}
      </nav>
    </article>
  );
}
