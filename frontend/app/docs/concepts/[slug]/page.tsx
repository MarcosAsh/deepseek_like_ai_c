import { notFound } from "next/navigation";
import { getConceptBySlug, CONCEPTS } from "@/lib/docs-content";

export function generateStaticParams() {
  return CONCEPTS.map((c) => ({ slug: c.slug }));
}

export default async function ConceptPage({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  const doc = getConceptBySlug(slug);
  if (!doc) notFound();

  return (
    <article className="max-w-3xl mx-auto px-6 py-8">
      <div
        className="prose dark:prose-invert max-w-none prose-headings:scroll-mt-20 prose-pre:bg-muted prose-pre:text-foreground prose-code:before:content-none prose-code:after:content-none"
        dangerouslySetInnerHTML={{ __html: doc.content }}
      />
    </article>
  );
}
