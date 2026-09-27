import { Hero } from "../components/Hero";
import { SearchWidget } from "../components/SearchWidget";

export function LandingPage() {
  return (
    <div className="min-h-dvh bg-surface-950">
      <Hero />
      <main className="mx-auto max-w-7xl px-6 py-12 sm:py-16">
        <SearchWidget />
      </main>
    </div>
  );
}
