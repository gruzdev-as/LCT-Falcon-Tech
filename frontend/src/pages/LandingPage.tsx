import { Hero } from "../components/Hero";

export function LandingPage() {
  return (
    <div className="min-h-dvh bg-surface-950">
      <Hero />
      <main className="mx-auto max-w-5xl px-6 py-12 sm:py-16">
        {/* SearchWidget lands here */}
      </main>
    </div>
  );
}
