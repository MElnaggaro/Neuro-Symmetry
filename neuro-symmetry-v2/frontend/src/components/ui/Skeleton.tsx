// Shimmer skeleton primitives — used while WebSocket connects or first frame
// hasn't arrived yet. Plain Tailwind + .skeleton-shimmer utility from index.css.

type SkeletonProps = {
  className?: string;
  rounded?:   string;
};

export function Skeleton({ className = "", rounded = "rounded-md" }: SkeletonProps) {
  return <div className={`skeleton-shimmer ${rounded} ${className}`} aria-hidden="true" />;
}

export function SkeletonGauge() {
  return (
    <div className="flex items-center gap-4 p-4" aria-busy="true" aria-label="Loading symmetry gauge">
      <Skeleton className="w-[140px] h-[140px]" rounded="rounded-full" />
      <div className="flex-1 flex flex-col gap-2">
        <Skeleton className="h-4 w-24" />
        <Skeleton className="h-3 w-32" />
        <Skeleton className="h-3 w-20" />
      </div>
    </div>
  );
}

export function SkeletonGraph() {
  return (
    <div className="flex flex-col gap-2" aria-busy="true" aria-label="Loading score graph">
      <div className="flex justify-between">
        <Skeleton className="h-3 w-20" />
        <Skeleton className="h-3 w-16" />
      </div>
      <Skeleton className="h-32 w-full" />
    </div>
  );
}

export function SkeletonXAI() {
  return (
    <div className="flex flex-col gap-2" aria-busy="true" aria-label="Loading feature contributions">
      {Array.from({ length: 5 }).map((_, i) => (
        <div key={i} className="flex items-center gap-3">
          <Skeleton className="h-2.5 w-2.5" rounded="rounded-full" />
          <Skeleton className="h-3 flex-1" />
          <Skeleton className="h-3 w-12" />
        </div>
      ))}
    </div>
  );
}
