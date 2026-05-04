import { useHashRoute } from "@/router/api";

type RouteProps = {
  path:     string;
  children: React.ReactNode;
};

export function Route({ path, children }: RouteProps) {
  const current = useHashRoute();
  return current === path ? <>{children}</> : null;
}
