import { useCallback, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import type { Command } from "@/hooks/useCommands";
import { CommandCtx, type CommandCtxValue } from "./command-context";

export function CommandProvider({ children }: { children: ReactNode }) {
  const [commands, setCommands] = useState<Command[]>([]);
  const [open, setOpen] = useState(false);

  // Stable register: returns an unregister function. Using a ref to dedupe
  // across StrictMode double-invokes.
  const idsRef = useRef(new Set<string>());

  const register = useCallback((cmd: Command) => {
    setCommands((prev) => {
      if (idsRef.current.has(cmd.id)) {
        return prev.map((c) => (c.id === cmd.id ? cmd : c));
      }
      idsRef.current.add(cmd.id);
      return [...prev, cmd];
    });
    return () => {
      idsRef.current.delete(cmd.id);
      setCommands((prev) => prev.filter((c) => c.id !== cmd.id));
    };
  }, []);

  const toggle = useCallback(() => setOpen((v) => !v), []);

  const value = useMemo<CommandCtxValue>(
    () => ({ commands, register, open, setOpen, toggle }),
    [commands, register, open, toggle],
  );

  return <CommandCtx.Provider value={value}>{children}</CommandCtx.Provider>;
}
