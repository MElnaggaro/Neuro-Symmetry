import { Command } from "cmdk";
import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useMemo } from "react";
import { useCommands } from "@/hooks/useCommands";
import { useHotkey } from "@/hooks/useHotkey";

export function CommandPalette() {
  const { commands, open, setOpen, toggle } = useCommands();

  useHotkey("mod+k", toggle);
  useHotkey("esc",   () => setOpen(false));

  // Close palette on route change (hash update) so stale state doesn't linger.
  useEffect(() => {
    const onHash = () => setOpen(false);
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, [setOpen]);

  const groups = useMemo(() => {
    const m = new Map<string, typeof commands[number][]>();
    for (const c of commands) {
      const g = c.group ?? "Actions";
      if (!m.has(g)) m.set(g, []);
      m.get(g)!.push(c);
    }
    return Array.from(m.entries());
  }, [commands]);

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          key="cmdk-overlay"
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
          transition={{ duration: 0.12 }}
          className="fixed inset-0 z-[100] flex items-start justify-center pt-[15vh] bg-black/55 backdrop-blur-sm"
          onClick={() => setOpen(false)}
          role="dialog"
          aria-modal="true"
          aria-label="Command palette"
        >
          <motion.div
            key="cmdk-panel"
            initial={{ opacity: 0, y: -8, scale: 0.98 }}
            animate={{ opacity: 1, y: 0,  scale: 1    }}
            exit={{    opacity: 0, y: -8, scale: 0.98 }}
            transition={{ duration: 0.16, ease: [0.16, 1, 0.3, 1] }}
            className="glass-card glass-inset w-[min(640px,92vw)] rounded-2xl border border-neu-border overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            <Command label="Neuro-Symmetry command palette" loop>
              <div className="px-4 pt-3 pb-2 border-b border-neu-border">
                <Command.Input
                  placeholder="Search commands…"
                  className="w-full bg-transparent outline-none text-slate-200 text-body placeholder:text-slate-600"
                  autoFocus
                />
              </div>

              <Command.List className="max-h-[50vh] overflow-y-auto p-2">
                <Command.Empty className="px-4 py-6 text-center text-label text-slate-500">
                  No matching commands
                </Command.Empty>

                {groups.map(([group, list]) => (
                  <Command.Group
                    key={group}
                    heading={group}
                    className="text-micro tracking-cyber uppercase text-slate-600 px-2 pt-2 pb-1
                               [&_[cmdk-group-heading]]:px-2 [&_[cmdk-group-heading]]:py-1"
                  >
                    {list.map((c) => (
                      <Command.Item
                        key={c.id}
                        value={`${c.label} ${c.hint ?? ""}`}
                        onSelect={() => { c.run(); setOpen(false); }}
                        className="flex items-center justify-between px-3 py-2 rounded-lg cursor-pointer
                                   text-body text-slate-300
                                   data-[selected=true]:bg-cyan-900/15 data-[selected=true]:text-accent-cyan
                                   transition-colors"
                      >
                        <span>{c.label}</span>
                        {c.shortcut && (
                          <kbd className="text-micro font-mono text-slate-500 bg-bg-panel px-1.5 py-0.5 rounded border border-neu-border">
                            {c.shortcut}
                          </kbd>
                        )}
                      </Command.Item>
                    ))}
                  </Command.Group>
                ))}
              </Command.List>

              <div className="px-4 py-2 border-t border-neu-border flex justify-between text-micro text-slate-600">
                <span>↑↓ navigate · ↵ select</span>
                <span>esc close · ⌘K toggle</span>
              </div>
            </Command>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
