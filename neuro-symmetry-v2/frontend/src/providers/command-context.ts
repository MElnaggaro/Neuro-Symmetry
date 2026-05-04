// Command palette context. Extracted into its own module so the CommandProvider
// component file passes react-refresh/only-export-components.

import { createContext } from "react";
import type { Command } from "@/hooks/useCommands";

export type CommandCtxValue = {
  commands: ReadonlyArray<Command>;
  register: (cmd: Command) => () => void;
  open:     boolean;
  setOpen:  (v: boolean) => void;
  toggle:   () => void;
};

export const CommandCtx = createContext<CommandCtxValue | null>(null);
