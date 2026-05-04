// Hook + types for the command palette. Kept separate from CommandProvider
// so HMR fast-refresh works in the provider file.

import { useContext } from "react";
import { CommandCtx, type CommandCtxValue } from "@/providers/command-context";

export type Command = {
  id:        string;
  label:     string;
  group?:    string;
  hint?:     string;
  shortcut?: string;
  run:       () => void;
};

export function useCommands(): CommandCtxValue {
  const ctx = useContext(CommandCtx);
  if (!ctx) throw new Error("useCommands must be used inside <CommandProvider>");
  return ctx;
}
