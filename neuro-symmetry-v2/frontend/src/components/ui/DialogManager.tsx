/**
 * DialogManager — single global portal for modal UI.
 *
 * Only one dialog can be mounted at a time. Opening a new dialog replaces the
 * current one, which prevents overlapping report/triage popups and avoids
 * z-index collisions with transformed camera panels.
 */

import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
} from "react";
import { createPortal } from "react-dom";
import { AnimatePresence } from "framer-motion";

interface DialogEntry {
  id: string;
  content: React.ReactNode;
}

export interface DialogContextValue {
  openDialog: (id: string, content: React.ReactNode) => void;
  closeDialog: (id?: string) => void;
  activeDialogId: string | null;
  portalRoot: HTMLDivElement;
}

const DialogContext = createContext<DialogContextValue | null>(null);

export function useDialog(): DialogContextValue {
  const ctx = useContext(DialogContext);
  if (!ctx) throw new Error("useDialog must be inside <DialogProvider>");
  return ctx;
}

function renderDialogContent(entry: DialogEntry): React.ReactElement {
  if (React.isValidElement(entry.content)) {
    return React.cloneElement(entry.content, { key: entry.id });
  }
  return <React.Fragment key={entry.id}>{entry.content}</React.Fragment>;
}

export function DialogProvider({ children }: { children: React.ReactNode }) {
  const [activeDialog, setActiveDialog] = useState<DialogEntry | null>(null);
  const rootRef = useRef<HTMLDivElement | null>(null);

  if (!rootRef.current) {
    const root = document.createElement("div");
    root.id = "dialog-root";
    root.style.position = "relative";
    root.style.zIndex = "2147483647";
    document.body.appendChild(root);
    rootRef.current = root;
  }

  useEffect(() => {
    const root = rootRef.current!;
    return () => {
      if (document.body.contains(root)) document.body.removeChild(root);
    };
  }, []);

  const openDialog = useCallback((id: string, content: React.ReactNode) => {
    setActiveDialog({ id, content });
  }, []);

  const closeDialog = useCallback((id?: string) => {
    setActiveDialog((current) => {
      if (!current) return null;
      if (id && current.id !== id) return current;
      return null;
    });
  }, []);

  return (
    <DialogContext.Provider
      value={{
        openDialog,
        closeDialog,
        activeDialogId: activeDialog?.id ?? null,
        portalRoot: rootRef.current!,
      }}
    >
      {children}
      {createPortal(
        <AnimatePresence mode="wait">
          {activeDialog ? renderDialogContent(activeDialog) : null}
        </AnimatePresence>,
        rootRef.current!,
      )}
    </DialogContext.Provider>
  );
}
