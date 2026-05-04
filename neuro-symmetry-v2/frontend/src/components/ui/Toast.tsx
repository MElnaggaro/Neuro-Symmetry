import { Toaster } from "sonner";

// Pre-themed sonner Toaster: dark, glass, bottom-right.
// Helpers (toastInfo / toastSuccess / etc.) live in @/utils/toast.
export function NSToaster() {
  return (
    <Toaster
      theme="dark"
      position="bottom-right"
      richColors
      closeButton
      toastOptions={{
        className: "glass-card !text-slate-200 !border !border-neu-border",
        style: { fontSize: "12px", letterSpacing: "0.02em" },
      }}
    />
  );
}
