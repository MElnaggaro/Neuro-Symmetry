// Standardised toast helpers. Kept in a non-component file so HMR fast-refresh
// stays happy in the file that exports <NSToaster>.

import { toast } from "sonner";
import type { ReactNode } from "react";

const opts = { duration: 3200 };

export const toastInfo    = (msg: ReactNode) => toast(msg, opts);
export const toastSuccess = (msg: ReactNode) => toast.success(msg, opts);
export const toastError   = (msg: ReactNode) => toast.error(msg, { ...opts, duration: 5000 });
export const toastWarning = (msg: ReactNode) => toast.warning(msg, opts);

export const toastCalibration = {
  start:  () => toast.loading("Calibrating personal baseline…", { id: "ns-calib" }),
  done:   () => toast.success("Baseline ready", { id: "ns-calib", duration: 2400 }),
  cancel: () => toast.dismiss("ns-calib"),
};

export { toast };
