import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import Routed from "./Routed";
import { FaceTrackingProvider } from "@/providers/FaceTrackingProvider";
import { DialogProvider }       from "@/components/ui/DialogManager";
import { NSToaster }            from "@/components/ui/Toast";
import { CommandProvider }      from "@/providers/CommandProvider";
import { CommandPalette }       from "@/components/ui/CommandPalette";
import { OnboardingTour }       from "@/components/ui/OnboardingTour";

const root = document.getElementById("root");
if (!root) throw new Error("Root element #root not found.");

createRoot(root).render(
  <StrictMode>
    <FaceTrackingProvider>
      <DialogProvider>
        <CommandProvider>
          <Routed />
          <CommandPalette />
          <OnboardingTour />
          <NSToaster />
        </CommandProvider>
      </DialogProvider>
    </FaceTrackingProvider>
  </StrictMode>,
);
