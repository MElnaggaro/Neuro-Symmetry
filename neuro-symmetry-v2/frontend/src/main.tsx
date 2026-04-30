import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import App from "./App.tsx";
import { FaceTrackingProvider } from "@/providers/FaceTrackingProvider";
import { DialogProvider }       from "@/components/ui/DialogManager";

const root = document.getElementById("root");
if (!root) throw new Error("Root element #root not found.");

createRoot(root).render(
  <StrictMode>
    <FaceTrackingProvider>
      <DialogProvider>
        <App />
      </DialogProvider>
    </FaceTrackingProvider>
  </StrictMode>,
);
