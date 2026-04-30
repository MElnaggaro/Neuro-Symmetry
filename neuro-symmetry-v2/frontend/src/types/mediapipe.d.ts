// Minimal ambient declarations for @mediapipe packages (no official @types exist).

declare module "@mediapipe/face_mesh" {
  export interface NormalizedLandmark {
    x:           number;
    y:           number;
    z:           number;
    visibility?: number;
  }

  export type NormalizedLandmarkList = NormalizedLandmark[];

  export interface FaceMeshResults {
    multiFaceLandmarks?: NormalizedLandmark[][];
    image: HTMLVideoElement | HTMLImageElement | HTMLCanvasElement;
  }

  export interface FaceMeshOptions {
    maxNumFaces?:            number;
    refineLandmarks?:        boolean;
    minDetectionConfidence?: number;
    minTrackingConfidence?:  number;
  }

  export class FaceMesh {
    constructor(config?: { locateFile?: (path: string) => string });
    setOptions(options: FaceMeshOptions): void;
    onResults(callback: (results: FaceMeshResults) => void): void;
    send(inputs: { image: HTMLVideoElement | HTMLImageElement | HTMLCanvasElement }): Promise<void>;
    close(): void;
  }
}

declare module "@mediapipe/camera_utils" {
  export interface CameraOptions {
    onFrame:     () => Promise<void> | void;
    width?:      number;
    height?:     number;
    facingMode?: string;
  }

  export class Camera {
    constructor(video: HTMLVideoElement, options: CameraOptions);
    start(): void;
    stop(): void;
  }
}