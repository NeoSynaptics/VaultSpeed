import * as FileSystem from "expo-file-system/legacy";
import AsyncStorage from "@react-native-async-storage/async-storage";

const DEFAULT_API_URL = process.env.EXPO_PUBLIC_API_URL ?? "http://localhost:8000";
export const API_URL_KEY = "api_url";

export async function getApiUrl(): Promise<string> {
  const stored = await AsyncStorage.getItem(API_URL_KEY);
  return stored?.trim() || DEFAULT_API_URL;
}

export interface AnalysisStats {
  avg_kmh: number;
  peak_kmh: number;
  delta_kmh: number | null;
  pole_plant_frame: number;
  total_frames: number;
  fps: number;
  trimmed_seconds: number;
  original_frames: number;
}

export interface AnalysisResult {
  stats: AnalysisStats;
  annotatedVideoUri: string; // local file URI after download
}

export async function analyzeVideo(
  localVideoUri: string,
  runwayMeters: number,
  athleteId: string = "default",
  onProgress?: (pct: number) => void,
  source: "camera" | "library" = "camera"
): Promise<AnalysisResult> {
  onProgress?.(5);

  // Build form data
  const formData = new FormData();
  formData.append("video", {
    uri: localVideoUri,
    name: "run.mp4",
    type: "video/mp4",
  } as unknown as Blob);
  formData.append("runway_meters", String(runwayMeters));
  formData.append("athlete_id", athleteId);
  formData.append("source", source);

  onProgress?.(10);

  const apiUrl = await getApiUrl();

  // ── Phase 1: upload + server processing via XHR ─────────────────────────
  // XHR gives real upload progress (10 → 60%).
  // After upload completes the server processes; a slow crawl covers that gap.
  const { status: httpStatus, body: responseText } = await new Promise<{
    status: number;
    body: string;
  }>((resolve, reject) => {
    const xhr = new XMLHttpRequest();

    // Real upload progress: 10% → 60%
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onProgress) {
        onProgress(Math.round(10 + (e.loaded / e.total) * 50));
      }
    };

    // After upload finishes, crawl 60 → 68% while server processes
    let crawlPct = 60;
    let crawlTimer: ReturnType<typeof setInterval> | null = null;
    xhr.upload.onloadend = () => {
      onProgress?.(60);
      crawlTimer = setInterval(() => {
        crawlPct = Math.min(crawlPct + 0.15, 68);
        onProgress?.(Math.round(crawlPct));
      }, 500);
    };

    xhr.onload = () => {
      if (crawlTimer) clearInterval(crawlTimer);
      resolve({ status: xhr.status, body: xhr.responseText });
    };
    xhr.onerror = () => {
      if (crawlTimer) clearInterval(crawlTimer);
      reject(new Error("Network request failed — check tunnel URL in Settings"));
    };
    xhr.ontimeout = () => {
      if (crawlTimer) clearInterval(crawlTimer);
      reject(new Error("Request timed out — server may still be processing"));
    };

    xhr.open("POST", `${apiUrl}/analyze`);
    xhr.setRequestHeader("bypass-tunnel-reminder", "true");
    xhr.send(formData);
  });

  if (httpStatus < 200 || httpStatus >= 300) {
    throw new Error(`Server error ${httpStatus}: ${responseText}`);
  }

  onProgress?.(70);

  const json = JSON.parse(responseText);
  const stats: AnalysisStats = json.stats;
  const videoFilename: string = json.video_filename;

  // Download the annotated video as a binary file — keeps the /analyze JSON
  // response tiny so it passes through any tunnel without size limits.
  const outputUri = FileSystem.cacheDirectory + `annotated_${Date.now()}.mp4`;
  const downloadResult = await FileSystem.downloadAsync(
    `${apiUrl}/video/${encodeURIComponent(videoFilename)}`,
    outputUri,
    { headers: { "bypass-tunnel-reminder": "true" } },
  );

  if (downloadResult.status !== 200) {
    throw new Error(`Video download failed: HTTP ${downloadResult.status}`);
  }

  onProgress?.(100);

  return { stats, annotatedVideoUri: downloadResult.uri };
}

export async function healthCheck(): Promise<boolean> {
  try {
    const apiUrl = await getApiUrl();
    const r = await fetch(`${apiUrl}/health`, {
      method: "GET",
      headers: { "bypass-tunnel-reminder": "true" },
    });
    return r.ok;
  } catch {
    return false;
  }
}
