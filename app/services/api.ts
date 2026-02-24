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

  // Linear crawl: +0.25% every 500ms → reaches 90% in ~5.5 min.
  // Caps at 90 and stays there until the server responds.
  // Cleared the instant the response arrives, then snaps to real values.
  let crawlPct = 10;
  const crawlTimer = onProgress
    ? setInterval(() => {
        crawlPct = Math.min(crawlPct + 0.25, 90);
        onProgress(Math.round(crawlPct));
      }, 500)
    : null;

  const apiUrl = await getApiUrl();
  let response: Response;
  try {
    response = await fetch(`${apiUrl}/analyze`, {
      method: "POST",
      headers: { "bypass-tunnel-reminder": "true" },
      body: formData,
    });
  } finally {
    if (crawlTimer !== null) clearInterval(crawlTimer);
  }

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Server error ${response.status}: ${text}`);
  }

  onProgress?.(70);

  const json = await response.json();
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
