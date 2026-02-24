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
  onProgress?: (pct: number) => void
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

  onProgress?.(10);

  const apiUrl = await getApiUrl();
  const response = await fetch(`${apiUrl}/analyze`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Server error ${response.status}: ${text}`);
  }

  onProgress?.(70);

  const json = await response.json();
  const stats: AnalysisStats = json.stats;
  const videoB64: string = json.video_b64;

  // Write annotated video to local cache
  const outputUri = FileSystem.cacheDirectory + `annotated_${Date.now()}.mp4`;
  await FileSystem.writeAsStringAsync(outputUri, videoB64, {
    encoding: FileSystem.EncodingType.Base64,
  });

  onProgress?.(100);

  return { stats, annotatedVideoUri: outputUri };
}

export async function healthCheck(): Promise<boolean> {
  try {
    const apiUrl = await getApiUrl();
    const r = await fetch(`${apiUrl}/health`, { method: "GET" });
    return r.ok;
  } catch {
    return false;
  }
}
