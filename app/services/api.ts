import * as FileSystem from "expo-file-system";

const API_URL = process.env.EXPO_PUBLIC_API_URL ?? "http://localhost:8000";

export interface AnalysisStats {
  avg_kmh: number;
  peak_kmh: number;
  delta_kmh: number | null;
  pole_plant_frame: number;
  total_frames: number;
  fps: number;
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

  const response = await fetch(`${API_URL}/analyze`, {
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
    const r = await fetch(`${API_URL}/health`, { method: "GET" });
    return r.ok;
  } catch {
    return false;
  }
}
