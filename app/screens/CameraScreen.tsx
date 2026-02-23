import React, { useRef, useState, useCallback } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  ActivityIndicator,
  Alert,
} from "react-native";
import { CameraView, useCameraPermissions, useMicrophonePermissions } from "expo-camera";
import * as MediaLibrary from "expo-media-library";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { analyzeVideo, AnalysisResult } from "../services/api";

interface Props {
  onResult: (result: AnalysisResult, originalUri: string) => void;
}

type Status = "idle" | "recording" | "uploading";

export default function CameraScreen({ onResult }: Props) {
  const cameraRef = useRef<CameraView>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [progress, setProgress] = useState(0);
  const [cameraPermission, requestCameraPermission] = useCameraPermissions();
  const [micPermission, requestMicPermission] = useMicrophonePermissions();
  const [mediaPermission, requestMediaPermission] = MediaLibrary.usePermissions();

  const allGranted =
    cameraPermission?.granted && micPermission?.granted && mediaPermission?.granted;

  const requestAll = async () => {
    if (!cameraPermission?.granted) await requestCameraPermission();
    if (!micPermission?.granted) await requestMicPermission();
    if (!mediaPermission?.granted) await requestMediaPermission();
  };

  const startRecording = useCallback(async () => {
    if (!cameraRef.current || status !== "idle") return;
    setStatus("recording");
    try {
      // startRecording resolves when stopRecording is called
      const video = await cameraRef.current.recordAsync({ maxDuration: 30 });
      if (!video?.uri) return;

      setStatus("uploading");
      setProgress(0);

      // Save original to camera roll immediately
      await MediaLibrary.saveToLibraryAsync(video.uri);

      // Upload + analyze
      const runway = parseFloat(
        (await AsyncStorage.getItem("runway_meters")) ?? "40"
      );

      const result = await analyzeVideo(video.uri, runway, "default", setProgress);
      onResult(result, video.uri);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      Alert.alert("Error", msg);
      setStatus("idle");
    }
  }, [status, onResult]);

  const stopRecording = useCallback(() => {
    if (status === "recording") {
      cameraRef.current?.stopRecording();
    }
  }, [status]);

  if (!allGranted) {
    return (
      <View style={styles.permissionContainer}>
        <Text style={styles.permissionTitle}>VaultSpeed needs camera access</Text>
        <TouchableOpacity style={styles.permissionButton} onPress={requestAll}>
          <Text style={styles.permissionButtonText}>Grant Permissions</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <CameraView
        ref={cameraRef}
        style={StyleSheet.absoluteFill}
        facing="back"
        mode="video"
        videoQuality="720p"
      />

      {/* Overlay */}
      <View style={styles.overlay}>
        {status === "idle" && (
          <Text style={styles.hint}>Hold to record</Text>
        )}
        {status === "recording" && (
          <View style={styles.recIndicator}>
            <View style={styles.recDot} />
            <Text style={styles.recText}>REC</Text>
          </View>
        )}
        {status === "uploading" && (
          <View style={styles.uploadingContainer}>
            <ActivityIndicator color="#e5242f" size="large" />
            <Text style={styles.uploadingText}>
              Analyzing... {Math.round(progress)}%
            </Text>
          </View>
        )}
      </View>

      {/* Record button */}
      {status !== "uploading" && (
        <View style={styles.buttonArea}>
          <TouchableOpacity
            style={[styles.recordButton, status === "recording" && styles.recordingActive]}
            onLongPress={startRecording}
            onPressOut={stopRecording}
            delayLongPress={100}
            activeOpacity={0.8}
          >
            <View style={[styles.recordInner, status === "recording" && styles.recordStop]} />
          </TouchableOpacity>
          <Text style={styles.buttonHint}>
            {status === "idle" ? "Hold to record" : "Release to stop"}
          </Text>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#000" },
  overlay: {
    position: "absolute",
    top: 60,
    left: 0,
    right: 0,
    alignItems: "center",
  },
  hint: { color: "rgba(255,255,255,0.6)", fontSize: 14 },
  recIndicator: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "rgba(229,36,47,0.85)",
    borderRadius: 20,
    paddingHorizontal: 14,
    paddingVertical: 6,
  },
  recDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: "#fff",
    marginRight: 6,
  },
  recText: { color: "#fff", fontWeight: "700", fontSize: 14 },
  uploadingContainer: { alignItems: "center", gap: 12 },
  uploadingText: { color: "#fff", fontSize: 16, fontWeight: "600" },
  buttonArea: {
    position: "absolute",
    bottom: 50,
    left: 0,
    right: 0,
    alignItems: "center",
  },
  recordButton: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: "rgba(255,255,255,0.15)",
    borderWidth: 3,
    borderColor: "#fff",
    justifyContent: "center",
    alignItems: "center",
  },
  recordingActive: { borderColor: "#e5242f" },
  recordInner: {
    width: 54,
    height: 54,
    borderRadius: 27,
    backgroundColor: "#e5242f",
  },
  recordStop: {
    borderRadius: 8,
    width: 34,
    height: 34,
  },
  buttonHint: {
    color: "rgba(255,255,255,0.5)",
    fontSize: 12,
    marginTop: 10,
  },
  permissionContainer: {
    flex: 1,
    backgroundColor: "#0a0a0a",
    justifyContent: "center",
    alignItems: "center",
    padding: 32,
  },
  permissionTitle: {
    color: "#fff",
    fontSize: 18,
    fontWeight: "600",
    marginBottom: 24,
    textAlign: "center",
  },
  permissionButton: {
    backgroundColor: "#e5242f",
    borderRadius: 12,
    paddingVertical: 14,
    paddingHorizontal: 28,
  },
  permissionButtonText: { color: "#fff", fontSize: 16, fontWeight: "700" },
});
