import React, { useEffect, useState } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  ScrollView,
} from "react-native";
import { VideoView, useVideoPlayer } from "expo-video";
import * as MediaLibrary from "expo-media-library";
import { AnalysisResult } from "../services/api";

interface Props {
  result: AnalysisResult;
  onRecordAgain: () => void;
}

type SaveStatus = "saving" | "saved" | "error";

export default function ResultScreen({ result, onRecordAgain }: Props) {
  const { stats, annotatedVideoUri } = result;
  const [saveStatus, setSaveStatus] = useState<SaveStatus>("saving");

  const player = useVideoPlayer(annotatedVideoUri, (p) => {
    p.loop = true;
    p.play();
  });

  // Auto-save annotated video to camera roll as soon as screen mounts
  useEffect(() => {
    MediaLibrary.saveToLibraryAsync(annotatedVideoUri)
      .then(() => setSaveStatus("saved"))
      .catch(() => setSaveStatus("error"));
  }, [annotatedVideoUri]);

  const deltaColor =
    stats.delta_kmh === null
      ? "#888"
      : stats.delta_kmh >= 0
      ? "#4cde80"
      : "#e5242f";

  const saveLabel =
    saveStatus === "saved"
      ? "✓ Saved to camera roll"
      : saveStatus === "error"
      ? "⚠ Could not save to camera roll"
      : "Saving to camera roll…";

  const saveLabelColor =
    saveStatus === "saved" ? "#4cde80" : saveStatus === "error" ? "#e5242f" : "#888";

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      {/* Video player */}
      <View style={styles.videoWrapper}>
        <VideoView player={player} style={styles.video} contentFit="contain" />
      </View>

      {/* Stats cards */}
      <View style={styles.statsRow}>
        <StatCard label="Avg speed" value={`${stats.avg_kmh.toFixed(1)}`} unit="km/h" />
        <StatCard label="Peak" value={`${stats.peak_kmh.toFixed(1)}`} unit="km/h" />
        {stats.delta_kmh !== null && (
          <StatCard
            label="vs last run"
            value={`${stats.delta_kmh >= 0 ? "+" : ""}${stats.delta_kmh.toFixed(1)}`}
            unit="km/h"
            valueColor={deltaColor}
          />
        )}
      </View>

      {/* Info row */}
      <Text style={styles.subtext}>
        {stats.trimmed_seconds?.toFixed(1)}s clip (trimmed from{" "}
        {(stats.original_frames / stats.fps).toFixed(1)}s) •{" "}
        plant @{stats.pole_plant_frame}f • {stats.fps.toFixed(0)} fps
      </Text>

      {/* Auto-save status */}
      <Text style={[styles.saveStatus, { color: saveLabelColor }]}>{saveLabel}</Text>

      <TouchableOpacity style={styles.secondaryButton} onPress={onRecordAgain}>
        <Text style={styles.secondaryButtonText}>Record Again</Text>
      </TouchableOpacity>
    </ScrollView>
  );
}

function StatCard({
  label,
  value,
  unit,
  valueColor = "#fff",
}: {
  label: string;
  value: string;
  unit: string;
  valueColor?: string;
}) {
  return (
    <View style={styles.card}>
      <Text style={styles.cardLabel}>{label}</Text>
      <Text style={[styles.cardValue, { color: valueColor }]}>{value}</Text>
      <Text style={styles.cardUnit}>{unit}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#0a0a0a" },
  content: { paddingBottom: 40 },
  videoWrapper: {
    width: "100%",
    aspectRatio: 16 / 9,
    backgroundColor: "#111",
  },
  video: { flex: 1 },
  statsRow: {
    flexDirection: "row",
    justifyContent: "space-around",
    padding: 20,
    gap: 12,
  },
  card: {
    flex: 1,
    backgroundColor: "#1a1a1a",
    borderRadius: 12,
    padding: 14,
    alignItems: "center",
  },
  cardLabel: { color: "#666", fontSize: 11, fontWeight: "600", textTransform: "uppercase" },
  cardValue: { fontSize: 28, fontWeight: "800", marginTop: 4 },
  cardUnit: { color: "#555", fontSize: 12, marginTop: 2 },
  subtext: {
    color: "#444",
    fontSize: 12,
    textAlign: "center",
    marginBottom: 8,
    paddingHorizontal: 16,
  },
  saveStatus: {
    fontSize: 12,
    textAlign: "center",
    marginBottom: 28,
    fontWeight: "600",
  },
  secondaryButton: {
    borderRadius: 12,
    padding: 16,
    alignItems: "center",
    marginHorizontal: 20,
    borderWidth: 1,
    borderColor: "#333",
  },
  secondaryButtonText: { color: "#888", fontSize: 16 },
});
