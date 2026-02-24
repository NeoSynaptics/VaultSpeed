import React, { useState, useEffect } from "react";
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  KeyboardAvoidingView,
  Platform,
} from "react-native";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { healthCheck, getApiUrl, API_URL_KEY } from "../services/api";

interface Props {
  onDone: () => void;
}

type ConnStatus = "checking" | "ok" | "fail";

export default function SetupScreen({ onDone }: Props) {
  const [apiUrl, setApiUrl] = useState("");
  const [connStatus, setConnStatus] = useState<ConnStatus>("checking");

  useEffect(() => {
    getApiUrl().then(setApiUrl);
  }, []);

  useEffect(() => {
    if (!apiUrl) return;
    setConnStatus("checking");
    const t = setTimeout(async () => {
      const ok = await healthCheck();
      setConnStatus(ok ? "ok" : "fail");
    }, 400);
    return () => clearTimeout(t);
  }, [apiUrl]);

  const save = async () => {
    await AsyncStorage.setItem(API_URL_KEY, apiUrl.trim());
    onDone();
  };

  const statusColor =
    connStatus === "ok" ? "#4cde80" : connStatus === "fail" ? "#e5242f" : "#888";
  const statusLabel =
    connStatus === "ok" ? "Connected" : connStatus === "fail" ? "No connection" : "Checking…";

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === "ios" ? "padding" : undefined}
    >
      <Text style={styles.title}>VaultSpeed V1</Text>
      <Text style={styles.subtitle}>Settings</Text>

      <View style={styles.apiRow}>
        <Text style={styles.label}>Backend URL</Text>
        <View style={styles.connRow}>
          <View style={[styles.connDot, { backgroundColor: statusColor }]} />
          <Text style={[styles.connLabel, { color: statusColor }]}>{statusLabel}</Text>
        </View>
      </View>
      <TextInput
        style={styles.input}
        value={apiUrl}
        onChangeText={setApiUrl}
        autoCapitalize="none"
        autoCorrect={false}
        keyboardType="url"
        placeholder="https://xxxx.ngrok-free.app"
        placeholderTextColor="#555"
      />

      <TouchableOpacity style={styles.button} onPress={save}>
        <Text style={styles.buttonText}>Save</Text>
      </TouchableOpacity>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#0a0a0a",
    justifyContent: "center",
    padding: 32,
  },
  title: {
    fontSize: 36,
    fontWeight: "800",
    color: "#fff",
    marginBottom: 4,
  },
  subtitle: {
    fontSize: 16,
    color: "#666",
    marginBottom: 40,
  },
  label: {
    fontSize: 16,
    color: "#ccc",
    marginBottom: 6,
    fontWeight: "600",
  },
  input: {
    backgroundColor: "#1a1a1a",
    borderRadius: 10,
    padding: 16,
    fontSize: 18,
    color: "#fff",
    borderWidth: 1,
    borderColor: "#333",
    marginBottom: 32,
  },
  button: {
    backgroundColor: "#2563eb",
    borderRadius: 12,
    padding: 18,
    alignItems: "center",
  },
  buttonText: {
    color: "#fff",
    fontSize: 18,
    fontWeight: "700",
  },
  apiRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 6,
  },
  connRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
  },
  connDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  connLabel: {
    fontSize: 12,
    fontWeight: "600",
  },
});
