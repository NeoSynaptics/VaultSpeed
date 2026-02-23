import React, { useState } from "react";
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

interface Props {
  onDone: () => void;
}

export default function SetupScreen({ onDone }: Props) {
  const [runway, setRunway] = useState("40");

  const save = async () => {
    const meters = parseFloat(runway);
    if (isNaN(meters) || meters < 5 || meters > 100) return;
    await AsyncStorage.setItem("runway_meters", String(meters));
    onDone();
  };

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === "ios" ? "padding" : undefined}
    >
      <Text style={styles.title}>VaultSpeed</Text>
      <Text style={styles.subtitle}>One-time setup</Text>

      <Text style={styles.label}>Runway length (meters)</Text>
      <Text style={styles.hint}>
        Standard pole vault runway is 40–45 m. This calibrates pixel → km/h.
      </Text>
      <TextInput
        style={styles.input}
        value={runway}
        onChangeText={setRunway}
        keyboardType="decimal-pad"
        placeholder="40"
        placeholderTextColor="#555"
      />

      <TouchableOpacity style={styles.button} onPress={save}>
        <Text style={styles.buttonText}>Let's go</Text>
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
  hint: {
    fontSize: 13,
    color: "#555",
    marginBottom: 12,
    lineHeight: 18,
  },
  input: {
    backgroundColor: "#1a1a1a",
    borderRadius: 10,
    padding: 16,
    fontSize: 22,
    color: "#fff",
    borderWidth: 1,
    borderColor: "#333",
    marginBottom: 32,
  },
  button: {
    backgroundColor: "#e5242f",
    borderRadius: 12,
    padding: 18,
    alignItems: "center",
  },
  buttonText: {
    color: "#fff",
    fontSize: 18,
    fontWeight: "700",
  },
});
