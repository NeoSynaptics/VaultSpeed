import React, { useEffect, useState } from "react";
import { StatusBar } from "expo-status-bar";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { AnalysisResult } from "./services/api";
import SetupScreen from "./screens/SetupScreen";
import CameraScreen from "./screens/CameraScreen";
import ResultScreen from "./screens/ResultScreen";

type Screen = "setup" | "camera" | "result";

export default function App() {
  const [screen, setScreen] = useState<Screen>("setup");
  const [result, setResult] = useState<AnalysisResult | null>(null);

  useEffect(() => {
    AsyncStorage.getItem("runway_meters").then((v) => {
      if (v) setScreen("camera");
    });
  }, []);

  const handleResult = (r: AnalysisResult) => {
    setResult(r);
    setScreen("result");
  };

  return (
    <>
      <StatusBar style="light" />
      {screen === "setup" && (
        <SetupScreen onDone={() => setScreen("camera")} />
      )}
      {screen === "camera" && (
        <CameraScreen onResult={handleResult} onSettings={() => setScreen("setup")} />
      )}
      {screen === "result" && result && (
        <ResultScreen
          result={result}
          onRecordAgain={() => setScreen("camera")}
        />
      )}
    </>
  );
}
