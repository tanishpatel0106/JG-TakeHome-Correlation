import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { useCallback, useEffect, useMemo, useState, useSyncExternalStore } from "react";

import { api } from "./api";

// --- data hooks (one TanStack Query hook per endpoint) -----------------------

export function useMeta() {
  return useQuery({ queryKey: ["meta"], queryFn: ({ signal }) => api.meta(signal), staleTime: Infinity });
}

export function useTimeline() {
  return useQuery({ queryKey: ["timeline"], queryFn: ({ signal }) => api.timeline(signal), staleTime: Infinity });
}

export function useSummary(date: string | null) {
  return useQuery({
    queryKey: ["summary", date],
    queryFn: ({ signal }) => api.summary(date as string, signal),
    enabled: date !== null,
    placeholderData: keepPreviousData,
    staleTime: Infinity,
  });
}

export function useNeighbors(date: string | null, ticker: string | null, k: number) {
  return useQuery({
    queryKey: ["neighbors", date, ticker, k],
    queryFn: ({ signal }) => api.neighbors(date as string, ticker as string, k, signal),
    enabled: date !== null && ticker !== null,
    placeholderData: keepPreviousData,
    staleTime: Infinity,
  });
}

export function useMatrix(date: string | null, tickers: string[]) {
  return useQuery({
    queryKey: ["matrix", date, tickers],
    queryFn: ({ signal }) => api.matrix(date as string, tickers, signal),
    enabled: date !== null && tickers.length >= 2,
    placeholderData: keepPreviousData,
    staleTime: Infinity,
  });
}

export function usePair(pair: [string, string] | null) {
  return useQuery({
    queryKey: ["pair", pair?.[0], pair?.[1]],
    queryFn: ({ signal }) => api.pair(pair![0], pair![1], signal),
    enabled: pair !== null,
    placeholderData: keepPreviousData,
    staleTime: Infinity,
  });
}

// --- URL state ---------------------------------------------------------------

export interface ViewState {
  date: string | null;
  focus: string | null;
  basket: string[];
  pair: [string, string] | null;
}

function splitList(value: string | null): string[] {
  if (!value) return [];
  const out: string[] = [];
  const parts = value.split(",");
  for (let i = 0; i < parts.length; i++) {
    const t = parts[i]!.trim().toUpperCase();
    if (t && !out.includes(t)) out.push(t);
  }
  return out;
}

/** Read the initial view from `?date=&focus=&tickers=&pair=`. */
export function readUrlState(): ViewState {
  const params = new URLSearchParams(window.location.search);
  const pair = splitList(params.get("pair"));
  return {
    date: params.get("date"),
    focus: params.get("focus")?.trim().toUpperCase() || null,
    basket: splitList(params.get("tickers")),
    pair: pair.length === 2 ? [pair[0]!, pair[1]!] : null,
  };
}

/** Mirror the view into the URL (replaceState: shareable, reload-safe, no history spam). */
export function useUrlSync(state: ViewState, ready: boolean) {
  useEffect(() => {
    if (!ready) return;
    const params = new URLSearchParams();
    if (state.date) params.set("date", state.date);
    if (state.focus) params.set("focus", state.focus);
    if (state.basket.length) params.set("tickers", state.basket.join(","));
    if (state.pair) params.set("pair", state.pair.join(","));
    const query = params.toString().replace(/%2C/g, ",");
    const url = `${window.location.pathname}${query ? `?${query}` : ""}`;
    if (url !== `${window.location.pathname}${window.location.search}`) {
      window.history.replaceState(null, "", url);
    }
  }, [state, ready]);
}

// --- colour mode ---------------------------------------------------------------

export type ThemeChoice = "system" | "light" | "dark";
export type ColorMode = "light" | "dark";

const THEME_KEY = "corr-explorer-theme";
const darkQuery = "(prefers-color-scheme: dark)";

function subscribeSystem(callback: () => void) {
  const mq = window.matchMedia(darkQuery);
  mq.addEventListener("change", callback);
  return () => mq.removeEventListener("change", callback);
}

function readStoredTheme(): ThemeChoice {
  try {
    const value = window.localStorage.getItem(THEME_KEY);
    if (value === "light" || value === "dark") return value;
  } catch {
    // storage unavailable (private mode): fall back to the system setting
  }
  return "system";
}

/** Theme choice (system / light / dark), persisted per browser, applied as `data-theme`. */
export function useColorMode(): { choice: ThemeChoice; mode: ColorMode; cycle: () => void } {
  const [choice, setChoice] = useState<ThemeChoice>(readStoredTheme);
  const systemDark = useSyncExternalStore(subscribeSystem, () => window.matchMedia(darkQuery).matches);
  const mode: ColorMode = choice === "system" ? (systemDark ? "dark" : "light") : choice;

  useEffect(() => {
    const root = document.documentElement;
    if (choice === "system") delete root.dataset.theme;
    else root.dataset.theme = choice;
    try {
      if (choice === "system") window.localStorage.removeItem(THEME_KEY);
      else window.localStorage.setItem(THEME_KEY, choice);
    } catch {
      // ignore
    }
  }, [choice]);

  const cycle = useCallback(() => {
    setChoice((c) => (c === "system" ? "light" : c === "light" ? "dark" : "system"));
  }, []);
  return useMemo(() => ({ choice, mode, cycle }), [choice, mode, cycle]);
}
