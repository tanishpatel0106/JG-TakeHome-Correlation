import { defineConfig, devices } from "@playwright/test";

// Browser regressions against the real API + built UI (run `npm run build` first; `make e2e` does both).
// The server is e2e/server.py on a throwaway demo store; PYTHON selects the interpreter (default: python).
const PORT = Number(process.env.E2E_PORT ?? 4179);

export default defineConfig({
  testDir: "e2e",
  workers: 1,
  retries: 0,
  timeout: 60_000,
  reporter: [["list"]],
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    viewport: { width: 1440, height: 1000 },
    acceptDownloads: true,
    trace: "retain-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"], viewport: { width: 1440, height: 1000 } } }],
  webServer: {
    command: `${process.env.PYTHON ?? "python"} e2e/server.py --port ${PORT}`,
    url: `http://127.0.0.1:${PORT}/api/health`,
    cwd: ".",
    env: { ...(process.env as Record<string, string>), PYTHONPATH: "../.." },
    timeout: 120_000,
    reuseExistingServer: false,
  },
});
