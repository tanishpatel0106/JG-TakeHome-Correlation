// Browser regressions for the version switch, the download journey and seeding (real API, built UI).
import { type Page, expect, test } from "@playwright/test";

const BASKET = 'ul[aria-label="Heatmap tickers"] li span';

/** Collect uncaught errors (including unhandled promise rejections) and console errors. */
function watchErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
  page.on("console", (m) => {
    // A failed resource load is logged by the browser itself; the app's own handling is asserted per test.
    if (m.type() === "error" && !m.text().startsWith("Failed to load resource")) errors.push(`console: ${m.text()}`);
  });
  return errors;
}

async function openApp(page: Page) {
  await page.goto("/");
  await expect(page.locator(BASKET).first()).toBeVisible({ timeout: 20_000 });
}

async function activeDownloads(page: Page): Promise<number> {
  return ((await (await page.request.get("/__e2e/downloads/active")).json()) as { active: number }).active;
}

test.afterEach(async ({ page }) => {
  await page.request.post("/__e2e/downloads/release");
  await expect.poll(() => activeDownloads(page)).toBe(0);
});

test("switching to a newly published version remounts the charts without crashing", async ({ page }) => {
  const errors = watchErrors(page);
  await openApp(page);
  const badge = page.getByText(/· data as of/).first();
  const before = await badge.textContent();

  for (const kind of ["append", "correction"]) {
    const published = (await (await page.request.post("/__e2e/publish")).json()) as { published: string; kind: string };
    expect(published.kind).toBe(kind);
    const switchButton = page.getByRole("button", { name: "Switch to latest" });
    await expect(switchButton).toBeVisible({ timeout: 20_000 }); // the banner polls every 10 s
    await switchButton.click();
    await expect(badge).toContainText(published.published);
    await expect(page.locator("#timeline-title")).toBeVisible();
    await expect(page.getByText("Something went wrong in the page")).toHaveCount(0);
  }
  expect(await badge.textContent()).not.toBe(before);

  // The remounted timeline still selects a date on click.
  const urlDate = () => new URL(page.url()).searchParams.get("date");
  await expect.poll(urlDate).not.toBeNull();
  const dateBefore = urlDate();
  const canvas = page.locator('[aria-label="Mean pairwise correlation over time"] canvas').first();
  const box = (await canvas.boundingBox())!;
  await canvas.click({ position: { x: box.width * 0.3, y: box.height * 0.4 } });
  await expect.poll(urlDate).not.toBe(dateBefore);
  expect(errors).toEqual([]);
});

test.describe("download dialog", () => {
  async function openDialog(page: Page) {
    await openApp(page);
    await page.getByRole("button", { name: "Download ρ matrix" }).click();
    const dialog = page.getByRole("dialog");
    await expect(dialog.getByText(/Exact size/)).toBeVisible();
    return dialog;
  }

  test("a real 429 from the download cap is shown, and a retry succeeds once a slot is free", async ({ page }) => {
    const errors = watchErrors(page);
    const dialog = await openDialog(page);
    const url = new URL(page.url());
    // Occupy both server slots with downloads held in flight (headers sent, body paused).
    await page.request.post("/__e2e/downloads/hold");
    const date = url.searchParams.get("date");
    const held = [0, 1].map(() =>
      fetch(`${url.origin}/api/corr/${date}/download?format=npy`).then((r) => r.arrayBuffer().catch(() => null)),
    );
    await expect.poll(() => activeDownloads(page)).toBe(2);

    await dialog.getByRole("button", { name: /^Download \d/ }).click();
    const alert = dialog.getByRole("alert");
    await expect(alert).toContainText("Download failed.");
    await expect(alert).toContainText("At most 2 downloads at a time");
    await expect(alert).toContainText("retry in about 5 s");
    await expect(dialog).toBeVisible(); // the dialog stays open with the reason

    await page.request.post("/__e2e/downloads/release");
    await Promise.all(held);
    await expect.poll(() => activeDownloads(page)).toBe(0);
    const saved = page.waitForEvent("download");
    await dialog.getByRole("button", { name: "Retry download" }).click();
    const file = await saved;
    expect(file.suggestedFilename()).toMatch(/\.npz$/);
    await expect(dialog.getByText(/^Saved .*complete/)).toBeVisible();
    expect(errors).toEqual([]);
  });

  test("a stream dropped mid-download is reported and nothing is saved", async ({ page }) => {
    const errors = watchErrors(page);
    const dialog = await openDialog(page);
    let saves = 0;
    page.on("download", () => saves++);
    await page.request.post("/__e2e/downloads/drop");
    await dialog.getByRole("button", { name: /^Download \d/ }).click();
    const alert = dialog.getByRole("alert");
    await expect(alert).toContainText("Download failed.");
    await expect(alert).toContainText(/connection dropped|Incomplete download/);
    expect(saves).toBe(0);
    await expect.poll(() => activeDownloads(page)).toBe(0);

    const saved = page.waitForEvent("download");
    await dialog.getByRole("button", { name: "Retry download" }).click();
    await saved;
    expect(saves).toBe(1);
    expect(errors).toEqual([]);
  });

  test("a failing /download/info is shown and blocks the download until it succeeds", async ({ page }) => {
    const errors = watchErrors(page);
    let fail = true;
    await page.route("**/download/info**", (route) =>
      fail
        ? route.fulfill({ status: 503, contentType: "application/json", body: JSON.stringify({ detail: { code: "unavailable", message: "Info service down" } }) })
        : route.continue(),
    );
    await openApp(page);
    await page.getByRole("button", { name: "Download ρ matrix" }).click();
    const dialog = page.getByRole("dialog");
    await expect(dialog.getByRole("alert")).toContainText("Info service down");
    await expect(dialog.getByRole("button", { name: /^Download \d/ })).toBeDisabled();
    fail = false;
    await dialog.getByRole("button", { name: "Retry" }).click();
    await expect(dialog.getByText(/Exact size/)).toBeVisible();
    await expect(dialog.getByRole("button", { name: /^Download \d/ })).toBeEnabled();
    expect(errors).toEqual([]);
  });
});

test("a failed seed request is reported, keeps the basket, and can be retried", async ({ page }) => {
  const errors = watchErrors(page);
  await openApp(page);
  const before = await page.locator(BASKET).allTextContents();
  let fail = true;
  await page.route(/\/api\/corr\/[^/]+\/stock\/[^/?]+\?.*k=14/, (route) =>
    fail
      ? route.fulfill({
          status: 503,
          contentType: "application/json",
          body: JSON.stringify({ detail: { code: "AUDIT_SEED_UNAVAILABLE", message: "Neighbour service unavailable" } }),
        })
      : route.continue(),
  );
  // Start from a basket that differs from the seed result, so a successful retry is visible.
  await page.getByRole("button", { name: `Remove ${before[before.length - 1]}` }).click();
  const kept = await page.locator(BASKET).allTextContents();

  await page.getByRole("button", { name: /^Seed: focus \+ top 14/ }).click();
  const alert = page.getByRole("alert").filter({ hasText: "Could not seed" });
  await expect(alert).toContainText("Neighbour service unavailable");
  await expect(alert).toContainText("left unchanged");
  expect(await page.locator(BASKET).allTextContents()).toEqual(kept);
  await expect(page.getByRole("button", { name: /^Seed: focus \+ top 14/ })).toBeEnabled();

  fail = false;
  await alert.getByRole("button", { name: "Retry" }).click();
  await expect(alert).toHaveCount(0);
  await expect(page.locator(BASKET)).toHaveCount(15);
  expect(errors).toEqual([]);
});
