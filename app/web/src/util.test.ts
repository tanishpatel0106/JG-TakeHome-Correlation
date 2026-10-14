import { describe, expect, it } from "vitest";

import { fisherCI, isFresh, snapDate } from "./util";

describe("isFresh (stale-result guard)", () => {
  const data = { date: "2024-09-06", ticker: "A", k: 20 };
  it("accepts data that matches the current selection", () => {
    expect(isFresh(data, false, { date: "2024-09-06", ticker: "A", k: 20 })).toBe(true);
  });
  it("rejects placeholder data and any mismatch (previous ticker, date or k)", () => {
    expect(isFresh(data, true, { date: "2024-09-06", ticker: "A", k: 20 })).toBe(false);
    expect(isFresh(data, false, { date: "2024-09-06", ticker: "NA", k: 20 })).toBe(false);
    expect(isFresh(data, false, { date: "2024-09-05", ticker: "A", k: 20 })).toBe(false);
    expect(isFresh(data, false, { date: "2024-09-06", ticker: "A", k: 50 })).toBe(false);
    expect(isFresh(undefined, false, { date: "2024-09-06" })).toBe(false);
  });
});

describe("fisherCI", () => {
  it("gives the documented 95% interval for r = 0.5, n = 20", () => {
    const ci = fisherCI(0.5, 20)!;
    expect(ci[0]).toBeCloseTo(0.07, 2);
    expect(ci[1]).toBeCloseTo(0.77, 2);
    expect(fisherCI(null, 20)).toBeNull();
  });
});

describe("snapDate", () => {
  it("snaps to the nearest selectable date", () => {
    expect(snapDate("2024-01-03", ["2024-01-01", "2024-01-04"])).toBe("2024-01-04");
    expect(snapDate("2024-01-01", ["2024-01-01"])).toBe("2024-01-01");
  });
});
