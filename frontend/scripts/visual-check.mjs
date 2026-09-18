// One-off manual QA driver (not part of the app or its build) -- launches
// headless Chromium against the running Vite dev server, walks each page,
// captures console errors and a screenshot per page. Run with:
//   node scripts/visual-check.mjs
import { chromium } from "playwright";
import fs from "node:fs";

const BASE = process.env.BASE_URL || "http://localhost:5174";
const OUT_DIR = "scripts/screenshots";
fs.mkdirSync(OUT_DIR, { recursive: true });

const PAGES = [
  ["/", "dashboard"],
  ["/raise-request", "raise-request"],
  ["/pending-approval", "pending-approval"],
  ["/recommended-scheduling", "recommended-scheduling"],
  ["/schedule", "schedule"],
  ["/history", "history"],
];

const browser = await chromium.launch({ args: ["--no-sandbox"] });
const page = await browser.newPage({ viewport: { width: 1440, height: 960 } });

const allErrors = [];
page.on("console", (msg) => {
  if (msg.type() === "error") allErrors.push(`[console] ${msg.text()}`);
});
page.on("pageerror", (err) => allErrors.push(`[pageerror] ${err.message}`));
page.on("requestfailed", (req) => allErrors.push(`[requestfailed] ${req.url()} — ${req.failure()?.errorText}`));

for (const [path, name] of PAGES) {
  const errorsBefore = allErrors.length;
  await page.goto(`${BASE}${path}`, { waitUntil: "networkidle" });
  await page.waitForTimeout(600); // let async fetches settle
  const shotPath = `${OUT_DIR}/${name}.png`;
  await page.screenshot({ path: shotPath, fullPage: true });
  const newErrors = allErrors.slice(errorsBefore);
  console.log(`\n=== ${path} -> ${shotPath} ===`);
  if (newErrors.length) {
    console.log("ERRORS:");
    newErrors.forEach((e) => console.log("  " + e));
  } else {
    console.log("no console/page/request errors");
  }
}

await browser.close();
console.log(`\nTotal errors across all pages: ${allErrors.length}`);
process.exit(allErrors.length > 0 ? 1 : 0);
