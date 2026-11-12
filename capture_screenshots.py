"""
Automated screenshot generator for Solar Sense – Chengalpattu.
Uses Playwright with local Google Chrome to navigate to each page,
wait for Plotly charts and calculations, and capture full-page screenshots.
"""

import time
import os
from playwright.sync_api import sync_playwright

CHROME_PATH = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def capture_all():
    os.makedirs("screenshots", exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=CHROME_PATH,
            headless=True,
            args=["--no-sandbox", "--disable-gpu"],
        )
        context = browser.new_context(
            viewport={"width": 1380, "height": 950},
            device_scale_factor=1.5,
        )
        page = context.new_page()

        print("Navigating to http://localhost:8501...")
        page.goto("http://localhost:8501", wait_until="networkidle", timeout=30000)
        time.sleep(4)

        pages = [
            ("🏠 Home / Overview", "01_home_overview.png"),
            ("⚡ Live Check", "02_live_check.png"),
            ("🎛️ What-if Simulator", "03_whatif_simulator.png"),
            ("📊 Model Performance", "04_model_performance.png"),
            ("🔍 Explainability", "05_explainability.png"),
            ("🌱 Solar & CO₂ Calculator", "06_solar_co2_calculator.png"),
            ("ℹ️ About & Data", "07_about_data.png"),
        ]

        for nav_text, filename in pages:
            print(f"Selecting {nav_text}...")
            label = page.locator(f'label:has-text("{nav_text}")')
            if label.count() > 0:
                label.first.click()
            else:
                print(f"Warning: Label {nav_text} not found directly, searching text...")
                page.get_by_text(nav_text).first.click()

            # Wait for Streamlit to process and rerender
            time.sleep(3.5)
            try:
                page.wait_for_selector(".stSpinner", state="detached", timeout=5000)
            except Exception:
                pass
            time.sleep(1.5)

            filepath = os.path.join("screenshots", filename)
            page.screenshot(path=filepath, full_page=True)
            print(f"Saved screenshot: {filepath} ({os.path.getsize(filepath)} bytes)")

        # Also capture Live Check in Force Offline Mode to demonstrate fallback behavior!
        print("Selecting ⚡ Live Check for offline fallback demo...")
        page.locator('label:has-text("⚡ Live Check")').first.click()
        time.sleep(2)
        checkbox = page.locator('label:has-text("Simulate Offline Mode")')
        if checkbox.count() > 0:
            checkbox.first.click()
            time.sleep(3)
            offline_path = os.path.join("screenshots", "02b_live_check_offline_mode.png")
            page.screenshot(path=offline_path, full_page=True)
            print(f"Saved screenshot: {offline_path} ({os.path.getsize(offline_path)} bytes)")

        browser.close()
        print("All screenshots successfully captured!")


if __name__ == "__main__":
    capture_all()
