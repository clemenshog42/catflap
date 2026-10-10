import asyncio
import base64
import os
import time
import urllib.request
from playwright.async_api import async_playwright

SAVE_DIR = r"c:\Public\Studium\Bachelorarbeit\code\catprey"

async def main():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    print("Connecting to existing Chrome browser...")
    
    async with async_playwright() as p:
        try:
            browser = await p.chromium.connect_over_cdp("http://localhost:9222")
        except Exception as e:
            print("Failed to connect to Chrome. Did you launch it with --remote-debugging-port=9222 ?")
            return

        default_context = browser.contexts[0]
        page = default_context.pages[0]
        
        print("\n" + "="*50)
        print("IMPORTANT: Please make sure you are on the Gemini tab.")
        print("Scroll through the entire chat to ensure all images are fully loaded on your screen.")
        print("="*50 + "\n")
        input("Press Enter when you are ready...")

        print("Looking for generated images...")
        
        # 1. Find all images on the page
        imgs = await page.locator("img").all()
        large_imgs = []
        for img in imgs:
            try:
                # Check the bounding box to identify large generated images
                # We use 100x100 to be extremely safe, as the UI might scale them down
                box = await img.bounding_box()
                if box and box['width'] >= 100 and box['height'] >= 100:
                    large_imgs.append(img)
            except:
                pass
                
        if not large_imgs:
            print("No large images found. Make sure they are visible on screen.")
            return
            
        print(f"Found {len(large_imgs)} large images! Extracting them directly from the browser's memory...")
        
        saved_count = 0
        for i, img in enumerate(large_imgs):
            try:
                # Scroll the image into view to ensure it is rendered
                await img.scroll_into_view_if_needed()
                await page.wait_for_timeout(300)
                
                # Evaluate JS directly on the image element to fetch its binary data from memory
                # This completely bypasses the need to click any buttons
                b64_data = await img.evaluate("""
                    async (imgEl) => {
                        try {
                            const response = await fetch(imgEl.src);
                            const blob = await response.blob();
                            return await new Promise((resolve, reject) => {
                                const reader = new FileReader();
                                reader.onloadend = () => resolve(reader.result);
                                reader.onerror = reject;
                                reader.readAsDataURL(blob);
                            });
                        } catch (e) {
                            return "ERROR: " + e.toString();
                        }
                    }
                """)
                
                if b64_data and not b64_data.startswith("ERROR"):
                    if "," in b64_data:
                        header, encoded = b64_data.split(",", 1)
                    else:
                        encoded = b64_data
                        
                    image_data = base64.b64decode(encoded)
                    filename = f"gemini_image_{int(time.time())}_{i+1}.png"
                    filepath = os.path.join(SAVE_DIR, filename)
                    
                    with open(filepath, "wb") as f:
                        f.write(image_data)
                    print(f"Extracted and saved: {filename}")
                    saved_count += 1
                else:
                    # Fallback: If it's a standard URL that blocked JS fetch, download it directly in Python
                    src = await img.get_attribute("src")
                    if src and src.startswith("http"):
                        print(f"Direct JS extraction failed, falling back to Python download for image {i+1}...")
                        req = urllib.request.Request(src, headers={'User-Agent': 'Mozilla/5.0'})
                        with urllib.request.urlopen(req) as response:
                            image_data = response.read()
                            filename = f"gemini_image_{int(time.time())}_{i+1}.png"
                            filepath = os.path.join(SAVE_DIR, filename)
                            with open(filepath, "wb") as f:
                                f.write(image_data)
                            print(f"Downloaded and saved: {filename}")
                            saved_count += 1
                    else:
                        print(f"Could not extract image {i+1}. Error: {b64_data}")
            except Exception as e:
                print(f"Failed to process image {i+1}: {e}")

        print(f"\nDone! Successfully saved {saved_count} images to {SAVE_DIR}.")

if __name__ == "__main__":
    asyncio.run(main())
