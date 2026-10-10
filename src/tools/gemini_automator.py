import asyncio
import random
from playwright.async_api import async_playwright

# --- 1. Prompt Generation ---
BREEDS = ["Domestic Shorthair", "Domestic Longhair", "Maine Coon", "Norwegian Forest Cat", "Siberian", "Persian", "British Shorthair", "Bengal", "Siamese", "Ragdoll", "Turkish Van", "Oriental Shorthair"]
COLORS = ["solid black", "solid white", "orange (ginger)", "cream", "chocolate brown", "cinnamon", "silver tabby", "brown tabby", "tuxedo", "calico", "tortoiseshell", "colorpoint", "bicolor", "tricolor"]
PREY = ["rabbit", "bird", "mouse", "rat", "squirrel", "vole", "chipmunk"]
POSES = ["walking", "standing", "running", "jumping", "crouching", "carrying its prey home", "entering a cat flap", "looking slightly away from the camera"]
ENVIRONMENTS = ["sunny garden", "cloudy backyard", "lawn at dawn", "dense forest at dusk", "grassy field", "gravel path after rainfall", "suburban sidewalk", "rural property in the snow", "farm at night with porch lighting", "garden captured with infrared security camera lighting"]

def generate_prompts(count=80):
    prompts = []
    for i in range(count):
        breed = random.choice(BREEDS)
        color = random.choice(COLORS)
        prey_animal = random.choice(PREY)
        pose = random.choice(POSES)
        env = random.choice(ENVIRONMENTS)
        angle_h = random.randint(25, 65)
        angle_dir = random.choice(["left", "right"])
        angle_v = random.randint(-15, 15)
        
        prompt = (
            f"A highly detailed, photorealistic wildlife-style DSLR photograph of a {color} {breed} cat {pose} in a {env}. "
            f"The cat is carrying a {prey_animal} firmly in its mouth. The prey's body and tail are clearly visible extending outside the mouth. "
            f"Viewed from a {angle_h}° horizontal angle facing {angle_dir}, with a {angle_v}° vertical angle. "
            f"Entire head and mouth are visible. The cat occupies 60% of the frame. Ultra-photorealistic, realistic fur texture, 4K detail. "
            f"Negative constraints: no humans, no toys, no fake prey, no duplicate poses, no front view, no perfect side profile, clean, pristine, alive."
        )
        prompts.append(prompt)
    return prompts

# --- 2. Playwright Automation ---
async def main():
    prompts = generate_prompts(80)
    
    print("Connecting to existing Chrome browser...")
    print("NOTE: You MUST launch Chrome from your terminal first with this exact command:")
    print('  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe" --remote-debugging-port=9222')
    
    async with async_playwright() as p:
        try:
            # Connect to the Chrome instance you opened manually
            browser = await p.chromium.connect_over_cdp("http://localhost:9222")
        except Exception as e:
            print("Failed to connect to Chrome. Did you launch it with --remote-debugging-port=9222 ?")
            return

        default_context = browser.contexts[0]
        page = default_context.pages[0]
        
        if "gemini.google.com" not in page.url:
            await page.goto("https://gemini.google.com/app")
            print("Please log in to Gemini if you aren't already, then press Enter here to continue.")
            input()

        print(f"Starting generation of {len(prompts)} images...")
        
        for i, prompt_text in enumerate(prompts):
            print(f"\n[{i+1}/{len(prompts)}] Sending prompt...")
            
            try:
                # 1. Locate the input box. Gemini's input is typically a contenteditable div
                chat_input = page.locator('div[contenteditable="true"]').first
                await chat_input.wait_for(state="visible")
                
                # 2. Type the prompt
                await chat_input.fill(prompt_text)
                await page.wait_for_timeout(1000)
                
                # 3. Press Enter to submit
                await chat_input.press("Enter")
                
                # 4. Wait for the image to generate. 
                # We wait 30 seconds for the response to fully stream and render the image.
                print("Waiting 30 seconds for generation to complete...")
                await page.wait_for_timeout(30000) 
                
                # We do not automate downloading because Gemini uses dynamic blob URLs and complex React states.
                # The easiest method is to let the script run to completion, then manually scroll up and 
                # click the 'Download all images' button that Gemini provides on each response.
                
                # 5. Cooldown to avoid triggering spam filters on the web UI
                print("Cooling down for 10 seconds before next prompt...")
                await page.wait_for_timeout(10000)
                
            except Exception as e:
                print(f"Error during prompt {i+1}: {e}")
                print("Sleeping for 30s before trying the next one...")
                await page.wait_for_timeout(30000)

        print("\nAll prompts sent! You can now manually save the generated images from your browser.")

if __name__ == "__main__":
    asyncio.run(main())
