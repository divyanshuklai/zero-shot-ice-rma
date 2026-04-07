import numpy as np
from PIL import Image
import noise 

# --- CONFIGURATION ---
WIDTH, HEIGHT = 500, 500   # 500px / 20m = 4cm resolution
OCTAVES = 2
LACUNARITY = 2.0
GAIN = 0.25

def generate_jagged_terrain():
    # --- THE FIX IS HERE ---
    # scale = 80.0 (Old: created smooth, 3-meter wide hills)
    # scale = 5.0  (New: creates jagged, 20cm wide bumps)
    scale = 5.0 
    
    terrain = np.zeros((HEIGHT, WIDTH))
    seed = np.random.randint(0, 100)

    for y in range(HEIGHT):
        for x in range(WIDTH):
            val = noise.pnoise2(x / scale, 
                                y / scale, 
                                octaves=OCTAVES, 
                                persistence=GAIN, 
                                lacunarity=LACUNARITY, 
                                repeatx=WIDTH, 
                                repeaty=HEIGHT, 
                                base=seed)
            terrain[y][x] = val

    # Normalize carefully to ensure sharp contrast
    min_val = np.min(terrain)
    max_val = np.max(terrain)
    terrain = (terrain - min_val) / (max_val - min_val)
    
    return (terrain * 255).astype(np.uint8)

img_data = generate_jagged_terrain()
Image.fromarray(img_data, mode='L').save('fractal_terrain.png')
print(f"Generated JAGGED terrain.")