from datasets import load_dataset

def stream_real_and_fake_samples(sample_limit=1500):
    """
    Streams a balanced set of real/synthetic images straight from the Hub.
    Uses virtually zero local disk storage.
    """
    print("Connecting to remote dataset stream...")
    # CIFAKE is an excellent, light 32x32 dataset; ARTIFACTS contains 1024x1024 high-res files
    remote_stream = load_dataset("polytechnique-montreal/artifacts", split="train", streaming=True)
    
    processed_pool = []
    real_count = 0
    fake_count = 0
    target_per_class = sample_limit // 2
    
    for item in remote_stream:
        is_ai = item['label'] # 1 for fake, 0 for real
        
        if is_ai == 1 and fake_count < target_per_class:
            fake_count += 1
        elif is_ai == 0 and real_count < target_per_class:
            real_count += 1
        else:
            continue # Skip if class balance is met
            
        # Add your nested metadata tracking on the fly
        record = {
            "image": item['image'], # Kept as an active PIL object in memory
            "label": is_ai,
            "metadata": {
                "software_sig": "None" if is_ai else "Camera Hardware Signature",
                "exif_profile": "None" if is_ai else "Embedded EXIF Profile",
                "color_space": "sRGB"
            }
        }
        processed_pool.append(record)
        
        if len(processed_pool) >= sample_limit:
            break
            
    print(f"Streamed {len(processed_pool)} balanced images into memory successfully.")
    return processed_pool