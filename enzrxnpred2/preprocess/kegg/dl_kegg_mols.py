import os
import time

import requests

from adaptplm.core.default_path import DefaultPath

input_file = DefaultPath().build / 'kegg' / 'kegg_mol_ids_for_esp_fine_tuning.txt'
# input_file = DefaultPath().build / 'kegg' / 'sample_for_test.txt'
output_dir = DefaultPath().build / 'kegg' / "kegg_entries"
output_dir.mkdir(parents=True, exist_ok=True)

# API settings
BASE_URL = "https://rest.kegg.jp/get/"
BATCH_SIZE = 10  # Number of IDs to retrieve at once (KEGG allows multiple IDs joined with '+')
WAIT_TIME = 2  # Interval between requests (seconds)
MAX_RETRIES = 3  # Maximum number of retries


def main():
    with open(input_file, "r") as f:
        ids = [line.strip() for line in f if line.strip()]

    for i in range(0, len(ids), BATCH_SIZE):
        batch_ids = ids[i:i + BATCH_SIZE]
        batch_str = "+".join(batch_ids)
        url = BASE_URL + batch_str + "/mol"

        print(f"[{i + 1}/{len(ids)}] Fetching {len(batch_ids)} MOL files: {batch_ids}")

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    mol_blocks = response.text.split("\n$$$$\n")
                    for mol_block, mol_id in zip(mol_blocks, batch_ids):
                        # Get until 'M  END'
                        if "M  END" in mol_block:
                            mol_block = mol_block.split("M  END")[0] + "M  END\n"
                        mol_path = os.path.join(output_dir, f"{mol_id}.mol")
                        with open(mol_path, "w") as mf:
                            mf.write(mol_block)
                    print(f"  ✅ Saved batch {i // BATCH_SIZE + 1}")
                    break
                else:
                    print(f"  ⚠️ Status {response.status_code}, retry {attempt}/{MAX_RETRIES}")
            except requests.RequestException as e:
                print(f"  ❌ Error: {e}, retry {attempt}/{MAX_RETRIES}")

            time.sleep(WAIT_TIME)

        time.sleep(WAIT_TIME)


if __name__ == '__main__':
    main()
