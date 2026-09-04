"""Generate repairs for any missing problem kernels."""

import json
from pathlib import Path
from kernelbench.utils import create_inference_server_from_presets, extract_first_code

def main():
    manifest_path = Path("runs/cloud_agents_l2_turn2/diagnostic_manifest.json")
    with open(manifest_path) as f:
        m = json.load(f)

    server = create_inference_server_from_presets("google", "gemini-3.8-flash", 8192, 0.0)

    for pid in [33, 35]:
        kernel_path = Path("runs/cloud_agents_l2_turn2") / f"level_2_problem_{pid}_sample_0_kernel.py"
        if kernel_path.exists():
            print(f"Problem {pid} already exists.")
            continue
        print(f"Generating Problem {pid}...")
        prompt = m["failures"][str(pid)]["prompt"]
        resp = server(prompt)
        code = extract_first_code(resp, ["python", "cpp"])
        if not code and "class ModelNew" in resp:
            idx = resp.find("import torch")
            if idx == -1:
                idx = resp.find("class ModelNew")
            end_idx = resp.rfind("```")
            if end_idx != -1 and end_idx > idx:
                code = resp[idx:end_idx].strip("` \n")
            else:
                code = resp[idx:].strip("` \n")

        if code:
            kernel_path.write_text(code, encoding="utf-8")
            print(f"Problem {pid} saved successfully ({len(code)} bytes).")
        else:
            print(f"Failed to extract code for Problem {pid}.")

if __name__ == "__main__":
    main()
