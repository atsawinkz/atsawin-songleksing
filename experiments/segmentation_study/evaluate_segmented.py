"""
Evaluation Script for Segmented Rice Leaf RT-DETR Model (Ablation Study)
Evaluates best.pt on the Segmented Test Set (354 images) and compares with Champion Model v5.
To be executed on NVIDIA GeForce RTX 4090 or local machine.
"""
import os
import glob
from pathlib import Path
from ultralytics import RTDETR

# Champion Model v5 Benchmark Metrics (on Test Set)
V5_BENCHMARK = {
    "bacterial_leaf_blight": 0.695,
    "brown_spot": 0.461,
    "leaf_blast": 0.655,
    "leaf_scald": 0.848,
    "sheath_blight": 0.871,
    "all": 0.706
}

def find_segmented_weights(script_dir: Path):
    candidate_paths = [
        script_dir / "runs" / "rtdetr_segmented_ablation" / "weights" / "best.pt",
        script_dir / "runs" / "weights" / "best.pt",
    ]
    for p in candidate_paths:
        if p.exists():
            return p
            
    files = list(script_dir.glob("**/best.pt"))
    if files:
        files.sort(key=os.path.getmtime, reverse=True)
        return files[0]
        
    return None

def main():
    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)

    dataset_yaml = script_dir / "dataset_segmented" / "data.yaml"
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Segmented dataset config not found at: {dataset_yaml}")

    weights_path = find_segmented_weights(script_dir)
    if not weights_path:
        raise FileNotFoundError(
            "Could not find 'best.pt' in runs/rtdetr_segmented_ablation/weights/. "
            "Please ensure train_segmented_rtdetr.py has finished training."
        )

    print("=" * 70)
    print("RT-DETR-L Segmented Model Evaluation on Test Set (Ablation Study)")
    print("=" * 70)
    print(f"Target Weights : {weights_path}")
    print(f"Dataset Config : {dataset_yaml}")
    print("=" * 70)

    model = RTDETR(str(weights_path))

    # Evaluate strictly on the Test Set (split='test') at imgsz=800
    print("\nEvaluating on Test Set (split='test', imgsz=800)...")
    run_dir = weights_path.parent.parent
    metrics = model.val(
        data=str(dataset_yaml),
        split='test',
        imgsz=800,
        project=str(run_dir),
        name='eval_test',
        exist_ok=True
    )

    # Class mappings
    names = model.names if hasattr(model, 'names') else {
        0: 'bacterial_leaf_blight',
        1: 'brown_spot',
        2: 'leaf_blast',
        3: 'leaf_scald',
        4: 'sheath_blight'
    }

    # Extract per-class mAP50
    class_map50 = {}
    if hasattr(metrics.box, 'maps') and metrics.box.maps is not None:
        for idx, ap50 in enumerate(metrics.box.maps):
            c_name = names.get(idx, f"class_{idx}")
            class_map50[c_name] = float(ap50)

    overall_map50 = float(metrics.box.map50)
    overall_map = float(metrics.box.map)
    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)

    # Build comparison summary
    lines = []
    lines.append("=" * 75)
    lines.append("Ablation Study Comparison: Champion Model v5 vs Segmented-First Model")
    lines.append("=" * 75)
    lines.append(f"{'Disease Class':<25} | {'Champion v5 (mAP50)':<20} | {'Segmented (mAP50)':<18} | {'Diff'}")
    lines.append("-" * 75)

    for c_id in sorted(names.keys()):
        c_name = names[c_id]
        v5_score = V5_BENCHMARK.get(c_name, 0.0)
        seg_score = class_map50.get(c_name, 0.0)
        diff = seg_score - v5_score
        sign = "+" if diff >= 0 else ""
        lines.append(f"{c_name:<25} | {v5_score*100:>18.2f}% | {seg_score*100:>16.2f}% | {sign}{diff*100:.2f}%")

    all_diff = overall_map50 - V5_BENCHMARK['all']
    all_sign = "+" if all_diff >= 0 else ""
    lines.append("-" * 75)
    lines.append(f"{'OVERALL (mAP@0.5)':<25} | {V5_BENCHMARK['all']*100:>18.2f}% | {overall_map50*100:>16.2f}% | {all_sign}{all_diff*100:.2f}%")
    lines.append(f"{'mAP@0.5:0.95':<25} | {'-':>19} | {overall_map*100:>16.2f}% | -")
    lines.append(f"{'Precision':<25} | {'-':>19} | {precision*100:>16.2f}% | -")
    lines.append(f"{'Recall':<25} | {'-':>19} | {recall*100:>16.2f}% | -")
    lines.append("=" * 75)

    summary_text = "\n".join(lines)
    print("\n" + summary_text)

    # Save to file
    out_file = run_dir / "ablation_test_results.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(f"\nResults successfully saved to: {out_file}")

if __name__ == '__main__':
    main()
