"""
Evaluation Script for Segmented Rice Leaf RT-DETR Model (Ablation Study)
Evaluates best.pt on the Segmented Test Set (354 images) and compares with Champion Model v5.
To be executed on NVIDIA GeForce RTX 4090 or local machine.
"""
import os
import sys
import argparse
from pathlib import Path
from ultralytics import RTDETR

# Champion Model v5 Benchmark Metrics (on Test Set)
V5_BENCHMARK = {
    "bacterial_leaf_blight": 0.695,
    "brown_spot": 0.461,
    "leaf_blast": 0.655,
    "leaf_scald": 0.848,
    "sheath_blight": 0.871,
    "all": 0.706,
    "map": 0.355,
    "precision": 0.735,
    "recall": 0.702
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
    parser = argparse.ArgumentParser(description="Evaluate RT-DETR Segmented Model on Test Set")
    parser.add_argument('--weights', type=str, default=None, help="Path to best.pt")
    parser.add_argument('--data', type=str, default=None, help="Path to data.yaml")
    parser.add_argument('--imgsz', type=int, default=800, help="Image size (default: 800)")
    parser.add_argument('--device', type=str, default=None, help="Device (default: auto)")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)

    dataset_yaml = Path(args.data).resolve() if args.data else (script_dir / "dataset_segmented" / "data.yaml")
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Segmented dataset config not found at: {dataset_yaml}")

    if args.weights:
        weights_path = Path(args.weights).resolve()
    else:
        weights_path = find_segmented_weights(script_dir)

    if not weights_path or not weights_path.exists():
        raise FileNotFoundError(
            "Could not find 'best.pt' in runs/rtdetr_segmented_ablation/weights/. "
            "Please ensure train_segmented_rtdetr.py has finished training."
        )

    print("=" * 80)
    print("RT-DETR-L Segmented Model Evaluation on Test Set (Ablation Study)")
    print("=" * 80)
    print(f"Target Weights : {weights_path}")
    print(f"Dataset Config : {dataset_yaml}")
    print(f"Image Size     : {args.imgsz}")
    print("=" * 80)

    model = RTDETR(str(weights_path))

    run_dir = weights_path.parent.parent
    eval_kwargs = {
        "data": str(dataset_yaml),
        "split": 'test',
        "imgsz": args.imgsz,
        "project": str(run_dir),
        "name": 'eval_test',
        "exist_ok": True,
        "save_json": False,
        "plots": True
    }
    if args.device is not None:
        eval_kwargs["device"] = args.device

    print(f"\nEvaluating on Test Set (split='test', imgsz={args.imgsz})...")
    metrics = model.val(**eval_kwargs)

    names = model.names if hasattr(model, 'names') else {
        0: 'bacterial_leaf_blight',
        1: 'brown_spot',
        2: 'leaf_blast',
        3: 'leaf_scald',
        4: 'sheath_blight'
    }

    # Extract per-class metrics accurately
    class_map50 = {}
    class_map = {}
    class_p = {}
    class_r = {}

    if hasattr(metrics.box, 'ap_class_index') and len(metrics.box.ap_class_index) > 0:
        for i, c in enumerate(metrics.box.ap_class_index):
            c_name = names.get(c, f"class_{c}")
            p, r, ap50, ap = metrics.box.class_result(i)
            class_map50[c_name] = float(ap50)
            class_p[c_name] = float(p)
            class_r[c_name] = float(r)
            class_map[c_name] = float(ap)
    elif hasattr(metrics.box, 'ap50') and len(metrics.box.ap50) > 0:
        for idx, ap50 in enumerate(metrics.box.ap50):
            c_name = names.get(idx, f"class_{idx}")
            class_map50[c_name] = float(ap50)
    elif hasattr(metrics.box, 'maps') and metrics.box.maps is not None:
        for idx, ap in enumerate(metrics.box.maps):
            c_name = names.get(idx, f"class_{idx}")
            class_map50[c_name] = float(ap)

    overall_map50 = float(metrics.box.map50)
    overall_map = float(metrics.box.map)
    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)

    # Build comparison summary
    lines = []
    lines.append("=" * 80)
    lines.append("Ablation Study Comparison: Champion Model v5 vs Segmented-First Model")
    lines.append("=" * 80)
    lines.append(f"{'Disease Class':<25} | {'Champion v5 (mAP50)':<20} | {'Segmented (mAP50)':<18} | {'Diff'}")
    lines.append("-" * 80)

    for c_id in sorted(names.keys()):
        c_name = names[c_id]
        v5_score = V5_BENCHMARK.get(c_name, 0.0)
        seg_score = class_map50.get(c_name, 0.0)
        diff = seg_score - v5_score
        sign = "+" if diff >= 0 else ""
        lines.append(f"{c_name:<25} | {v5_score*100:>18.2f}% | {seg_score*100:>16.2f}% | {sign}{diff*100:.2f}%")

    all_diff = overall_map50 - V5_BENCHMARK['all']
    all_sign = "+" if all_diff >= 0 else ""
    p_diff = precision - V5_BENCHMARK['precision']
    p_sign = "+" if p_diff >= 0 else ""
    r_diff = recall - V5_BENCHMARK['recall']
    r_sign = "+" if r_diff >= 0 else ""
    map_diff = overall_map - V5_BENCHMARK['map']
    map_sign = "+" if map_diff >= 0 else ""

    lines.append("-" * 80)
    lines.append(f"{'OVERALL (mAP@0.5)':<25} | {V5_BENCHMARK['all']*100:>18.2f}% | {overall_map50*100:>16.2f}% | {all_sign}{all_diff*100:.2f}%")
    lines.append(f"{'mAP@0.5:0.95':<25} | {V5_BENCHMARK['map']*100:>18.2f}% | {overall_map*100:>16.2f}% | {map_sign}{map_diff*100:.2f}%")
    lines.append(f"{'Precision':<25} | {V5_BENCHMARK['precision']*100:>18.2f}% | {precision*100:>16.2f}% | {p_sign}{p_diff*100:.2f}%")
    lines.append(f"{'Recall':<25} | {V5_BENCHMARK['recall']*100:>18.2f}% | {recall*100:>16.2f}% | {r_sign}{r_diff*100:.2f}%")
    lines.append("=" * 80)

    summary_text = "\n".join(lines)
    print("\n" + summary_text)

    # Save to file
    out_file = run_dir / "ablation_test_results.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(f"\nResults successfully saved to: {out_file}")

if __name__ == '__main__':
    main()
