import json
from pathlib import Path
import numpy as np
import tensorflow as tf

IMAGE_SIZE = 512
SEED = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)

def load_coco_records(coco_json_path: Path, images_dir: Path):
    coco = json.load(open(coco_json_path, "r", encoding="utf-8"))
    id2img = {img["id"]: img for img in coco.get("images", [])}
    anns_by_img = {}
    for ann in coco.get("annotations", []):
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    records = []
    for img_id, info in id2img.items():
        p = images_dir / info["file_name"]
        if not p.exists():
            continue

        boxes = []
        for a in anns_by_img.get(img_id, []):
            bbox = a.get("bbox", None)
            if bbox is None or len(bbox) < 4:
                continue
            x, y, w, h = map(float, bbox[:4])
            x1, y1, x2, y2 = x, y, x + w, y + h
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])

        if len(boxes) == 0:
            continue

        records.append({
            "image_path": str(p),
            "width": int(info.get("width", 0) or 0),
            "height": int(info.get("height", 0) or 0),
            "boxes": np.array(boxes, dtype=np.float32),
        })
    return records

def boxes_to_mask(boxes, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    for (x1, y1, x2, y2) in boxes:
        x1i = int(max(0, np.floor(x1)))
        y1i = int(max(0, np.floor(y1)))
        x2i = int(min(width, np.ceil(x2)))
        y2i = int(min(height, np.ceil(y2)))
        if x2i > x1i and y2i > y1i:
            mask[y1i:y2i, x1i:x2i] = 1
    return mask[..., None] 

def make_segmentation_dataset(records, batch_size=4, shuffle=True, training=True):
    output_signature = {
        "image_path": tf.TensorSpec(shape=(), dtype=tf.string),
        "boxes": tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
        "height": tf.TensorSpec(shape=(), dtype=tf.int32),
        "width": tf.TensorSpec(shape=(), dtype=tf.int32),
    }

    def gen():
        for r in records:
            yield {
                "image_path": r["image_path"],
                "boxes": r["boxes"].astype(np.float32),
                "height": int(r["height"]),
                "width": int(r["width"]),
            }

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    if shuffle:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)

    def _load(item):
        path = item["image_path"]
        boxes = item["boxes"]
        h = item["height"]
        w = item["width"]
        img_bytes = tf.io.read_file(path)
        img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.convert_image_dtype(img, tf.float32)
        def _make_mask_py(boxes_np, h_np, w_np):
            m = boxes_to_mask(boxes_np, int(h_np), int(w_np))
            return m.astype(np.float32)

        mask = tf.numpy_function(
            _make_mask_py,
            [boxes, h, w],
            Tout=tf.float32
        )
        mask.set_shape([None, None, 1])
        if training:
            flip = tf.less(tf.random.uniform([]), 0.5)
            img = tf.cond(flip, lambda: tf.image.flip_left_right(img), lambda: img)
            mask = tf.cond(flip, lambda: tf.image.flip_left_right(mask), lambda: mask)
        img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE], antialias=True)
        mask = tf.image.resize(mask, [IMAGE_SIZE, IMAGE_SIZE], method="nearest")
        return img, mask
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
