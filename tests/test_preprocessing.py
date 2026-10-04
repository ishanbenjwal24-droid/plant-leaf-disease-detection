from io import BytesIO

import pytest
from PIL import Image
import pandas as pd

from backend.app.image_validation import ImageValidationError, decode_image
from training import common
from training import prepare_data
from training.prepare_data import sha256_file, split_one_class


def png_bytes(size=(32, 24)):
    buffer = BytesIO()
    Image.new("RGB", size, (40, 120, 60)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_decode_image_returns_rgb_image():
    image = decode_image(png_bytes())
    assert image.mode == "RGB"
    assert image.size == (32, 24)


def test_decode_image_rejects_corrupt_content():
    with pytest.raises(ImageValidationError, match="not a readable image"):
        decode_image(b"this is not an image")


def test_decode_image_checks_dimensions():
    with pytest.raises(ImageValidationError, match="dimensions"):
        decode_image(png_bytes((80, 24)), max_image_side=64)


def test_split_is_reproducible_and_disjoint():
    items = list(range(200))
    first = split_one_class(items, __import__("random").Random(42))
    second = split_one_class(items, __import__("random").Random(42))
    assert first == second
    assert len(first["train"]) == 140
    assert len(first["validation"]) == 30
    assert len(first["test"]) == 30
    assert not set(first["train"]) & set(first["validation"])
    assert not set(first["train"]) & set(first["test"])
    assert not set(first["validation"]) & set(first["test"])


def test_dataset_preparation_excludes_exact_duplicates(tmp_path, monkeypatch):
    dataset_root = tmp_path / "dataset"
    for class_name, base_color in (("Apple___healthy", 40), ("Apple___scab", 120)):
        class_dir = dataset_root / class_name
        class_dir.mkdir(parents=True)
        for index in range(4):
            Image.new("RGB", (24, 24), (base_color + index, 90, 50)).save(
                class_dir / f"image_{index}.png"
            )
        (class_dir / "broken.jpg").write_bytes(b"not a real image")
    duplicate_source = dataset_root / "Apple___healthy" / "image_0.png"
    duplicate_target = dataset_root / "Apple___healthy" / "copy.png"
    duplicate_target.write_bytes(duplicate_source.read_bytes())

    data_output = tmp_path / "output" / "data"
    artifacts_output = tmp_path / "output" / "artifacts"
    monkeypatch.setattr(prepare_data, "DATA_DIR", data_output)
    monkeypatch.setattr(prepare_data, "ARTIFACTS_DIR", artifacts_output)
    monkeypatch.setattr(common, "ARTIFACTS_DIR", artifacts_output)
    monkeypatch.setattr(common, "METADATA_PATH", artifacts_output / "metadata.json")

    prepare_data.main(["--data-dir", str(dataset_root), "--max-per-class", "20"])

    manifest = pd.read_csv(data_output / "splits.csv")
    duplicates = pd.read_csv(data_output / "duplicates.csv")
    unreadable = pd.read_csv(data_output / "unreadable_files.csv")
    config = __import__("json").loads((data_output / "dataset_config.json").read_text())
    assert len(duplicates) == 2
    assert len(unreadable) == 2
    assert config["num_images"] == 7
    hashes = [sha256_file(dataset_root / rel_path) for rel_path in manifest["path"]]
    assert len(hashes) == len(set(hashes))
