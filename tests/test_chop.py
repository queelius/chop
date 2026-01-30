"""Tests for chop CLI with multi-image composition model (v0.3.0)."""

from __future__ import annotations

import io
import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from PIL import Image

from chop.operations import (
    parse_size,
    parse_crop,
    op_resize,
    op_crop,
    op_rotate,
    op_flip,
    op_pad,
    op_border,
    op_fit,
    op_fill,
    op_hstack,
    op_vstack,
    op_overlay,
    op_tile,
    op_grid,
    apply_operation,
    OPERATIONS,
    COMPOSITION_OPS,
)
from chop.pipeline import (
    PipelineState,
    load_image,
    image_to_arrays,
    execute_composition,
)
from chop.output import handle_output


@pytest.fixture
def test_image() -> Image.Image:
    """Create a simple test image."""
    return Image.new("RGBA", (100, 80), color=(128, 64, 32, 255))


@pytest.fixture
def test_image_file(test_image: Image.Image) -> str:
    """Save test image to temp file and return path."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        test_image.save(f.name)
        return f.name


@pytest.fixture
def second_image_file() -> str:
    """Create a second test image file (different size/color)."""
    img = Image.new("RGBA", (50, 40), color=(255, 0, 0, 255))
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img.save(f.name)
        return f.name


class TestParseSize:
    """Tests for parse_size function."""

    def test_percentage(self):
        assert parse_size("50%", (100, 80)) == (50, 40)
        assert parse_size("200%", (100, 80)) == (200, 160)

    def test_exact_dimensions(self):
        assert parse_size("800x600", (100, 80)) == (800, 600)
        assert parse_size("50x50", (100, 80)) == (50, 50)

    def test_width_only(self):
        assert parse_size("w50", (100, 80)) == (50, 40)
        assert parse_size("w200", (100, 80)) == (200, 160)

    def test_height_only(self):
        assert parse_size("h40", (100, 80)) == (50, 40)
        assert parse_size("h160", (100, 80)) == (200, 160)


class TestParseCrop:
    """Tests for parse_crop function."""

    def test_pixels(self):
        result = parse_crop(["10", "20", "50", "40"], (100, 80))
        assert result == (10, 20, 50, 40)

    def test_percentages(self):
        result = parse_crop(["10%", "25%", "50%", "50%"], (100, 80))
        assert result == (10, 20, 50, 40)

    def test_invalid_args(self):
        with pytest.raises(ValueError):
            parse_crop(["10", "20", "50"], (100, 80))


class TestOperations:
    """Tests for image operations."""

    def test_resize(self, test_image: Image.Image):
        result = op_resize(test_image, "50%")
        assert result.size == (50, 40)

    def test_crop(self, test_image: Image.Image):
        result = op_crop(test_image, 10, 10, 50, 40)
        assert result.size == (50, 40)

    def test_rotate_90(self, test_image: Image.Image):
        result = op_rotate(test_image, 90)
        assert result.size[0] == test_image.size[1]
        assert result.size[1] == test_image.size[0]

    def test_flip_horizontal(self, test_image: Image.Image):
        result = op_flip(test_image, "h")
        assert result.size == test_image.size

    def test_flip_vertical(self, test_image: Image.Image):
        result = op_flip(test_image, "v")
        assert result.size == test_image.size

    def test_flip_invalid(self, test_image: Image.Image):
        with pytest.raises(ValueError):
            op_flip(test_image, "x")

    def test_pad_uniform(self, test_image: Image.Image):
        result = op_pad(test_image, 10)
        assert result.size == (120, 100)

    def test_pad_vertical_horizontal(self, test_image: Image.Image):
        result = op_pad(test_image, 10, 20)
        assert result.size == (140, 100)

    def test_pad_all_sides(self, test_image: Image.Image):
        result = op_pad(test_image, 10, 20, 30, 40)
        assert result.size == (160, 120)

    def test_pad_with_color(self, test_image: Image.Image):
        result = op_pad(test_image, 10, color="red")
        assert result.size == (120, 100)
        pixel = result.getpixel((0, 0))
        assert pixel[:3] == (255, 0, 0)

    def test_pad_transparent(self, test_image: Image.Image):
        result = op_pad(test_image, 10)
        pixel = result.getpixel((0, 0))
        assert pixel[3] == 0

    def test_border(self, test_image: Image.Image):
        result = op_border(test_image, 5)
        assert result.size == (110, 90)

    def test_border_with_color(self, test_image: Image.Image):
        result = op_border(test_image, 5, color="#00ff00")
        pixel = result.getpixel((0, 0))
        assert pixel[:3] == (0, 255, 0)

    def test_fit_landscape_to_square(self, test_image: Image.Image):
        result = op_fit(test_image, "50x50")
        assert result.size == (50, 40)

    def test_fit_preserves_aspect(self, test_image: Image.Image):
        result = op_fit(test_image, "200x100")
        assert result.size == (125, 100)

    def test_fill_crops_to_exact_size(self, test_image: Image.Image):
        result = op_fill(test_image, "50x50")
        assert result.size == (50, 50)

    def test_fill_center_crops(self, test_image: Image.Image):
        result = op_fill(test_image, "80x80")
        assert result.size == (80, 80)


class TestOperationsRegistry:
    """Tests for the operations registry and apply_operation."""

    def test_registry_contains_basic_ops(self):
        assert "resize" in OPERATIONS
        assert "crop" in OPERATIONS
        assert "rotate" in OPERATIONS
        assert "flip" in OPERATIONS
        assert "pad" in OPERATIONS
        assert "border" in OPERATIONS
        assert "fit" in OPERATIONS
        assert "fill" in OPERATIONS

    def test_composition_ops_set(self):
        """Test that COMPOSITION_OPS contains the expected ops."""
        assert COMPOSITION_OPS == {"hstack", "vstack", "overlay", "grid"}

    def test_composition_ops_not_in_operations(self):
        """Test that composition ops are NOT in the single-image OPERATIONS dict."""
        for op in COMPOSITION_OPS:
            assert op not in OPERATIONS

    def test_apply_operation_resize(self, test_image: Image.Image):
        result = apply_operation(test_image, "resize", "50%")
        assert result.size == (50, 40)

    def test_apply_operation_rotate(self, test_image: Image.Image):
        result = apply_operation(test_image, "rotate", 90)
        assert result.size[0] == test_image.size[1]

    def test_apply_operation_pad(self, test_image: Image.Image):
        result = apply_operation(test_image, "pad", 10, color="red")
        assert result.size == (120, 100)

    def test_apply_operation_border(self, test_image: Image.Image):
        result = apply_operation(test_image, "border", 5, color="blue")
        assert result.size == (110, 90)

    def test_apply_operation_fit(self, test_image: Image.Image):
        result = apply_operation(test_image, "fit", "50x50")
        assert result.size == (50, 40)

    def test_apply_operation_fill(self, test_image: Image.Image):
        result = apply_operation(test_image, "fill", "50x50")
        assert result.size == (50, 50)

    def test_apply_operation_unknown(self, test_image: Image.Image):
        with pytest.raises(ValueError, match="Unknown operation"):
            apply_operation(test_image, "unknown_op")


class TestPipeline:
    """Tests for pipeline utilities with v3 model."""

    def test_state_creation_empty(self):
        """Test creating an empty PipelineState (no path field)."""
        state = PipelineState()
        assert state.ops == []
        assert state.metadata == {}

    def test_add_op(self):
        """Test adding operations to state."""
        state = PipelineState()
        state.add_op("resize", "50%")
        state.add_op("pad", 10, color="red")

        assert len(state.ops) == 2
        assert state.ops[0] == ("resize", ("50%",), {})
        assert state.ops[1] == ("pad", (10,), {"color": "red"})

    def test_state_to_json_and_back(self, test_image_file: str):
        """Test JSON serialization roundtrip."""
        state = PipelineState(metadata={"test": "value"})
        state.add_op("load", test_image_file)
        state.add_op("resize", "50%")
        state.add_op("pad", 10, color="red")

        json_str = state.to_json()
        restored = PipelineState.from_json(json_str)

        assert restored.metadata["test"] == "value"
        assert len(restored.ops) == 3
        assert restored.ops[0][0] == "load"
        assert restored.ops[1] == ("resize", ("50%",), {})
        assert restored.ops[2] == ("pad", (10,), {"color": "red"})

    def test_json_format_v3(self, test_image_file: str):
        """Test that JSON format is v3 with no path field."""
        state = PipelineState()
        state.add_op("load", test_image_file)
        state.add_op("resize", "50%")

        json_str = state.to_json()
        data = json.loads(json_str)

        assert data["version"] == 3
        assert "path" not in data
        assert "image" not in data
        assert len(data["ops"]) == 2
        assert len(json_str) < 1000

    def test_has_load(self, test_image_file: str):
        """Test has_load detection."""
        state = PipelineState()
        assert not state.has_load()

        state.add_op("resize", "50%")
        assert not state.has_load()

        state.add_op("load", test_image_file)
        assert state.has_load()

    def test_materialize_loads_and_applies(self, test_image_file: str):
        """Test that materialize loads image and applies ops."""
        state = PipelineState()
        state.add_op("load", test_image_file)
        state.add_op("resize", "50%")

        image = state.materialize()

        assert isinstance(image, Image.Image)
        assert image.size == (50, 40)

    def test_materialize_multiple_ops(self, test_image_file: str):
        """Test materializing with multiple operations."""
        state = PipelineState()
        state.add_op("load", test_image_file)
        state.add_op("resize", "50%")
        state.add_op("rotate", 90)

        image = state.materialize()
        assert image.size == (40, 50)

    def test_materialize_no_ops_raises(self):
        """Test that empty pipeline raises."""
        state = PipelineState()
        with pytest.raises(ValueError, match="Pipeline produced no image"):
            state.materialize()

    def test_load_image_file(self, test_image_file: str):
        """Test load_image with file path."""
        image = load_image(test_image_file)
        assert image.mode == "RGBA"
        assert image.size == (100, 80)

    def test_image_to_arrays(self, test_image: Image.Image):
        """Test image_to_arrays conversion."""
        bitmap, colors = image_to_arrays(test_image)

        assert bitmap.shape == (80, 100)
        assert colors.shape == (80, 100, 3)
        assert bitmap.dtype == np.float32
        assert colors.dtype == np.float32
        assert 0 <= bitmap.min() <= bitmap.max() <= 1
        assert 0 <= colors.min() <= colors.max() <= 1


class TestV2Rejected:
    """Test that v2 JSON format is rejected."""

    def test_v2_json_rejected(self):
        """Test that version 2 format raises with clear error."""
        v2_json = json.dumps({
            "version": 2,
            "path": "photo.jpg",
            "ops": [["resize", ["50%"], {}]],
            "metadata": {},
        })
        with pytest.raises(ValueError, match="Unsupported pipeline version: 2"):
            PipelineState.from_json(v2_json)

    def test_v1_json_rejected(self):
        """Test that version 1 format is rejected."""
        v1_json = '{"version": 1, "image": {}, "history": []}'
        with pytest.raises(ValueError, match="Unsupported pipeline version: 1"):
            PipelineState.from_json(v1_json)

    def test_no_version_rejected(self):
        """Test that missing version is rejected (defaults to 1)."""
        no_version = '{"path": "photo.jpg", "ops": []}'
        with pytest.raises(ValueError, match="Unsupported pipeline version"):
            PipelineState.from_json(no_version)


class TestLoadAsOp:
    """Test that load works as an operation in the ops list."""

    def test_load_is_first_op(self, test_image_file: str):
        """Test load as first operation creates context."""
        state = PipelineState()
        state.add_op("load", test_image_file)

        image = state.materialize()
        assert image.size == (100, 80)

    def test_load_in_middle(self, test_image_file: str, second_image_file: str):
        """Test load in middle of pipeline adds to context."""
        state = PipelineState()
        state.add_op("load", test_image_file)
        state.add_op("resize", "50%")
        state.add_op("load", second_image_file)

        # Cursor should be on the second image (50x40)
        image = state.materialize()
        assert image.size == (50, 40)

    def test_load_json_roundtrip(self, test_image_file: str):
        """Test load op survives JSON serialization."""
        state = PipelineState()
        state.add_op("load", test_image_file, **{"as": "bg"})
        state.add_op("resize", "50%")

        json_str = state.to_json()
        restored = PipelineState.from_json(json_str)

        assert restored.ops[0] == ("load", (test_image_file,), {"as": "bg"})
        assert restored.ops[1] == ("resize", ("50%",), {})


class TestAutoLabeling:
    """Test auto-labeling of loaded images."""

    def test_first_load_is_img(self, test_image_file: str):
        """Test first auto-label is 'img'."""
        state = PipelineState()
        state.add_op("load", test_image_file)
        state.add_op("select", "img")

        # Should not raise — label exists
        image = state.materialize()
        assert image.size == (100, 80)

    def test_second_load_is_img2(self, test_image_file: str, second_image_file: str):
        """Test second auto-label is 'img2'."""
        state = PipelineState()
        state.add_op("load", test_image_file)
        state.add_op("load", second_image_file)
        state.add_op("select", "img2")

        image = state.materialize()
        assert image.size == (50, 40)

    def test_explicit_as_does_not_consume_counter(self, test_image_file: str, second_image_file: str):
        """Test that --as doesn't consume auto counter."""
        state = PipelineState()
        state.add_op("load", test_image_file, **{"as": "background"})
        state.add_op("load", second_image_file)
        # Second load should be "img" (counter starts at 1, first load used explicit)
        state.add_op("select", "img")

        image = state.materialize()
        assert image.size == (50, 40)

    def test_three_auto_loads(self, test_image_file: str):
        """Test auto-labeling: img, img2, img3."""
        state = PipelineState()
        state.add_op("load", test_image_file)
        state.add_op("load", test_image_file)
        state.add_op("load", test_image_file)

        # Verify all three labels exist
        state.add_op("select", "img")
        state.add_op("select", "img2")
        state.add_op("select", "img3")
        state.materialize()  # Should not raise


class TestMultiImagePipeline:
    """Test multi-image pipeline with context and cursor."""

    def test_two_loads_cursor_on_second(self, test_image_file: str, second_image_file: str):
        """Test cursor is on the last loaded image."""
        state = PipelineState()
        state.add_op("load", test_image_file)      # img: 100x80
        state.add_op("load", second_image_file)     # img2: 50x40

        image = state.materialize()
        assert image.size == (50, 40)

    def test_on_targets_specific_image(self, test_image_file: str, second_image_file: str):
        """Test --on targets a specific labeled image."""
        state = PipelineState()
        state.add_op("load", test_image_file, **{"as": "bg"})   # bg: 100x80
        state.add_op("load", second_image_file, **{"as": "fg"}) # fg: 50x40
        state.add_op("resize", "50%", on="bg")  # Resize bg → 50x40

        # Cursor is still on fg (last load), but bg was resized
        state.add_op("select", "bg")
        image = state.materialize()
        assert image.size == (50, 40)

    def test_as_names_load_result(self, test_image_file: str):
        """Test --as names the loaded image."""
        state = PipelineState()
        state.add_op("load", test_image_file, **{"as": "photo"})
        state.add_op("select", "photo")

        image = state.materialize()
        assert image.size == (100, 80)

    def test_transform_without_load_raises(self):
        """Test transform without any image raises."""
        state = PipelineState()
        state.add_op("resize", "50%")

        with pytest.raises(ValueError, match="No current image"):
            state.materialize()

    def test_label_collision_replaces(self, test_image_file: str, second_image_file: str):
        """Test that loading with same label replaces the image."""
        state = PipelineState()
        state.add_op("load", test_image_file, **{"as": "x"})    # x: 100x80
        state.add_op("load", second_image_file, **{"as": "x"})  # x: 50x40

        image = state.materialize()
        assert image.size == (50, 40)


class TestCompositionOperations:
    """Tests for image composition operations (direct function calls)."""

    @pytest.fixture
    def small_image(self) -> Image.Image:
        return Image.new("RGBA", (50, 40), color=(255, 0, 0, 255))

    @pytest.fixture
    def large_image(self) -> Image.Image:
        return Image.new("RGBA", (100, 80), color=(0, 255, 0, 255))

    def test_hstack_same_size(self, test_image: Image.Image):
        result = op_hstack(test_image, test_image)
        assert result.size == (200, 80)

    def test_hstack_different_heights(self, small_image, large_image):
        result = op_hstack(small_image, large_image)
        assert result.size == (150, 80)

    def test_hstack_align_top(self, small_image, large_image):
        result = op_hstack(small_image, large_image, align="top")
        assert result.size == (150, 80)

    def test_hstack_align_bottom(self, small_image, large_image):
        result = op_hstack(small_image, large_image, align="bottom")
        assert result.size == (150, 80)

    def test_vstack_same_size(self, test_image: Image.Image):
        result = op_vstack(test_image, test_image)
        assert result.size == (100, 160)

    def test_vstack_different_widths(self, small_image, large_image):
        result = op_vstack(small_image, large_image)
        assert result.size == (100, 120)

    def test_vstack_align_left(self, small_image, large_image):
        result = op_vstack(small_image, large_image, align="left")
        assert result.size == (100, 120)

    def test_vstack_align_right(self, small_image, large_image):
        result = op_vstack(small_image, large_image, align="right")
        assert result.size == (100, 120)

    def test_overlay_basic(self, large_image, small_image):
        result = op_overlay(large_image, small_image, x=10, y=10)
        assert result.size == large_image.size

    def test_overlay_with_opacity(self, large_image, small_image):
        result = op_overlay(large_image, small_image, x=10, y=10, opacity=0.5)
        assert result.size == large_image.size

    def test_overlay_paste_mode(self, large_image, small_image):
        result = op_overlay(large_image, small_image, x=10, y=10, paste=True)
        assert result.size == large_image.size

    def test_tile(self, test_image: Image.Image):
        result = op_tile(test_image, cols=3, rows=2)
        assert result.size == (300, 160)

    def test_tile_1x1(self, test_image: Image.Image):
        result = op_tile(test_image, cols=1, rows=1)
        assert result.size == test_image.size

    def test_grid_same_size(self, test_image: Image.Image):
        others = [test_image.copy(), test_image.copy(), test_image.copy()]
        result = op_grid(test_image, others, cols=2)
        assert result.size == (200, 160)

    def test_grid_uneven(self, test_image: Image.Image):
        others = [test_image.copy(), test_image.copy()]
        result = op_grid(test_image, others, cols=2)
        assert result.size == (200, 160)

    def test_grid_resizes_images(self, large_image, small_image):
        result = op_grid(large_image, [small_image], cols=2)
        assert result.size == (200, 80)


class TestCompositionWithLabels:
    """Test composition operations via labeled context in materialize()."""

    def test_hstack_two_labels(self, test_image_file: str, second_image_file: str):
        """Test hstack with explicit label args."""
        state = PipelineState()
        state.add_op("load", test_image_file, **{"as": "a"})    # 100x80
        state.add_op("load", second_image_file, **{"as": "b"})  # 50x40
        state.add_op("hstack", "a", "b")

        image = state.materialize()
        assert image.size == (150, 80)

    def test_vstack_two_labels(self, test_image_file: str, second_image_file: str):
        """Test vstack with explicit label args."""
        state = PipelineState()
        state.add_op("load", test_image_file, **{"as": "a"})
        state.add_op("load", second_image_file, **{"as": "b"})
        state.add_op("vstack", "a", "b")

        image = state.materialize()
        assert image.size == (100, 120)

    def test_composition_default_all_images(self, test_image_file: str, second_image_file: str):
        """Test composition with no labels uses all context images."""
        state = PipelineState()
        state.add_op("load", test_image_file, **{"as": "a"})    # 100x80
        state.add_op("load", second_image_file, **{"as": "b"})  # 50x40
        state.add_op("hstack")  # No labels → all images

        image = state.materialize()
        assert image.size == (150, 80)

    def test_composition_result_label_default(self, test_image_file: str, second_image_file: str):
        """Test composition result defaults to '_' label."""
        state = PipelineState()
        state.add_op("load", test_image_file, **{"as": "a"})
        state.add_op("load", second_image_file, **{"as": "b"})
        state.add_op("hstack", "a", "b")
        # Cursor should now be on '_'
        state.add_op("select", "_")

        image = state.materialize()
        assert image.size == (150, 80)

    def test_composition_result_custom_label(self, test_image_file: str, second_image_file: str):
        """Test --as names composition result."""
        state = PipelineState()
        state.add_op("load", test_image_file, **{"as": "a"})
        state.add_op("load", second_image_file, **{"as": "b"})
        state.add_op("hstack", "a", "b", **{"as": "combined"})
        state.add_op("select", "combined")

        image = state.materialize()
        assert image.size == (150, 80)

    def test_overlay_labels(self, test_image_file: str, second_image_file: str):
        """Test overlay with label args."""
        state = PipelineState()
        state.add_op("load", test_image_file, **{"as": "bg"})
        state.add_op("load", second_image_file, **{"as": "fg"})
        state.add_op("overlay", "bg", "fg", x=10, y=10)

        image = state.materialize()
        assert image.size == (100, 80)

    def test_grid_labels(self, test_image_file: str):
        """Test grid with labels."""
        state = PipelineState()
        state.add_op("load", test_image_file, **{"as": "a"})
        state.add_op("load", test_image_file, **{"as": "b"})
        state.add_op("load", test_image_file, **{"as": "c"})
        state.add_op("load", test_image_file, **{"as": "d"})
        state.add_op("grid", "a", "b", "c", "d", cols=2)

        image = state.materialize()
        assert image.size == (200, 160)

    def test_composition_missing_label_raises(self, test_image_file: str):
        """Test composition with unknown label raises."""
        state = PipelineState()
        state.add_op("load", test_image_file, **{"as": "a"})
        state.add_op("hstack", "a", "nonexistent")

        with pytest.raises(ValueError, match="Label 'nonexistent' not found"):
            state.materialize()

    def test_composition_excludes_underscore(self, test_image_file: str, second_image_file: str):
        """Test default composition excludes '_' label."""
        state = PipelineState()
        state.add_op("load", test_image_file, **{"as": "a"})     # 100x80
        state.add_op("load", second_image_file, **{"as": "b"})   # 50x40
        state.add_op("hstack", "a", "b")  # Result stored as '_'
        # Now hstack with no args should use a, b (not _)
        state.add_op("hstack", **{"as": "final"})

        image = state.materialize()
        assert image.size == (150, 80)


class TestSelectAndDup:
    """Test select and dup operations."""

    def test_select_switches_cursor(self, test_image_file: str, second_image_file: str):
        """Test select switches the cursor."""
        state = PipelineState()
        state.add_op("load", test_image_file, **{"as": "a"})    # 100x80
        state.add_op("load", second_image_file, **{"as": "b"})  # 50x40
        state.add_op("select", "a")

        image = state.materialize()
        assert image.size == (100, 80)

    def test_select_unknown_label_raises(self, test_image_file: str):
        """Test select with unknown label raises."""
        state = PipelineState()
        state.add_op("load", test_image_file)
        state.add_op("select", "nonexistent")

        with pytest.raises(ValueError, match="Label 'nonexistent' not found"):
            state.materialize()

    def test_dup_copies_image(self, test_image_file: str):
        """Test dup creates a copy."""
        state = PipelineState()
        state.add_op("load", test_image_file, **{"as": "orig"})
        state.add_op("dup", "orig", "copy")
        state.add_op("resize", "50%", on="copy")
        state.add_op("select", "copy")

        image = state.materialize()
        assert image.size == (50, 40)

    def test_dup_does_not_affect_original(self, test_image_file: str):
        """Test dup copy is independent of original."""
        state = PipelineState()
        state.add_op("load", test_image_file, **{"as": "orig"})
        state.add_op("dup", "orig", "copy")
        state.add_op("resize", "50%", on="copy")
        state.add_op("select", "orig")

        image = state.materialize()
        assert image.size == (100, 80)

    def test_dup_unknown_source_raises(self, test_image_file: str):
        """Test dup with unknown source raises."""
        state = PipelineState()
        state.add_op("load", test_image_file)
        state.add_op("dup", "nonexistent", "copy")

        with pytest.raises(ValueError, match="Label 'nonexistent' not found"):
            state.materialize()


class TestUnboundPipeline:
    """Test unbound pipelines (ops without load)."""

    def test_unbound_json_roundtrip(self):
        """Test unbound pipeline serializes and deserializes."""
        state = PipelineState()
        state.add_op("resize", "50%")
        state.add_op("pad", 10, color="red")

        json_str = state.to_json()
        restored = PipelineState.from_json(json_str)

        assert not restored.has_load()
        assert len(restored.ops) == 2
        assert restored.ops[0] == ("resize", ("50%",), {})

    def test_unbound_materialize_fails(self):
        """Test materializing unbound pipeline raises."""
        state = PipelineState()
        state.add_op("resize", "50%")

        with pytest.raises(ValueError, match="No current image"):
            state.materialize()

    def test_unbound_has_no_load(self):
        """Test has_load is False for unbound pipeline."""
        state = PipelineState()
        state.add_op("resize", "50%")
        state.add_op("pad", 10)
        assert not state.has_load()


class TestOutputHandling:
    """Tests for centralized output handling."""

    @pytest.fixture
    def bound_state(self, test_image_file: str) -> PipelineState:
        """Create a bound pipeline state."""
        state = PipelineState()
        state.add_op("load", test_image_file)
        return state

    @pytest.fixture
    def unbound_state(self) -> PipelineState:
        """Create an unbound pipeline state."""
        state = PipelineState()
        state.add_op("resize", "50%")
        return state

    def test_json_flag_forces_json(self, bound_state: PipelineState):
        """Test -j flag forces JSON output."""
        args = mock.Mock(json=True, output=None)

        with mock.patch("chop.pipeline.write_pipeline_output") as mock_write:
            with mock.patch("sys.stdout.isatty", return_value=True):
                handle_output(bound_state, args)
                mock_write.assert_called_once_with(bound_state)

    def test_output_flag_saves_file(self, bound_state: PipelineState):
        """Test -o flag materializes and saves to file."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            args = mock.Mock(json=False, output=f.name)
            handle_output(bound_state, args)
            assert Path(f.name).exists()
            Path(f.name).unlink()

    def test_piped_outputs_json(self, bound_state: PipelineState):
        """Test piped output writes JSON."""
        args = mock.Mock(json=False, output=None)

        with mock.patch("sys.stdout.isatty", return_value=False):
            with mock.patch("chop.pipeline.write_pipeline_output") as mock_write:
                handle_output(bound_state, args)
                mock_write.assert_called_once_with(bound_state)

    def test_unbound_json_flag_works(self, unbound_state: PipelineState):
        """Test -j flag works on unbound pipeline."""
        args = mock.Mock(json=True, output=None)

        with mock.patch("chop.pipeline.write_pipeline_output") as mock_write:
            handle_output(unbound_state, args)
            mock_write.assert_called_once_with(unbound_state)

    def test_unbound_output_flag_errors(self, unbound_state: PipelineState):
        """Test -o flag on unbound pipeline exits with error."""
        args = mock.Mock(json=False, output="/tmp/out.png")

        with pytest.raises(SystemExit):
            handle_output(unbound_state, args)

    def test_tty_unbound_shows_program(self, unbound_state: PipelineState, capsys):
        """Test TTY + unbound shows program listing."""
        args = mock.Mock(json=False, output=None)

        with mock.patch("sys.stdout.isatty", return_value=True):
            handle_output(unbound_state, args)

        captured = capsys.readouterr()
        assert "Program:" in captured.err
        assert "resize 50%" in captured.err


class TestSaveCommand:
    """Tests for save command."""

    def test_save_command_handler(self, test_image_file: str):
        """Test cmd_save function materializes and saves."""
        from chop.cli import cmd_save

        state = PipelineState()
        state.add_op("load", test_image_file)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            args = mock.Mock(path=f.name, format=None)

            with mock.patch("chop.cli.read_pipeline_input", return_value=state):
                cmd_save(args)

            assert Path(f.name).exists()
            saved = Image.open(f.name)
            assert saved.size == (100, 80)
            Path(f.name).unlink()

    def test_save_unbound_raises(self):
        """Test save on unbound pipeline raises."""
        from chop.cli import cmd_save

        state = PipelineState()
        state.add_op("resize", "50%")
        args = mock.Mock(path="out.png", format=None)

        with mock.patch("chop.cli.read_pipeline_input", return_value=state):
            with pytest.raises(ValueError, match="pipeline has no load"):
                cmd_save(args)

    def test_save_empty_raises(self):
        """Test save on empty pipeline raises."""
        from chop.cli import cmd_save

        state = PipelineState()
        args = mock.Mock(path="out.png", format=None)

        with mock.patch("chop.cli.read_pipeline_input", return_value=state):
            with pytest.raises(ValueError, match="Nothing to save"):
                cmd_save(args)


class TestSaveToStdout:
    """Tests for save to stdout."""

    def test_save_stdout_requires_format(self, test_image_file: str):
        """Test save to '-' requires --format."""
        from chop.cli import cmd_save

        state = PipelineState()
        state.add_op("load", test_image_file)
        args = mock.Mock(path="-", format=None)

        with mock.patch("chop.cli.read_pipeline_input", return_value=state):
            with pytest.raises(ValueError, match="--format is required"):
                cmd_save(args)

    def test_save_stdout_with_format(self, test_image_file: str):
        """Test save to stdout with format specified."""
        from chop.cli import cmd_save

        state = PipelineState()
        state.add_op("load", test_image_file)
        args = mock.Mock(path="-", format="png")

        buf = io.BytesIO()
        with mock.patch("chop.cli.read_pipeline_input", return_value=state):
            with mock.patch("sys.stdout") as mock_stdout:
                mock_stdout.buffer = buf
                cmd_save(args)

        buf.seek(0)
        saved = Image.open(buf)
        assert saved.size == (100, 80)


class TestInfoCommand:
    """Tests for info command."""

    def test_info_bound_pipeline(self, test_image_file: str, capsys):
        """Test info on bound pipeline shows image info."""
        from chop.cli import cmd_info

        state = PipelineState()
        state.add_op("load", test_image_file)
        state.add_op("resize", "50%")
        args = mock.Mock()

        with mock.patch("chop.cli.read_pipeline_input", return_value=state):
            cmd_info(args)

        captured = capsys.readouterr()
        assert "Cursor image: 50x40" in captured.err

    def test_info_unbound_pipeline(self, capsys):
        """Test info on unbound pipeline shows program listing."""
        from chop.cli import cmd_info

        state = PipelineState()
        state.add_op("resize", "50%")
        state.add_op("pad", 10)
        args = mock.Mock()

        with mock.patch("chop.cli.read_pipeline_input", return_value=state):
            cmd_info(args)

        captured = capsys.readouterr()
        assert "Unbound program" in captured.err
        assert "resize 50%" in captured.err

    def test_info_empty_pipeline(self, capsys):
        """Test info on empty pipeline."""
        from chop.cli import cmd_info

        state = PipelineState()
        args = mock.Mock()

        with mock.patch("chop.cli.read_pipeline_input", return_value=state):
            cmd_info(args)

        captured = capsys.readouterr()
        assert "Empty pipeline" in captured.err

    def test_info_multi_load_shows_count(self, test_image_file: str, second_image_file: str, capsys):
        """Test info shows image count for multi-load pipeline."""
        from chop.cli import cmd_info

        state = PipelineState()
        state.add_op("load", test_image_file)
        state.add_op("load", second_image_file)
        args = mock.Mock()

        with mock.patch("chop.cli.read_pipeline_input", return_value=state):
            cmd_info(args)

        captured = capsys.readouterr()
        assert "Images loaded: 2" in captured.err


class TestApplyCommand:
    """Tests for the apply command (JSON programs)."""

    def test_apply_json_file(self, test_image_file: str):
        """Test apply with JSON program file."""
        from chop.cli import cmd_apply

        # Create program JSON
        program = PipelineState()
        program.add_op("resize", "50%")
        program.add_op("pad", 10)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(program.to_json())
            prog_path = f.name

        state = PipelineState()
        state.add_op("load", test_image_file)
        args = mock.Mock(program=prog_path)

        try:
            with mock.patch("chop.cli.read_pipeline_input", return_value=state):
                result = cmd_apply(args)

            assert len(result.ops) == 3  # load + resize + pad
            assert result.ops[1] == ("resize", ("50%",), {})
            assert result.ops[2] == ("pad", (10,), {})
        finally:
            Path(prog_path).unlink()

    def test_apply_appends_to_existing_ops(self, test_image_file: str):
        """Test apply appends to existing pipeline."""
        from chop.cli import cmd_apply

        program = PipelineState()
        program.add_op("resize", "50%")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(program.to_json())
            prog_path = f.name

        state = PipelineState()
        state.add_op("load", test_image_file)
        state.add_op("rotate", 90)
        args = mock.Mock(program=prog_path)

        try:
            with mock.patch("chop.cli.read_pipeline_input", return_value=state):
                result = cmd_apply(args)

            assert len(result.ops) == 3  # load + rotate + resize
            assert result.ops[0][0] == "load"
            assert result.ops[1] == ("rotate", (90,), {})
            assert result.ops[2] == ("resize", ("50%",), {})
        finally:
            Path(prog_path).unlink()

    def test_apply_integration_materialize(self, test_image_file: str):
        """Test applied ops work when materialized."""
        program = PipelineState()
        program.add_op("resize", "50%")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(program.to_json())
            prog_path = f.name

        try:
            state = PipelineState()
            state.add_op("load", test_image_file)

            restored = PipelineState.from_json(Path(prog_path).read_text())
            for name, op_args, kwargs in restored.ops:
                state.add_op(name, *op_args, **kwargs)

            image = state.materialize()
            assert image.size == (50, 40)
        finally:
            Path(prog_path).unlink()


class TestCLIHandlers:
    """Tests for CLI command handlers."""

    def test_cmd_load_basic(self):
        """Test cmd_load creates load op."""
        from chop.cli import cmd_load

        args = mock.Mock(source="photo.jpg", label=None)
        with mock.patch("chop.cli.read_pipeline_input", return_value=None):
            state = cmd_load(args)

        assert len(state.ops) == 1
        assert state.ops[0] == ("load", ("photo.jpg",), {})

    def test_cmd_load_with_as(self):
        """Test cmd_load with --as flag."""
        from chop.cli import cmd_load

        args = mock.Mock(source="photo.jpg", label="bg")
        with mock.patch("chop.cli.read_pipeline_input", return_value=None):
            state = cmd_load(args)

        assert state.ops[0] == ("load", ("photo.jpg",), {"as": "bg"})

    def test_cmd_resize_with_on(self):
        """Test cmd_resize with --on flag."""
        from chop.cli import cmd_resize

        args = mock.Mock(size="50%", on="fg")
        with mock.patch("chop.cli.read_pipeline_input", return_value=None):
            state = cmd_resize(args)

        assert state.ops[0] == ("resize", ("50%",), {"on": "fg"})

    def test_cmd_resize_without_on(self):
        """Test cmd_resize without --on flag (unbound)."""
        from chop.cli import cmd_resize

        args = mock.Mock(size="50%", on=None)
        with mock.patch("chop.cli.read_pipeline_input", return_value=None):
            state = cmd_resize(args)

        assert state.ops[0] == ("resize", ("50%",), {})

    def test_cmd_select(self):
        """Test cmd_select creates select op."""
        from chop.cli import cmd_select

        args = mock.Mock(label="bg")
        with mock.patch("chop.cli.read_pipeline_input", return_value=None):
            state = cmd_select(args)

        assert state.ops[0] == ("select", ("bg",), {})

    def test_cmd_dup(self):
        """Test cmd_dup creates dup op."""
        from chop.cli import cmd_dup

        args = mock.Mock(source="orig", dest="copy")
        with mock.patch("chop.cli.read_pipeline_input", return_value=None):
            state = cmd_dup(args)

        assert state.ops[0] == ("dup", ("orig", "copy"), {})

    def test_cmd_hstack_with_labels(self):
        """Test cmd_hstack with label args."""
        from chop.cli import cmd_hstack

        args = mock.Mock(images=["a", "b"], align="center", label=None)
        with mock.patch("chop.cli.read_pipeline_input", return_value=None):
            state = cmd_hstack(args)

        assert state.ops[0] == ("hstack", ("a", "b"), {"align": "center"})

    def test_cmd_hstack_no_labels(self):
        """Test cmd_hstack with no label args (default all)."""
        from chop.cli import cmd_hstack

        args = mock.Mock(images=[], align="center", label=None)
        with mock.patch("chop.cli.read_pipeline_input", return_value=None):
            state = cmd_hstack(args)

        assert state.ops[0] == ("hstack", (), {"align": "center"})

    def test_cmd_overlay_with_labels(self):
        """Test cmd_overlay with label args and position."""
        from chop.cli import cmd_overlay

        args = mock.Mock(images=["bg", "fg"], x=10, y=20, opacity=1.0, paste=False, label=None)
        with mock.patch("chop.cli.read_pipeline_input", return_value=None):
            state = cmd_overlay(args)

        assert state.ops[0] == ("overlay", ("bg", "fg"), {"x": 10, "y": 20})

    def test_cmd_grid_with_cols(self):
        """Test cmd_grid with cols and --as."""
        from chop.cli import cmd_grid

        args = mock.Mock(images=["a", "b", "c", "d"], cols=2, label="result")
        with mock.patch("chop.cli.read_pipeline_input", return_value=None):
            state = cmd_grid(args)

        assert state.ops[0] == ("grid", ("a", "b", "c", "d"), {"cols": 2, "as": "result"})

    def test_get_or_create_state_piped(self, test_image_file: str):
        """Test get_or_create_state reads piped input."""
        from chop.cli import get_or_create_state

        existing = PipelineState()
        existing.add_op("load", test_image_file)

        with mock.patch("chop.cli.read_pipeline_input", return_value=existing):
            state = get_or_create_state()

        assert len(state.ops) == 1
        assert state.ops[0][0] == "load"

    def test_get_or_create_state_no_pipe(self):
        """Test get_or_create_state creates fresh state."""
        from chop.cli import get_or_create_state

        with mock.patch("chop.cli.read_pipeline_input", return_value=None):
            state = get_or_create_state()

        assert len(state.ops) == 0


class TestLazyPipelineIntegration:
    """Integration tests for lazy pipeline behavior."""

    def test_pipeline_json_is_human_readable(self, test_image_file: str):
        """Test that pipeline JSON is small and readable."""
        state = PipelineState()
        state.add_op("load", test_image_file)
        state.add_op("resize", "50%")
        state.add_op("pad", 10)
        state.add_op("border", 5, color="red")

        json_str = state.to_json()
        data = json.loads(json_str)

        assert data["version"] == 3
        assert "path" not in data
        assert len(data["ops"]) == 4
        assert len(json_str) < 500

    def test_ops_are_recorded_not_applied(self, test_image_file: str):
        """Test that adding ops doesn't load/process the image."""
        state = PipelineState()
        state.add_op("load", test_image_file)

        for _ in range(100):
            state.add_op("resize", "99%")
            state.add_op("pad", 1)

        assert len(state.ops) == 201  # 1 load + 200 transforms

    def test_materialize_chains_operations(self, test_image_file: str):
        """Test that materialize correctly chains operations."""
        state = PipelineState()
        state.add_op("load", test_image_file)
        state.add_op("resize", "50%")
        state.add_op("rotate", 90)

        image = state.materialize()
        assert image.size == (40, 50)

    def test_full_multi_image_workflow(self, test_image_file: str, second_image_file: str):
        """Integration test: load two images, transform one, compose."""
        state = PipelineState()
        state.add_op("load", test_image_file, **{"as": "bg"})
        state.add_op("load", second_image_file, **{"as": "fg"})
        state.add_op("resize", "50%", on="bg")  # bg: 50x40
        state.add_op("hstack", "bg", "fg")       # 50+50=100 wide, max(40,40)=40 tall

        image = state.materialize()
        assert image.size == (100, 40)

    def test_dup_and_flip_mirror(self, test_image_file: str):
        """Integration test: dup, flip, hstack for mirror effect."""
        state = PipelineState()
        state.add_op("load", test_image_file, **{"as": "orig"})
        state.add_op("dup", "orig", "flipped")
        state.add_op("flip", "h", on="flipped")
        state.add_op("hstack", "orig", "flipped")

        image = state.materialize()
        assert image.size == (200, 80)

    def test_unbound_program_applied_to_image(self, test_image_file: str):
        """Integration test: create unbound program, apply to image."""
        # Create unbound program
        program = PipelineState()
        program.add_op("resize", "50%")
        program.add_op("border", 5, color="red")
        json_str = program.to_json()

        # Apply to image
        state = PipelineState()
        state.add_op("load", test_image_file)

        restored = PipelineState.from_json(json_str)
        for name, op_args, kwargs in restored.ops:
            state.add_op(name, *op_args, **kwargs)

        image = state.materialize()
        # 100x80 → resize 50% → 50x40 → border 5 → 60x50
        assert image.size == (60, 50)
