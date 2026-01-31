"""Tests for chop CLI with uniform output model (v0.4.0)."""

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
    op_brightness,
    op_contrast,
    op_saturation,
    op_sharpen,
    op_blur,
    op_grayscale,
    op_invert,
    op_trim,
    op_colorize,
    op_opacity,
    op_background,
    op_mask,
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


class TestColorOperations:
    """Tests for color/pixel operations."""

    def test_brightness_increase(self, test_image: Image.Image):
        result = op_brightness(test_image, 1.5)
        assert result.size == test_image.size
        # Brighter image should have higher average pixel value
        orig_avg = sum(test_image.getpixel((50, 40))[:3]) / 3
        result_avg = sum(result.getpixel((50, 40))[:3]) / 3
        assert result_avg > orig_avg

    def test_brightness_decrease(self, test_image: Image.Image):
        result = op_brightness(test_image, 0.5)
        orig_avg = sum(test_image.getpixel((50, 40))[:3]) / 3
        result_avg = sum(result.getpixel((50, 40))[:3]) / 3
        assert result_avg < orig_avg

    def test_brightness_no_change(self, test_image: Image.Image):
        result = op_brightness(test_image, 1.0)
        assert result.getpixel((50, 40)) == test_image.getpixel((50, 40))

    def test_contrast(self, test_image: Image.Image):
        result = op_contrast(test_image, 1.5)
        assert result.size == test_image.size

    def test_saturation(self, test_image: Image.Image):
        result = op_saturation(test_image, 1.5)
        assert result.size == test_image.size

    def test_saturation_zero_is_grayscale(self, test_image: Image.Image):
        result = op_saturation(test_image, 0.0)
        r, g, b, a = result.getpixel((50, 40))
        # With zero saturation, R=G=B (grayscale)
        assert r == g == b

    def test_sharpen(self, test_image: Image.Image):
        result = op_sharpen(test_image, 2.0)
        assert result.size == test_image.size

    def test_blur(self, test_image: Image.Image):
        result = op_blur(test_image, 3.0)
        assert result.size == test_image.size

    def test_grayscale_preserves_alpha(self):
        img = Image.new("RGBA", (10, 10), color=(255, 0, 0, 128))
        result = op_grayscale(img)
        assert result.mode == "RGBA"
        _, _, _, a = result.getpixel((5, 5))
        assert a == 128

    def test_grayscale_channels_equal(self, test_image: Image.Image):
        result = op_grayscale(test_image)
        r, g, b, _ = result.getpixel((50, 40))
        assert r == g == b

    def test_grayscale_rgb_input(self):
        img = Image.new("RGB", (10, 10), color=(255, 0, 0))
        result = op_grayscale(img)
        assert result.mode == "RGBA"

    def test_invert_rgb(self):
        img = Image.new("RGBA", (10, 10), color=(100, 150, 200, 255))
        result = op_invert(img)
        r, g, b, a = result.getpixel((5, 5))
        assert r == 155
        assert g == 105
        assert b == 55

    def test_invert_preserves_alpha(self):
        img = Image.new("RGBA", (10, 10), color=(100, 150, 200, 128))
        result = op_invert(img)
        _, _, _, a = result.getpixel((5, 5))
        assert a == 128


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

    def test_registry_contains_color_ops(self):
        assert "brightness" in OPERATIONS
        assert "contrast" in OPERATIONS
        assert "saturation" in OPERATIONS
        assert "sharpen" in OPERATIONS
        assert "blur" in OPERATIONS
        assert "grayscale" in OPERATIONS
        assert "invert" in OPERATIONS

    def test_registry_count(self):
        """OPERATIONS should have 21 entries (9 geometric + 7 color + 4 trim/alpha + 1 mask)."""
        assert len(OPERATIONS) == 21

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

    def test_apply_operation_brightness(self, test_image: Image.Image):
        result = apply_operation(test_image, "brightness", 1.5)
        assert result.size == test_image.size

    def test_apply_operation_blur(self, test_image: Image.Image):
        result = apply_operation(test_image, "blur", 2.0)
        assert result.size == test_image.size

    def test_apply_operation_grayscale(self, test_image: Image.Image):
        result = apply_operation(test_image, "grayscale")
        r, g, b, _ = result.getpixel((50, 40))
        assert r == g == b

    def test_apply_operation_invert(self, test_image: Image.Image):
        result = apply_operation(test_image, "invert")
        assert result.size == test_image.size

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
    """Tests for uniform output handling (always JSON)."""

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

    def test_always_json_output(self, bound_state: PipelineState):
        """Test handle_output always writes JSON."""
        with mock.patch("chop.pipeline.write_pipeline_output") as mock_write:
            handle_output(bound_state)
            mock_write.assert_called_once_with(bound_state)

    def test_unbound_outputs_json(self, unbound_state: PipelineState):
        """Test unbound pipeline also outputs JSON (no TTY special case)."""
        with mock.patch("chop.pipeline.write_pipeline_output") as mock_write:
            handle_output(unbound_state)
            mock_write.assert_called_once_with(unbound_state)

    def test_handle_output_no_args_parameter(self):
        """Test handle_output takes only state (no args)."""
        import inspect
        sig = inspect.signature(handle_output)
        assert list(sig.parameters.keys()) == ["state"]


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
                result = cmd_save(args)

            assert result is not None  # Returns state for chaining
            assert isinstance(result, PipelineState)
            assert Path(f.name).exists()
            saved = Image.open(f.name)
            assert saved.size == (100, 80)
            Path(f.name).unlink()

    def test_save_returns_state_for_chaining(self, test_image_file: str):
        """Test save returns PipelineState (non-terminal)."""
        from chop.cli import cmd_save

        state = PipelineState()
        state.add_op("load", test_image_file)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            args = mock.Mock(path=f.name, format=None)
            with mock.patch("chop.cli.read_pipeline_input", return_value=state):
                result = cmd_save(args)
            assert result is state
            Path(f.name).unlink()

    def test_save_stdout_returns_none(self, test_image_file: str):
        """Test save to stdout returns None (binary can't coexist with JSON)."""
        from chop.cli import cmd_save

        state = PipelineState()
        state.add_op("load", test_image_file)
        args = mock.Mock(path="-", format="png")

        buf = io.BytesIO()
        with mock.patch("chop.cli.read_pipeline_input", return_value=state):
            with mock.patch("sys.stdout") as mock_stdout:
                mock_stdout.buffer = buf
                result = cmd_save(args)

        assert result is None

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

    def test_save_stderr_message(self, test_image_file: str, capsys):
        """Test save prints confirmation to stderr."""
        from chop.cli import cmd_save

        state = PipelineState()
        state.add_op("load", test_image_file)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            args = mock.Mock(path=f.name, format=None)
            with mock.patch("chop.cli.read_pipeline_input", return_value=state):
                cmd_save(args)

            captured = capsys.readouterr()
            assert "Saved to" in captured.err
            Path(f.name).unlink()


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
            result = cmd_info(args)

        captured = capsys.readouterr()
        assert "Cursor image: 50x40" in captured.err
        assert isinstance(result, PipelineState)

    def test_info_returns_state(self, test_image_file: str):
        """Test info returns PipelineState for chaining."""
        from chop.cli import cmd_info

        state = PipelineState()
        state.add_op("load", test_image_file)
        args = mock.Mock()

        with mock.patch("chop.cli.read_pipeline_input", return_value=state):
            result = cmd_info(args)

        assert result is state

    def test_info_enriches_metadata(self, test_image_file: str):
        """Test info adds width/height/mode/images_loaded to metadata."""
        from chop.cli import cmd_info

        state = PipelineState()
        state.add_op("load", test_image_file)
        args = mock.Mock()

        with mock.patch("chop.cli.read_pipeline_input", return_value=state):
            result = cmd_info(args)

        assert result.metadata["width"] == 100
        assert result.metadata["height"] == 80
        assert result.metadata["mode"] == "RGBA"
        assert result.metadata["images_loaded"] == 1

    def test_info_metadata_survives_json(self, test_image_file: str):
        """Test info metadata survives JSON serialization."""
        from chop.cli import cmd_info

        state = PipelineState()
        state.add_op("load", test_image_file)
        args = mock.Mock()

        with mock.patch("chop.cli.read_pipeline_input", return_value=state):
            result = cmd_info(args)

        json_str = result.to_json()
        restored = PipelineState.from_json(json_str)
        assert restored.metadata["width"] == 100

    def test_info_unbound_pipeline(self, capsys):
        """Test info on unbound pipeline shows program listing."""
        from chop.cli import cmd_info

        state = PipelineState()
        state.add_op("resize", "50%")
        state.add_op("pad", 10)
        args = mock.Mock()

        with mock.patch("chop.cli.read_pipeline_input", return_value=state):
            result = cmd_info(args)

        captured = capsys.readouterr()
        assert "Unbound program" in captured.err
        assert "resize 50%" in captured.err
        assert isinstance(result, PipelineState)

    def test_info_empty_pipeline(self, capsys):
        """Test info on empty pipeline."""
        from chop.cli import cmd_info

        state = PipelineState()
        args = mock.Mock()

        with mock.patch("chop.cli.read_pipeline_input", return_value=state):
            result = cmd_info(args)

        captured = capsys.readouterr()
        assert "Empty pipeline" in captured.err
        assert isinstance(result, PipelineState)

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


class TestPrintCommand:
    """Tests for the print command."""

    def test_print_default_message(self, capsys):
        """Test print shows op count and bound/unbound status."""
        from chop.cli import cmd_print

        state = PipelineState()
        state.add_op("resize", "50%")
        state.add_op("pad", 10)
        args = mock.Mock(message=None)

        with mock.patch("chop.cli.read_pipeline_input", return_value=state):
            result = cmd_print(args)

        captured = capsys.readouterr()
        assert "Pipeline: 2 ops (unbound)" in captured.err
        assert isinstance(result, PipelineState)

    def test_print_bound_status(self, test_image_file: str, capsys):
        """Test print shows bound status when pipeline has load."""
        from chop.cli import cmd_print

        state = PipelineState()
        state.add_op("load", test_image_file)
        state.add_op("resize", "50%")
        args = mock.Mock(message=None)

        with mock.patch("chop.cli.read_pipeline_input", return_value=state):
            cmd_print(args)

        captured = capsys.readouterr()
        assert "Pipeline: 2 ops (bound)" in captured.err

    def test_print_custom_message(self, capsys):
        """Test print with custom message."""
        from chop.cli import cmd_print

        state = PipelineState()
        args = mock.Mock(message="custom debug info")

        with mock.patch("chop.cli.read_pipeline_input", return_value=state):
            result = cmd_print(args)

        captured = capsys.readouterr()
        assert "custom debug info" in captured.err
        assert isinstance(result, PipelineState)

    def test_print_returns_state_unchanged(self):
        """Test print does not modify state."""
        from chop.cli import cmd_print

        state = PipelineState()
        state.add_op("resize", "50%")
        args = mock.Mock(message=None)

        with mock.patch("chop.cli.read_pipeline_input", return_value=state):
            result = cmd_print(args)

        assert result is state
        assert len(result.ops) == 1


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

        args = mock.Mock(images=["a", "b"], align="center", gap=0, gap_color="transparent", label=None)
        with mock.patch("chop.cli.read_pipeline_input", return_value=None):
            state = cmd_hstack(args)

        assert state.ops[0] == ("hstack", ("a", "b"), {"align": "center"})

    def test_cmd_hstack_no_labels(self):
        """Test cmd_hstack with no label args (default all)."""
        from chop.cli import cmd_hstack

        args = mock.Mock(images=[], align="center", gap=0, gap_color="transparent", label=None)
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

        args = mock.Mock(images=["a", "b", "c", "d"], cols=2, gap=0, gap_color="transparent", label="result")
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


class TestColorCLIHandlers:
    """Tests for color operation CLI handlers."""

    def test_cmd_brightness(self):
        from chop.cli import cmd_brightness
        args = mock.Mock(factor=1.5, on=None)
        with mock.patch("chop.cli.read_pipeline_input", return_value=None):
            state = cmd_brightness(args)
        assert state.ops[0] == ("brightness", (1.5,), {})

    def test_cmd_brightness_with_on(self):
        from chop.cli import cmd_brightness
        args = mock.Mock(factor=1.5, on="fg")
        with mock.patch("chop.cli.read_pipeline_input", return_value=None):
            state = cmd_brightness(args)
        assert state.ops[0] == ("brightness", (1.5,), {"on": "fg"})

    def test_cmd_contrast(self):
        from chop.cli import cmd_contrast
        args = mock.Mock(factor=1.2, on=None)
        with mock.patch("chop.cli.read_pipeline_input", return_value=None):
            state = cmd_contrast(args)
        assert state.ops[0] == ("contrast", (1.2,), {})

    def test_cmd_saturation(self):
        from chop.cli import cmd_saturation
        args = mock.Mock(factor=0.5, on=None)
        with mock.patch("chop.cli.read_pipeline_input", return_value=None):
            state = cmd_saturation(args)
        assert state.ops[0] == ("saturation", (0.5,), {})

    def test_cmd_sharpen(self):
        from chop.cli import cmd_sharpen
        args = mock.Mock(factor=2.0, on=None)
        with mock.patch("chop.cli.read_pipeline_input", return_value=None):
            state = cmd_sharpen(args)
        assert state.ops[0] == ("sharpen", (2.0,), {})

    def test_cmd_blur(self):
        from chop.cli import cmd_blur
        args = mock.Mock(radius=3.0, on=None)
        with mock.patch("chop.cli.read_pipeline_input", return_value=None):
            state = cmd_blur(args)
        assert state.ops[0] == ("blur", (3.0,), {})

    def test_cmd_grayscale(self):
        from chop.cli import cmd_grayscale
        args = mock.Mock(on=None)
        with mock.patch("chop.cli.read_pipeline_input", return_value=None):
            state = cmd_grayscale(args)
        assert state.ops[0] == ("grayscale", (), {})

    def test_cmd_invert(self):
        from chop.cli import cmd_invert
        args = mock.Mock(on=None)
        with mock.patch("chop.cli.read_pipeline_input", return_value=None):
            state = cmd_invert(args)
        assert state.ops[0] == ("invert", (), {})

    def test_cmd_invert_with_on(self):
        from chop.cli import cmd_invert
        args = mock.Mock(on="bg")
        with mock.patch("chop.cli.read_pipeline_input", return_value=None):
            state = cmd_invert(args)
        assert state.ops[0] == ("invert", (), {"on": "bg"})


class TestColorPipeline:
    """Tests for color operations in pipeline context."""

    def test_color_op_materializes(self, test_image_file: str):
        """Test color op in pipeline materializes correctly."""
        state = PipelineState()
        state.add_op("load", test_image_file)
        state.add_op("brightness", 1.5)

        image = state.materialize()
        assert image.size == (100, 80)

    def test_chain_multiple_color_ops(self, test_image_file: str):
        """Test chaining multiple color operations."""
        state = PipelineState()
        state.add_op("load", test_image_file)
        state.add_op("brightness", 1.2)
        state.add_op("contrast", 1.3)
        state.add_op("saturation", 0.8)

        image = state.materialize()
        assert image.size == (100, 80)

    def test_grayscale_then_resize(self, test_image_file: str):
        """Test grayscale followed by resize."""
        state = PipelineState()
        state.add_op("load", test_image_file)
        state.add_op("grayscale")
        state.add_op("resize", "50%")

        image = state.materialize()
        assert image.size == (50, 40)

    def test_color_op_with_on_flag(self, test_image_file: str, second_image_file: str):
        """Test color op targeting a specific labeled image."""
        state = PipelineState()
        state.add_op("load", test_image_file, **{"as": "a"})
        state.add_op("load", second_image_file, **{"as": "b"})
        state.add_op("brightness", 1.5, on="a")

        # Should not raise — applies brightness to 'a', cursor on 'b'
        image = state.materialize()
        assert image.size == (50, 40)

    def test_color_ops_json_roundtrip(self):
        """Test color ops survive JSON serialization."""
        state = PipelineState()
        state.add_op("brightness", 1.5)
        state.add_op("grayscale")
        state.add_op("blur", 2.0)

        json_str = state.to_json()
        restored = PipelineState.from_json(json_str)

        assert len(restored.ops) == 3
        assert restored.ops[0] == ("brightness", (1.5,), {})
        assert restored.ops[1] == ("grayscale", (), {})
        assert restored.ops[2] == ("blur", (2.0,), {})


class TestUniformityIntegration:
    """Integration tests for uniform output model."""

    def test_multi_save_pipeline(self, test_image_file: str):
        """Test save returns state, enabling chaining."""
        from chop.cli import cmd_save

        state = PipelineState()
        state.add_op("load", test_image_file)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            args = mock.Mock(path=f.name, format=None)
            with mock.patch("chop.cli.read_pipeline_input", return_value=state):
                result = cmd_save(args)

            assert result is not None
            assert result is state
            Path(f.name).unlink()

    def test_info_in_middle_preserves_ops(self, test_image_file: str):
        """Test info in middle of pipeline preserves all ops."""
        from chop.cli import cmd_info

        state = PipelineState()
        state.add_op("load", test_image_file)
        state.add_op("resize", "50%")
        args = mock.Mock()

        with mock.patch("chop.cli.read_pipeline_input", return_value=state):
            result = cmd_info(args)

        assert len(result.ops) == 2
        assert result.ops[0][0] == "load"
        assert result.ops[1] == ("resize", ("50%",), {})

    def test_no_terminal_handlers_dict(self):
        """Test main() source has no terminal_handlers dict."""
        import inspect
        from chop.cli import main
        source = inspect.getsource(main)
        assert "terminal_handlers" not in source

    def test_unified_handlers_dict(self):
        """Test main() uses a single unified handlers dict."""
        import inspect
        from chop.cli import main
        source = inspect.getsource(main)
        assert "handlers" in source
        assert "state_handlers" not in source


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

    def test_color_ops_in_full_pipeline(self, test_image_file: str):
        """Integration test: color ops mixed with geometric ops."""
        state = PipelineState()
        state.add_op("load", test_image_file)
        state.add_op("brightness", 1.3)
        state.add_op("resize", "50%")
        state.add_op("grayscale")
        state.add_op("border", 2, color="white")

        image = state.materialize()
        # 100x80 → resize 50% → 50x40 → border 2 → 54x44
        assert image.size == (54, 44)


# =============================================================================
# Tests for new operations (v0.5.0)
# =============================================================================


class TestTrimOperation:
    """Tests for op_trim."""

    def test_trim_transparent_padding(self):
        """Trim removes transparent borders."""
        inner = Image.new("RGBA", (40, 30), (255, 0, 0, 255))
        padded = Image.new("RGBA", (80, 70), (0, 0, 0, 0))
        padded.paste(inner, (20, 20))
        result = op_trim(padded)
        assert result.size == (40, 30)

    def test_trim_uniform_color_border(self):
        """Trim removes solid color borders from opaque image."""
        inner = Image.new("RGB", (40, 30), (255, 0, 0))
        padded = Image.new("RGB", (80, 70), (255, 255, 255))
        padded.paste(inner, (20, 20))
        result = op_trim(padded)
        assert result.size == (40, 30)

    def test_trim_no_border(self, test_image: Image.Image):
        """Trim on image with no uniform border returns same size."""
        # test_image is uniform color, so trimming transparent around it does nothing
        # because it's fully opaque — but the RGBA image has no transparent border
        result = op_trim(test_image)
        assert result.size == test_image.size

    def test_trim_with_fuzz(self):
        """Trim with fuzz tolerance removes near-matching borders."""
        # Create inner content
        inner = Image.new("RGBA", (40, 30), (255, 0, 0, 255))
        # Create border with slightly-off transparent (alpha=5)
        padded = Image.new("RGBA", (80, 70), (0, 0, 0, 5))
        padded.paste(inner, (20, 20))
        # Without fuzz, the near-transparent border might not trim fully
        result_no_fuzz = op_trim(padded, fuzz=0)
        result_fuzz = op_trim(padded, fuzz=10)
        assert result_fuzz.size == (40, 30)

    def test_trim_fully_uniform(self):
        """Trim on fully uniform image returns original (nothing to crop)."""
        img = Image.new("RGBA", (50, 50), (0, 0, 0, 0))
        result = op_trim(img)
        assert result.size == (50, 50)

    def test_trim_via_apply_operation(self, test_image: Image.Image):
        """Test trim through apply_operation dispatch."""
        result = apply_operation(test_image, "trim")
        assert result.size == test_image.size

    def test_trim_pipeline(self, test_image_file: str):
        """Test trim in a pipeline: pad then trim restores size."""
        state = PipelineState()
        state.add_op("load", test_image_file)
        state.add_op("pad", 20, color="white")
        state.add_op("trim")
        image = state.materialize()
        assert image.size == (100, 80)


class TestColorizeOperation:
    """Tests for op_colorize."""

    def test_colorize_basic(self, test_image: Image.Image):
        """Colorize tints the image."""
        result = op_colorize(test_image, "red")
        assert result.size == test_image.size
        assert result.mode == "RGBA"
        r, g, b, _ = result.getpixel((50, 40))
        # Red tint: R should dominate, G and B should be low
        assert r > g
        assert r > b

    def test_colorize_with_hex(self, test_image: Image.Image):
        """Colorize works with hex color."""
        result = op_colorize(test_image, "#704214")
        assert result.size == test_image.size

    def test_colorize_strength_zero(self, test_image: Image.Image):
        """Colorize with strength=0 returns original."""
        result = op_colorize(test_image, "red", strength=0.0)
        orig = test_image.convert("RGBA")
        assert result.getpixel((50, 40)) == orig.getpixel((50, 40))

    def test_colorize_strength_partial(self, test_image: Image.Image):
        """Colorize with partial strength blends."""
        full = op_colorize(test_image, "blue", strength=1.0)
        partial = op_colorize(test_image, "blue", strength=0.5)
        orig = test_image.convert("RGBA")
        # Partial should be between original and full
        _, _, b_full, _ = full.getpixel((50, 40))
        _, _, b_partial, _ = partial.getpixel((50, 40))
        _, _, b_orig, _ = orig.getpixel((50, 40))
        assert min(b_orig, b_full) <= b_partial <= max(b_orig, b_full)

    def test_colorize_preserves_alpha(self):
        """Colorize preserves the alpha channel."""
        img = Image.new("RGBA", (10, 10), (255, 0, 0, 128))
        result = op_colorize(img, "blue")
        _, _, _, a = result.getpixel((5, 5))
        assert a == 128

    def test_colorize_via_apply_operation(self, test_image: Image.Image):
        """Test colorize through apply_operation dispatch."""
        result = apply_operation(test_image, "colorize", "#704214")
        assert result.size == test_image.size

    def test_colorize_pipeline(self, test_image_file: str):
        """Test colorize in pipeline with grayscale for sepia effect."""
        state = PipelineState()
        state.add_op("load", test_image_file)
        state.add_op("grayscale")
        state.add_op("colorize", "#704214")
        image = state.materialize()
        assert image.size == (100, 80)


class TestOpacityOperation:
    """Tests for op_opacity."""

    def test_opacity_half(self):
        """Opacity 0.5 halves alpha channel."""
        img = Image.new("RGBA", (10, 10), (255, 0, 0, 200))
        result = op_opacity(img, 0.5)
        _, _, _, a = result.getpixel((5, 5))
        assert a == 100

    def test_opacity_zero(self):
        """Opacity 0 makes fully transparent."""
        img = Image.new("RGBA", (10, 10), (255, 0, 0, 255))
        result = op_opacity(img, 0.0)
        _, _, _, a = result.getpixel((5, 5))
        assert a == 0

    def test_opacity_one(self):
        """Opacity 1.0 preserves alpha."""
        img = Image.new("RGBA", (10, 10), (255, 0, 0, 200))
        result = op_opacity(img, 1.0)
        _, _, _, a = result.getpixel((5, 5))
        assert a == 200

    def test_opacity_preserves_rgb(self):
        """Opacity only affects alpha, not RGB."""
        img = Image.new("RGBA", (10, 10), (100, 150, 200, 255))
        result = op_opacity(img, 0.5)
        r, g, b, _ = result.getpixel((5, 5))
        assert (r, g, b) == (100, 150, 200)

    def test_opacity_via_apply_operation(self, test_image: Image.Image):
        """Test opacity through apply_operation dispatch."""
        result = apply_operation(test_image, "opacity", 0.5)
        _, _, _, a = result.getpixel((50, 40))
        assert a == 127  # 255 * 0.5

    def test_opacity_pipeline(self, test_image_file: str):
        """Test opacity in pipeline."""
        state = PipelineState()
        state.add_op("load", test_image_file)
        state.add_op("opacity", 0.3)
        image = state.materialize()
        _, _, _, a = image.getpixel((50, 40))
        assert a == 76  # 255 * 0.3 = 76.5 → int 76


class TestBackgroundOperation:
    """Tests for op_background."""

    def test_background_white(self):
        """Background flattens onto white."""
        img = Image.new("RGBA", (10, 10), (255, 0, 0, 128))
        result = op_background(img, "white")
        r, g, b, a = result.getpixel((5, 5))
        assert a == 255  # Fully opaque
        # Semi-transparent red on white → pinkish
        assert r > g
        assert r > b

    def test_background_black(self):
        """Background flattens onto black."""
        img = Image.new("RGBA", (10, 10), (255, 0, 0, 128))
        result = op_background(img, "black")
        r, g, b, a = result.getpixel((5, 5))
        assert a == 255

    def test_background_opaque_image(self):
        """Background on fully opaque image preserves colors."""
        img = Image.new("RGBA", (10, 10), (100, 150, 200, 255))
        result = op_background(img, "white")
        r, g, b, a = result.getpixel((5, 5))
        assert (r, g, b, a) == (100, 150, 200, 255)

    def test_background_transparent_image(self):
        """Background on fully transparent image shows background color."""
        img = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
        result = op_background(img, "red")
        r, g, b, a = result.getpixel((5, 5))
        assert (r, g, b, a) == (255, 0, 0, 255)

    def test_background_via_apply_operation(self, test_image: Image.Image):
        """Test background through apply_operation dispatch."""
        result = apply_operation(test_image, "background", "white")
        _, _, _, a = result.getpixel((50, 40))
        assert a == 255

    def test_background_pipeline(self, test_image_file: str):
        """Test background in pipeline."""
        state = PipelineState()
        state.add_op("load", test_image_file)
        state.add_op("background", "white")
        image = state.materialize()
        _, _, _, a = image.getpixel((50, 40))
        assert a == 255


class TestMaskOperation:
    """Tests for op_mask."""

    def test_mask_circle(self, test_image: Image.Image):
        """Circle mask makes corners transparent."""
        result = op_mask(test_image, "circle")
        assert result.size == test_image.size
        # Corner should be transparent
        _, _, _, a = result.getpixel((0, 0))
        assert a == 0
        # Center should be opaque
        _, _, _, a = result.getpixel((50, 40))
        assert a == 255

    def test_mask_ellipse(self, test_image: Image.Image):
        """Ellipse mask fills image bounds."""
        result = op_mask(test_image, "ellipse")
        assert result.size == test_image.size
        # Corner should be transparent
        _, _, _, a = result.getpixel((0, 0))
        assert a == 0

    def test_mask_roundrect(self, test_image: Image.Image):
        """Roundrect mask with radius."""
        result = op_mask(test_image, "roundrect", radius=20)
        assert result.size == test_image.size
        # Corner should be transparent
        _, _, _, a = result.getpixel((0, 0))
        assert a == 0
        # Center should be opaque
        _, _, _, a = result.getpixel((50, 40))
        assert a == 255

    def test_mask_roundrect_zero_radius(self, test_image: Image.Image):
        """Roundrect with radius=0 is a full rectangle (everything visible)."""
        result = op_mask(test_image, "roundrect", radius=0)
        _, _, _, a = result.getpixel((0, 0))
        assert a == 255

    def test_mask_invert(self, test_image: Image.Image):
        """Inverted mask makes inside transparent, outside opaque."""
        result = op_mask(test_image, "circle", invert=True)
        # Center should be transparent (inverted)
        _, _, _, a = result.getpixel((50, 40))
        assert a == 0
        # Corner should be opaque (inverted)
        _, _, _, a = result.getpixel((0, 0))
        assert a == 255

    def test_mask_preserves_existing_alpha(self):
        """Mask multiplies with existing alpha (doesn't replace)."""
        img = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
        result = op_mask(img, "circle")
        # Center should have alpha = min(128, 255) via multiply → 128
        _, _, _, a = result.getpixel((50, 50))
        assert a == 128
        # Corner should be 0 (128 * 0 via multiply)
        _, _, _, a = result.getpixel((0, 0))
        assert a == 0

    def test_mask_unknown_shape(self, test_image: Image.Image):
        """Unknown shape raises ValueError."""
        with pytest.raises(ValueError, match="Unknown mask shape"):
            op_mask(test_image, "star")

    def test_mask_via_apply_operation(self, test_image: Image.Image):
        """Test mask through apply_operation dispatch."""
        result = apply_operation(test_image, "mask", "circle")
        _, _, _, a = result.getpixel((0, 0))
        assert a == 0

    def test_mask_pipeline(self, test_image_file: str):
        """Test mask in pipeline."""
        state = PipelineState()
        state.add_op("load", test_image_file)
        state.add_op("mask", "roundrect", radius=10)
        image = state.materialize()
        assert image.size == (100, 80)
        _, _, _, a = image.getpixel((0, 0))
        assert a == 0


class TestGapOnComposition:
    """Tests for --gap/--gap-color on hstack, vstack, grid."""

    @pytest.fixture
    def img_a(self) -> Image.Image:
        return Image.new("RGBA", (50, 40), (255, 0, 0, 255))

    @pytest.fixture
    def img_b(self) -> Image.Image:
        return Image.new("RGBA", (50, 40), (0, 255, 0, 255))

    def test_hstack_gap(self, img_a, img_b):
        """Hstack with gap adds spacing."""
        result = op_hstack(img_a, img_b, gap=10)
        assert result.size == (110, 40)  # 50 + 10 + 50

    def test_hstack_gap_color(self, img_a, img_b):
        """Hstack gap region has correct color."""
        result = op_hstack(img_a, img_b, gap=10, gap_color="white")
        # Gap pixel (between the two images) should be white
        r, g, b, a = result.getpixel((55, 20))
        assert (r, g, b) == (255, 255, 255)

    def test_hstack_gap_zero(self, img_a, img_b):
        """Hstack with gap=0 is same as no gap."""
        result = op_hstack(img_a, img_b, gap=0)
        assert result.size == (100, 40)

    def test_vstack_gap(self, img_a, img_b):
        """Vstack with gap adds spacing."""
        result = op_vstack(img_a, img_b, gap=10)
        assert result.size == (50, 90)  # 40 + 10 + 40

    def test_vstack_gap_color(self, img_a, img_b):
        """Vstack gap region has correct color."""
        result = op_vstack(img_a, img_b, gap=10, gap_color="blue")
        # Gap pixel (between the two images) should be blue
        r, g, b, a = result.getpixel((25, 45))
        assert (r, g, b) == (0, 0, 255)

    def test_grid_gap(self, img_a):
        """Grid with gap adds spacing between cells."""
        others = [img_a.copy(), img_a.copy(), img_a.copy()]
        result = op_grid(img_a, others, cols=2, gap=10)
        # 2 cols: 50 + 10 + 50 = 110 wide
        # 2 rows: 40 + 10 + 40 = 90 tall
        assert result.size == (110, 90)

    def test_grid_gap_color(self, img_a):
        """Grid gap has correct color."""
        img_b = Image.new("RGBA", (50, 40), (0, 255, 0, 255))
        result = op_grid(img_a, [img_b], cols=2, gap=10, gap_color="white")
        # Gap between the two images
        r, g, b, a = result.getpixel((55, 20))
        assert (r, g, b) == (255, 255, 255)

    def test_hstack_gap_pipeline(self, test_image_file: str, second_image_file: str):
        """Test gap in hstack via pipeline."""
        state = PipelineState()
        state.add_op("load", test_image_file, **{"as": "a"})
        state.add_op("load", second_image_file, **{"as": "b"})
        state.add_op("hstack", "a", "b", gap=10, gap_color="white")
        image = state.materialize()
        assert image.size == (160, 80)  # 100 + 10 + 50

    def test_grid_gap_pipeline(self, test_image_file: str):
        """Test gap in grid via pipeline."""
        state = PipelineState()
        state.add_op("load", test_image_file, **{"as": "a"})
        state.add_op("load", test_image_file, **{"as": "b"})
        state.add_op("load", test_image_file, **{"as": "c"})
        state.add_op("load", test_image_file, **{"as": "d"})
        state.add_op("grid", "a", "b", "c", "d", cols=2, gap=5)
        image = state.materialize()
        # 2 cols: 100 + 5 + 100 = 205 wide
        # 2 rows: 80 + 5 + 80 = 165 tall
        assert image.size == (205, 165)


class TestCanvasOperation:
    """Tests for canvas source operation."""

    def test_canvas_basic(self):
        """Canvas creates blank image."""
        state = PipelineState()
        state.add_op("canvas", "200x100")
        image = state.materialize()
        assert image.size == (200, 100)
        assert image.mode == "RGBA"

    def test_canvas_with_color(self):
        """Canvas with color fills the image."""
        state = PipelineState()
        state.add_op("canvas", "50x50", color="red")
        image = state.materialize()
        r, g, b, a = image.getpixel((25, 25))
        assert (r, g, b, a) == (255, 0, 0, 255)

    def test_canvas_transparent(self):
        """Canvas defaults to transparent."""
        state = PipelineState()
        state.add_op("canvas", "50x50")
        image = state.materialize()
        _, _, _, a = image.getpixel((25, 25))
        assert a == 0

    def test_canvas_with_label(self):
        """Canvas with --as creates named label."""
        state = PipelineState()
        state.add_op("canvas", "200x100", **{"as": "bg"})
        state.add_op("select", "bg")
        image = state.materialize()
        assert image.size == (200, 100)

    def test_canvas_auto_label(self):
        """Canvas without --as gets auto-label."""
        state = PipelineState()
        state.add_op("canvas", "200x100")
        state.add_op("select", "img")  # First auto-label
        state.materialize()  # Should not raise

    def test_canvas_has_load_true(self):
        """has_load returns True for canvas-based pipeline."""
        state = PipelineState()
        state.add_op("canvas", "200x100")
        assert state.has_load()

    def test_canvas_with_overlay(self, test_image_file: str):
        """Canvas as background for overlay composition."""
        state = PipelineState()
        state.add_op("canvas", "200x100", color="white", **{"as": "bg"})
        state.add_op("load", test_image_file, **{"as": "fg"})
        state.add_op("overlay", "bg", "fg", x=50, y=10)
        image = state.materialize()
        assert image.size == (200, 100)

    def test_canvas_json_roundtrip(self):
        """Canvas op survives JSON serialization."""
        state = PipelineState()
        state.add_op("canvas", "800x600", color="blue", **{"as": "bg"})
        json_str = state.to_json()
        restored = PipelineState.from_json(json_str)
        assert restored.ops[0] == ("canvas", ("800x600",), {"color": "blue", "as": "bg"})
        image = restored.materialize()
        assert image.size == (800, 600)

    def test_canvas_hex_color(self):
        """Canvas with hex color."""
        state = PipelineState()
        state.add_op("canvas", "50x50", color="#2d5016")
        image = state.materialize()
        r, g, b, a = image.getpixel((25, 25))
        assert (r, g, b) == (45, 80, 22)


class TestNewOpsCLIHandlers:
    """Tests for CLI handlers of new operations."""

    def test_cmd_trim_basic(self):
        from chop.cli import cmd_trim
        args = mock.Mock(on=None, fuzz=0)
        with mock.patch("chop.cli.read_pipeline_input", return_value=None):
            state = cmd_trim(args)
        assert state.ops[0] == ("trim", (), {})

    def test_cmd_trim_with_fuzz(self):
        from chop.cli import cmd_trim
        args = mock.Mock(on=None, fuzz=10)
        with mock.patch("chop.cli.read_pipeline_input", return_value=None):
            state = cmd_trim(args)
        assert state.ops[0] == ("trim", (), {"fuzz": 10})

    def test_cmd_colorize_basic(self):
        from chop.cli import cmd_colorize
        args = mock.Mock(color="red", strength=1.0, on=None)
        with mock.patch("chop.cli.read_pipeline_input", return_value=None):
            state = cmd_colorize(args)
        assert state.ops[0] == ("colorize", ("red",), {})

    def test_cmd_colorize_with_strength(self):
        from chop.cli import cmd_colorize
        args = mock.Mock(color="#704214", strength=0.5, on=None)
        with mock.patch("chop.cli.read_pipeline_input", return_value=None):
            state = cmd_colorize(args)
        assert state.ops[0] == ("colorize", ("#704214",), {"strength": 0.5})

    def test_cmd_opacity(self):
        from chop.cli import cmd_opacity
        args = mock.Mock(factor=0.5, on=None)
        with mock.patch("chop.cli.read_pipeline_input", return_value=None):
            state = cmd_opacity(args)
        assert state.ops[0] == ("opacity", (0.5,), {})

    def test_cmd_background(self):
        from chop.cli import cmd_background
        args = mock.Mock(color="white", on=None)
        with mock.patch("chop.cli.read_pipeline_input", return_value=None):
            state = cmd_background(args)
        assert state.ops[0] == ("background", ("white",), {})

    def test_cmd_mask_circle(self):
        from chop.cli import cmd_mask
        args = mock.Mock(shape="circle", radius=0, on=None, invert=False)
        with mock.patch("chop.cli.read_pipeline_input", return_value=None):
            state = cmd_mask(args)
        assert state.ops[0] == ("mask", ("circle",), {})

    def test_cmd_mask_roundrect_with_radius(self):
        from chop.cli import cmd_mask
        args = mock.Mock(shape="roundrect", radius=20, on=None, invert=False)
        with mock.patch("chop.cli.read_pipeline_input", return_value=None):
            state = cmd_mask(args)
        assert state.ops[0] == ("mask", ("roundrect",), {"radius": 20})

    def test_cmd_mask_invert(self):
        from chop.cli import cmd_mask
        args = mock.Mock(shape="circle", radius=0, on=None, invert=True)
        with mock.patch("chop.cli.read_pipeline_input", return_value=None):
            state = cmd_mask(args)
        assert state.ops[0] == ("mask", ("circle",), {"invert": True})

    def test_cmd_canvas_basic(self):
        from chop.cli import cmd_canvas
        args = mock.Mock(size="800x600", color="transparent", label=None)
        with mock.patch("chop.cli.read_pipeline_input", return_value=None):
            state = cmd_canvas(args)
        assert state.ops[0] == ("canvas", ("800x600",), {})

    def test_cmd_canvas_with_color_and_label(self):
        from chop.cli import cmd_canvas
        args = mock.Mock(size="800x600", color="white", label="bg")
        with mock.patch("chop.cli.read_pipeline_input", return_value=None):
            state = cmd_canvas(args)
        assert state.ops[0] == ("canvas", ("800x600",), {"color": "white", "as": "bg"})

    def test_cmd_hstack_with_gap(self):
        from chop.cli import cmd_hstack
        args = mock.Mock(images=["a", "b"], align="center", gap=10, gap_color="white", label=None)
        with mock.patch("chop.cli.read_pipeline_input", return_value=None):
            state = cmd_hstack(args)
        assert state.ops[0] == ("hstack", ("a", "b"), {"align": "center", "gap": 10, "gap_color": "white"})

    def test_cmd_vstack_with_gap(self):
        from chop.cli import cmd_vstack
        args = mock.Mock(images=[], align="center", gap=5, gap_color="transparent", label=None)
        with mock.patch("chop.cli.read_pipeline_input", return_value=None):
            state = cmd_vstack(args)
        assert state.ops[0] == ("vstack", (), {"align": "center", "gap": 5})

    def test_cmd_grid_with_gap(self):
        from chop.cli import cmd_grid
        args = mock.Mock(images=[], cols=3, gap=10, gap_color="black", label=None)
        with mock.patch("chop.cli.read_pipeline_input", return_value=None):
            state = cmd_grid(args)
        assert state.ops[0] == ("grid", (), {"cols": 3, "gap": 10, "gap_color": "black"})


class TestNewOpsIntegration:
    """Integration tests combining new operations."""

    def test_mask_then_background(self, test_image_file: str):
        """Mask + background: circular avatar on white."""
        state = PipelineState()
        state.add_op("load", test_image_file)
        state.add_op("mask", "circle")
        state.add_op("background", "white")
        image = state.materialize()
        assert image.size == (100, 80)
        # Corner: white background (circle mask made it transparent, then background filled)
        r, g, b, a = image.getpixel((0, 0))
        assert (r, g, b, a) == (255, 255, 255, 255)

    def test_canvas_overlay_composition(self, test_image_file: str):
        """Canvas + load + overlay composition."""
        state = PipelineState()
        state.add_op("canvas", "200x200", color="white", **{"as": "bg"})
        state.add_op("load", test_image_file, **{"as": "fg"})
        state.add_op("resize", "50%", on="fg")
        state.add_op("overlay", "bg", "fg", x=50, y=60)
        image = state.materialize()
        assert image.size == (200, 200)

    def test_colorize_sepia_workflow(self, test_image_file: str):
        """Full sepia workflow: grayscale → colorize."""
        state = PipelineState()
        state.add_op("load", test_image_file)
        state.add_op("grayscale")
        state.add_op("colorize", "#704214")
        state.add_op("resize", "50%")
        image = state.materialize()
        assert image.size == (50, 40)

    def test_hstack_with_gap_and_pad(self, test_image_file: str, second_image_file: str):
        """Hstack with gap, then pad result."""
        state = PipelineState()
        state.add_op("load", test_image_file, **{"as": "a"})
        state.add_op("load", second_image_file, **{"as": "b"})
        state.add_op("hstack", "a", "b", gap=10, gap_color="white")
        state.add_op("pad", 5, color="black")
        image = state.materialize()
        # (100+10+50) x 80 = 160x80, then +5 pad on all sides = 170x90
        assert image.size == (170, 90)
