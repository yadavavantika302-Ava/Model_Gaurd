"""Safe, metadata-driven inference for single-input image TFLite classifiers."""

from base64 import b64encode
from hashlib import sha256
from io import BytesIO

import numpy as np
try:
    import cv2
except ImportError:  # The generic image adapter does not require OpenCV.
    cv2 = None
try:
    from PIL import Image, ImageDraw, ImageFilter, ImageOps, UnidentifiedImageError
except ImportError:  # Keep the rest of ModelGuard navigable before new deps are installed.
    Image = ImageDraw = ImageFilter = ImageOps = None
    UnidentifiedImageError = OSError


MAX_MODEL_BYTES = 20 * 1024 * 1024
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# This profile is tied to the exact CanSat model bytes found beside the original
# deployment scripts. It prevents user-selected unit normalization from being
# quantized as binary pixels. The scripts are inference evidence, not training
# metadata; the class order remains identified as deployment-code-derived.
MODEL_PROFILES = {
    "ea9dbe938b412671416d3d037b3bde50e20cde0740a3fb12d9e7f8b6118fc0ca": {
        "name": "CanSat terrain_model_int8.tflite",
        "input_shape": [1, 224, 224, 3],
        "input_dtype": "uint8",
        "input_scale": 1.0,
        "input_zero_point": 0,
        "output_shape": [1, 6],
        "output_dtype": "uint8",
        "output_scale": 1 / 256,
        "output_zero_point": 0,
        "preprocessing": "raw",
        "resize_strategy": "stretch",
        "resize_interpolation": "opencv_linear",
        "color_order": "rgb",
        "output_interpretation": "probabilities",
        "class_names": ["barren", "building", "grass", "road", "vegetation", "water"],
        "source": "Exact model fingerprint plus adjacent image_detect.py and live.camera_detect.py inference code",
        "output_source": "TFLite graph ends with SOFTMAX, then quantizes the six-value output",
        "unresolved": "The model embeds no class-label metadata and the original training configuration was not found. The listed class order comes from adjacent deployment scripts, not the training artifact.",
    },
}


def _read_upload(upload, label, max_bytes, extensions):
    if upload is None or not upload.filename:
        raise ValueError(f"Choose a {label} file.")
    filename = upload.filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix not in extensions:
        raise ValueError(f"{label.title()} must use one of these formats: {', '.join(sorted(extensions))}.")
    data = upload.stream.read(max_bytes + 1)
    if not data:
        raise ValueError(f"The {label} file is empty.")
    if len(data) > max_bytes:
        raise ValueError(f"The {label} file must be no larger than {max_bytes // (1024 * 1024)} MB.")
    return data, filename


def _interpreter(model_bytes):
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            try:
                from tensorflow.lite import Interpreter
            except ImportError:
                try:
                    from tensorflow.lite.python.interpreter import Interpreter
                except ImportError as exc:
                    raise ValueError(
                        "A TensorFlow Lite runtime is missing. Install TensorFlow or ai-edge-litert "
                        "in the environment running ModelGuard, then restart the app."
                    ) from exc
    try:
        instance = Interpreter(model_content=model_bytes)
        instance.allocate_tensors()
        return instance
    except Exception as exc:
        raise ValueError(f"Could not open this TFLite model: {str(exc).splitlines()[0][:250]}") from exc


def _parse_labels(raw, class_count):
    labels = [line.strip() for line in raw.replace(",", "\n").splitlines() if line.strip()]
    if labels and len(labels) != class_count:
        raise ValueError(
            f"The model returns {class_count} class values, but you supplied {len(labels)} labels. "
            "Enter one label per output in the model's class order, or leave labels blank."
        )
    return labels or [f"Class {index}" for index in range(class_count)]


def _input_quantization(detail):
    params = detail.get("quantization_parameters") or {}
    scales = np.asarray(params.get("scales", []), dtype=np.float64)
    zero_points = np.asarray(params.get("zero_points", []), dtype=np.int64)
    if not scales.size:
        scale, zero_point = detail.get("quantization", (0.0, 0))
        scales = np.asarray([scale], dtype=np.float64) if scale else np.asarray([])
        zero_points = np.asarray([zero_point], dtype=np.int64) if scale else np.asarray([])
    return scales, zero_points, int(params.get("quantized_dimension", 0))


def _model_profile(model_bytes, input_detail, output_detail):
    profile = MODEL_PROFILES.get(sha256(model_bytes).hexdigest())
    if profile is None:
        return None
    checks = (
        (list(map(int, input_detail["shape"])), profile["input_shape"], "input shape"),
        (np.dtype(input_detail["dtype"]).name, profile["input_dtype"], "input dtype"),
        (list(map(int, output_detail["shape"])), profile["output_shape"], "output shape"),
        (np.dtype(output_detail["dtype"]).name, profile["output_dtype"], "output dtype"),
    )
    for actual, expected, label in checks:
        if actual != expected:
            raise ValueError(f"Recognized model fingerprint has unexpected {label}; refusing its saved preprocessing profile.")
    in_scales, in_zero_points, _ = _input_quantization(input_detail)
    out_scales, out_zero_points, _ = _input_quantization(output_detail)
    if (len(in_scales) != 1 or not np.isclose(in_scales[0], profile["input_scale"])
            or not len(in_zero_points) or int(in_zero_points[0]) != profile["input_zero_point"]):
        raise ValueError("Recognized model fingerprint has unexpected input quantization metadata.")
    if (len(out_scales) != 1 or not np.isclose(out_scales[0], profile["output_scale"])
            or not len(out_zero_points) or int(out_zero_points[0]) != profile["output_zero_point"]):
        raise ValueError("Recognized model fingerprint has unexpected output quantization metadata.")
    return profile


def _resolve_preprocessing(detail, requested):
    dtype = np.dtype(detail["dtype"])
    scales, zero_points, _ = _input_quantization(detail)
    domain = None
    if np.issubdtype(dtype, np.integer) and len(scales) == 1 and scales[0] > 0:
        bounds = np.iinfo(dtype)
        zero_point = int(zero_points[0]) if len(zero_points) else 0
        low = (bounds.min - zero_point) * scales[0]
        high = (bounds.max - zero_point) * scales[0]
        if abs(low) <= 0.05 and 200 <= high <= 300:
            domain = "raw"
        elif abs(low) <= 0.05 and 0.8 <= high <= 1.2:
            domain = "unit"
        elif -1.2 <= low <= -0.8 and 0.8 <= high <= 1.2:
            domain = "minus_one_one"
    if requested == "auto":
        if domain is None:
            raise ValueError(
                "The model input tensor does not identify a standard pixel range. "
                "Choose the training-time normalization explicitly, or check the model's preprocessing metadata."
            )
        return domain, f"Inferred from input tensor quantization range ({dtype.name})", False
    if requested not in {"raw", "unit", "minus_one_one"}:
        raise ValueError("Choose the model's training-time pixel normalization.")
    source = "Selected by user"
    if domain is not None and requested != domain:
        source += f"; tensor metadata suggests {domain.replace('_', ' ')} pixels"
    return requested, source, bool(domain is not None and requested != domain)


def _quantize(values, detail):
    dtype = np.dtype(detail["dtype"])
    if not np.issubdtype(dtype, np.integer):
        return values.astype(dtype)
    scales, zero_points, axis = _input_quantization(detail)
    if not scales.size or np.any(scales <= 0):
        raise ValueError(f"Integer input type {dtype.name} has no usable affine quantization scale.")
    if scales.size == 1:
        quantized = values / scales[0] + (int(zero_points[0]) if zero_points.size else 0)
    else:
        if not zero_points.size or len(zero_points) != len(scales):
            raise ValueError("The model's per-axis input quantization metadata is invalid.")
        axis = axis if axis >= 0 else values.ndim + axis
        if axis < 0 or axis >= values.ndim or values.shape[axis] != len(scales):
            raise ValueError("Per-axis quantization does not match the model input tensor shape.")
        shape = [1] * values.ndim
        shape[axis] = len(scales)
        quantized = values / scales.reshape(shape) + zero_points.reshape(shape)
    bounds = np.iinfo(dtype)
    return np.clip(np.rint(quantized), bounds.min, bounds.max).astype(dtype)


def _dequantize(values, detail):
    scales, zero_points, axis = _input_quantization(detail)
    if not scales.size:
        return values.astype(np.float64)
    if scales.size == 1:
        return (values.astype(np.float64) - (int(zero_points[0]) if zero_points.size else 0)) * scales[0]
    axis = axis if axis >= 0 else values.ndim + axis
    shape = [1] * values.ndim
    shape[axis] = len(scales)
    return (values.astype(np.float64) - zero_points.reshape(shape)) * scales.reshape(shape)


def _image_descriptors(image):
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]
    pixels = rgb.reshape(-1, 3)
    means = [float(value) for value in pixels.mean(axis=0)]
    stds = [float(value) for value in pixels.std(axis=0)]
    mins = [int(value) for value in pixels.min(axis=0)]
    maxs = [int(value) for value in pixels.max(axis=0)]
    histograms = []
    for channel in range(3):
        counts, _ = np.histogram(pixels[:, channel], bins=16, range=(0, 256))
        histograms.append([round(float(value) / max(len(pixels), 1), 5) for value in counts])

    # Coarse 5-bit RGB buckets make a small, deterministic dominant-colour summary.
    buckets = pixels // 32
    codes = buckets[:, 0] * 64 + buckets[:, 1] * 8 + buckets[:, 2]
    counts = np.bincount(codes, minlength=512)
    dominant = []
    for code in np.argsort(counts)[::-1][:5]:
        if counts[code] == 0:
            continue
        group = pixels[codes == code]
        color = np.median(group, axis=0).astype(int)
        dominant.append({
            "hex": "#%02x%02x%02x" % tuple(color),
            "rgb": [int(value) for value in color],
            "share": round(float(counts[code]) / len(pixels) * 100, 2),
        })

    # Keep auxiliary texture calculations bounded for very large source images.
    thumb = image.convert("L").resize((min(192, width), min(192, height)), Image.Resampling.BILINEAR)
    gray = np.asarray(thumb, dtype=np.float32) / 255.0
    mid_y, mid_x = gray.shape[0] // 2, gray.shape[1] // 2
    regions = {
        "top_left": gray[:max(mid_y, 1), :max(mid_x, 1)],
        "top_right": gray[:max(mid_y, 1), mid_x:],
        "bottom_left": gray[mid_y:, :max(mid_x, 1)],
        "bottom_right": gray[mid_y:, mid_x:],
    }
    quadrant_luminance = {
        name: round(float(region.mean() * 255.0), 2)
        for name, region in regions.items() if region.size
    }
    quantized = np.minimum((gray * 16).astype(np.uint8), 15)
    glcm = np.zeros((16, 16), dtype=np.float64)
    for first, second in ((quantized[:, :-1], quantized[:, 1:]), (quantized[:-1, :], quantized[1:, :])):
        np.add.at(glcm, (first.ravel(), second.ravel()), 1)
    total = glcm.sum()
    if total:
        glcm /= total
    levels = np.arange(16, dtype=np.float64)
    distance = np.abs(levels[:, None] - levels[None, :])
    contrast = float(np.sum(glcm * distance ** 2))
    homogeneity = float(np.sum(glcm / (1.0 + distance)))

    if min(gray.shape) >= 3:
        center = gray[1:-1, 1:-1]
        neighbors = [gray[:-2, :-2], gray[:-2, 1:-1], gray[:-2, 2:], gray[1:-1, 2:],
                     gray[2:, 2:], gray[2:, 1:-1], gray[2:, :-2], gray[1:-1, :-2]]
        bits = np.stack([neighbor >= center for neighbor in neighbors], axis=0)
        transitions = np.sum(bits != np.roll(bits, 1, axis=0), axis=0)
        lbp = np.where(transitions <= 2, np.sum(bits, axis=0), 9)
        lbp_hist = np.bincount(lbp.astype(np.uint8).ravel(), minlength=10).astype(np.float64)
        lbp_hist /= max(lbp_hist.sum(), 1)
        lbp_uniform = float(lbp_hist[:9].sum())
    else:
        lbp_uniform = None

    if min(gray.shape) > 1:
        gy, gx = np.gradient(gray)
    else:
        gy, gx = np.zeros_like(gray), np.zeros_like(gray)
    magnitude = np.sqrt(gx * gx + gy * gy)
    edge_density = float(np.mean(magnitude > 0.12))
    gradient_mean = float(magnitude.mean())
    gray_counts = np.bincount(np.minimum((gray * 16).astype(np.uint8), 15).ravel(), minlength=16)
    gray_probs = gray_counts / max(gray_counts.sum(), 1)
    gray_entropy = float(-np.sum(gray_probs[gray_probs > 0] * np.log2(gray_probs[gray_probs > 0])))
    orientation = np.arctan2(gy, gx)
    orientation_counts = np.histogram(orientation[magnitude > 0.04], bins=8, range=(-np.pi, np.pi))[0]
    orientation_entropy = None
    if orientation_counts.sum():
        orientation_probs = orientation_counts / orientation_counts.sum()
        orientation_entropy = float(-np.sum(orientation_probs[orientation_probs > 0] * np.log2(orientation_probs[orientation_probs > 0])))
    return {
        "width": width,
        "height": height,
        "format": image.format or "Decoded image",
        "channels": 3,
        "channel_means": dict(zip(["Red", "Green", "Blue"], [round(value, 2) for value in means])),
        "channel_stds": dict(zip(["Red", "Green", "Blue"], [round(value, 2) for value in stds])),
        "channel_mins": dict(zip(["Red", "Green", "Blue"], mins)),
        "channel_maxs": dict(zip(["Red", "Green", "Blue"], maxs)),
        "quadrant_luminance": quadrant_luminance,
        "histograms": dict(zip(["Red", "Green", "Blue"], histograms)),
        "dominant_colors": dominant,
        "glcm_contrast": round(contrast, 4),
        "glcm_homogeneity": round(homogeneity, 4),
        "grayscale_entropy_bits": round(gray_entropy, 4),
        "lbp_uniform_fraction": round(lbp_uniform, 4) if lbp_uniform is not None else None,
        "edge_density": round(edge_density, 4),
        "mean_gradient_magnitude": round(gradient_mean, 5),
        "edge_orientation_entropy_bits": round(orientation_entropy, 4) if orientation_entropy is not None else None,
    }


def _model_scores(interpreter, input_detail, output_detail, input_tensor, interpretation):
    interpreter.set_tensor(input_detail["index"], input_tensor)
    interpreter.invoke()
    raw = np.asarray(interpreter.get_tensor(output_detail["index"]))
    if not raw.size or not np.isfinite(raw.astype(np.float64)).all():
        raise ValueError("Model returned an empty or non-finite output.")
    if raw.ndim not in (1, 2) or (raw.ndim == 2 and raw.shape[0] != 1):
        raise ValueError(f"Expected one class-score vector; model returned shape {raw.shape}.")
    values = raw.reshape(-1).astype(np.float64)
    if np.issubdtype(raw.dtype, np.integer):
        values = _dequantize(raw, output_detail).reshape(-1)
    if len(values) < 2:
        raise ValueError("This output is not a multi-class score vector.")
    looks_probabilistic = (np.all(values >= -1e-6) and np.all(values <= 1.000001)
                           and np.isclose(values.sum(), 1.0, atol=0.03))
    selected = interpretation
    if selected == "auto":
        selected = "probabilities" if looks_probabilistic else "scores"
    if selected == "probabilities":
        if not looks_probabilistic:
            raise ValueError("Output does not look like a probability vector (bounded values summing near 1).")
        scores = np.clip(values, 0.0, 1.0)
        score_label = "Probability-like model output (calibration not assessed)"
    elif selected == "logits":
        if looks_probabilistic:
            scores = np.clip(values, 0.0, 1.0)
            selected = "probabilities"
            score_label = "Probability-like output; Softmax skipped (calibration not assessed)"
        else:
            shifted = values - values.max()
            exp_values = np.exp(shifted)
            scores = exp_values / exp_values.sum()
            score_label = "Softmax-derived score from selected logits"
    else:
        scores = values
        score_label = None
    return scores, selected, score_label, raw


def _occlusion_explanation(interpreter, input_detail, output_detail, tensor, scores, target_index,
                           interpretation, image_size, input_mode):
    height, width = tensor.shape[-3:-1]
    baseline_image = Image.new("RGB", (width, height), (128, 128, 128))
    baseline, _, _ = _preprocess(baseline_image, input_detail, input_mode, "stretch", "rgb")
    cell_h, cell_w = max(1, height // 6), max(1, width // 6)
    heat = np.zeros((6, 6), dtype=np.float32)
    reference = float(scores[target_index])
    for row in range(6):
        for column in range(6):
            altered = tensor.copy()
            y0, x0 = row * cell_h, column * cell_w
            y1 = height if row == 5 else min(height, y0 + cell_h)
            x1 = width if column == 5 else min(width, x0 + cell_w)
            altered[:, y0:y1, x0:x1, :] = baseline[:, y0:y1, x0:x1, :]
            changed, _, _, _ = _model_scores(interpreter, input_detail, output_detail, altered, interpretation)
            heat[row, column] = reference - float(changed[target_index])
    if not np.any(heat):
        return None
    scale = float(np.max(np.abs(heat)))
    if scale == 0:
        return None
    signed = Image.fromarray(np.clip((heat / scale + 1) * 127.5, 0, 255).astype(np.uint8), mode="L")
    signed = np.asarray(signed.resize(image_size, Image.Resampling.BILINEAR), dtype=np.float32)
    strength = np.abs(signed - 127.5) / 127.5
    color_map = np.zeros((signed.shape[0], signed.shape[1], 3), dtype=np.uint8)
    positive = signed >= 127.5
    color_map[positive] = [220, 55, 45]   # occlusion lowers score: visible region supported the class
    color_map[~positive] = [45, 105, 220] # occlusion raises score: visible region opposed the class
    overlay = Image.fromarray(color_map, mode="RGB")
    alpha = Image.fromarray(np.clip(strength * 175, 0, 175).astype(np.uint8), mode="L")
    return Image.composite(overlay, Image.new("RGB", image_size, (255, 255, 255)), alpha)


def _controlled_experiments(image, input_detail, output_detail, interpreter, mode, resize_strategy,
                            color_order, interpretation, target_index, original_score,
                            resize_interpolation="pillow_bilinear"):
    """Compare single altered images using the same model preprocessing and output path."""
    width, height = image.size
    gray = image.convert("L").convert("RGB")
    blurred = image.convert("RGB").filter(ImageFilter.GaussianBlur(radius=max(1.0, min(width, height) / 160)))
    edges = image.convert("RGB").filter(ImageFilter.FIND_EDGES).convert("L").convert("RGB")
    region = image.convert("RGB").copy()
    draw = ImageDraw.Draw(region)
    left, top, right, bottom = width // 3, height // 3, (2 * width) // 3, (2 * height) // 3
    draw.rectangle((left, top, right, bottom), fill=(128, 128, 128))
    variants = [
        ("Colour", "Grayscale, replicated across RGB channels", gray, "Low: removes chroma and changes the input distribution."),
        ("Texture", "Gaussian blur (radius scaled to image size)", blurred, "Low: suppresses fine texture and also softens edges and small shapes."),
        ("Shape and edges", "PIL edge representation converted to three channels", edges, "Very low: replaces natural image colours and textures; likely out of distribution."),
        ("Region", "Replace the central third with neutral gray", region, "Low: introduces an artificial patch and its boundary; interpretation is local to this mask."),
    ]
    results = []
    for category, transform, modified, reliability in variants:
        tensor, _, _ = _preprocess(modified, input_detail, mode, resize_strategy, color_order,
                                   resize_interpolation=resize_interpolation)
        changed, _, _, _ = _model_scores(interpreter, input_detail, output_detail, tensor, interpretation)
        value = float(changed[target_index])
        delta = value - float(original_score)
        results.append({
            "category": category, "transformation": transform,
            "original_score": float(original_score), "modified_score": value,
            "delta": delta, "absolute_delta": abs(delta), "reliability": reliability,
        })
    return results


def _preprocess(image, detail, mode, resize_strategy, color_order, return_diagnostics=False,
                resize_interpolation="pillow_bilinear"):
    shape = [int(value) for value in detail["shape"]]
    if len(shape) == 4:
        if shape[0] not in (1,):
            raise ValueError("This page supports one image at a time; the model batch size must be 1.")
        _, height, width, channels = shape
    elif len(shape) == 3:
        height, width, channels = shape
    else:
        raise ValueError(f"Expected a rank-3 or rank-4 image input; model input shape is {shape}.")
    if height <= 0 or width <= 0 or height > 4096 or width > 4096:
        raise ValueError(f"The model has unsupported or dynamic image dimensions: {shape}.")
    if channels not in (1, 3, 4):
        raise ValueError(f"Model expects {channels} channels. Supported image inputs use 1, 3, or 4 channels.")

    if channels == 1:
        image = image.convert("L")
    elif channels == 4:
        image = image.convert("RGBA")
    else:
        image = image.convert("RGB")
    if resize_strategy == "center_crop":
        crop_size = min(image.size)
        left = (image.width - crop_size) // 2
        top = (image.height - crop_size) // 2
        image = image.crop((left, top, left + crop_size, top + crop_size))
    elif resize_strategy != "stretch":
        raise ValueError("Choose the resize method used during training.")
    if resize_interpolation == "opencv_linear":
        if cv2 is None:
            raise ValueError("This recognized model uses OpenCV bilinear resizing. Install opencv-python-headless and restart ModelGuard.")
        values = cv2.resize(np.asarray(image), (width, height), interpolation=cv2.INTER_LINEAR)
    elif resize_interpolation == "pillow_bilinear":
        image = image.resize((width, height), Image.Resampling.BILINEAR)
        values = np.asarray(image)
    else:
        raise ValueError("Unsupported resize interpolation configured for this model.")
    if channels == 1:
        values = values[..., np.newaxis]
    values = values.astype(np.float32)

    if mode == "unit":
        values /= 255.0
    elif mode == "minus_one_one":
        values = values / 127.5 - 1.0
    elif mode != "raw":
        raise ValueError("Choose the image normalization used when this model was trained.")

    if color_order not in {"rgb", "bgr"}:
        raise ValueError("Choose the channel order used during training.")
    if color_order == "bgr" and channels >= 3:
        values = values[..., [2, 1, 0] + ([3] if channels == 4 else [])]
    normalized_min = float(values.min())
    normalized_max = float(values.max())
    dtype = np.dtype(detail["dtype"])
    if len(shape) == 4:
        values = values[np.newaxis, ...]
    values = _quantize(values, detail)
    expected = tuple(int(value) for value in detail["shape"])
    if values.shape != expected:
        raise ValueError(f"Preprocessed image shape {values.shape} does not match model input {expected}.")
    if return_diagnostics:
        return values, (width, height), dtype.name, {
            "normalized_min": normalized_min,
            "normalized_max": normalized_max,
            "tensor_min": float(values.min()),
            "tensor_max": float(values.max()),
            "tensor_dtype": values.dtype.name,
            "tensor_shape": list(values.shape),
        }
    return values, (width, height), dtype.name


def run_image_test(model_upload, image_upload, labels_text, resize_strategy, preprocessing, output_interpretation, color_order="rgb", true_label=""):
    model_bytes, model_name = _read_upload(model_upload, "TFLite model", MAX_MODEL_BYTES, {".tflite"})
    image_bytes, image_name = _read_upload(image_upload, "image", MAX_IMAGE_BYTES, ALLOWED_IMAGE_EXTENSIONS)
    if Image is None:
        raise ValueError("Image support is not installed. Run 'py -m pip install -r requirements.txt' and restart ModelGuard.")
    if len(model_bytes) < 8 or model_bytes[4:8] != b"TFL3":
        raise ValueError("This file does not have a valid TensorFlow Lite model identifier.")
    if resize_strategy not in {"stretch", "center_crop"}:
        raise ValueError("Select the image resize method used during model training.")
    if preprocessing not in {"auto", "raw", "unit", "minus_one_one"}:
        raise ValueError("Choose the image normalization used when this model was trained.")
    if color_order not in {"rgb", "bgr"}:
        raise ValueError("Choose the channel order used during training.")
    if output_interpretation not in {"auto", "probabilities", "logits", "scores"}:
        raise ValueError("Select how the model's output values should be interpreted.")

    try:
        with Image.open(BytesIO(image_bytes)) as source:
            if source.format not in {"JPEG", "PNG"}:
                raise ValueError("The image contents must be JPEG or PNG, regardless of the filename.")
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise ValueError("Image dimensions are too large. Use an image with at most 20 megapixels.")
            source_format = source.format
            source.verify()
        with Image.open(BytesIO(image_bytes)) as source:
            image = ImageOps.exif_transpose(source).copy()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Could not decode this image. Choose a valid JPG, JPEG, or PNG file.") from exc

    interpreter = _interpreter(model_bytes)
    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("This page supports TFLite models with exactly one input tensor and one output tensor.")
    input_detail, output_detail = inputs[0], outputs[0]
    model_profile = _model_profile(model_bytes, input_detail, output_detail)
    profile_warnings = []
    if model_profile:
        preprocessing_used = model_profile["preprocessing"]
        preprocessing_source = model_profile["source"]
        effective_resize = model_profile["resize_strategy"]
        effective_resize_interpolation = model_profile["resize_interpolation"]
        effective_color_order = model_profile["color_order"]
        effective_output_interpretation = model_profile["output_interpretation"]
        requested_settings = {
            "pixel range": (preprocessing, preprocessing_used),
            "resize method": (resize_strategy, effective_resize),
            "channel order": (color_order, effective_color_order),
        }
        for setting, (requested, effective) in requested_settings.items():
            if requested not in {"auto", effective}:
                profile_warnings.append(
                    f"Requested {setting} '{requested}' was replaced with '{effective}' based on the exact model profile."
                )
        if output_interpretation not in {"auto", effective_output_interpretation}:
            profile_warnings.append(
                f"Requested output interpretation '{output_interpretation}' was replaced with "
                f"'{effective_output_interpretation}' because the model graph contains SOFTMAX."
            )
        preprocessing_mismatch = bool(profile_warnings)
    else:
        preprocessing_used, preprocessing_source, preprocessing_mismatch = _resolve_preprocessing(input_detail, preprocessing)
        effective_resize = resize_strategy
        effective_resize_interpolation = "pillow_bilinear"
        effective_color_order = color_order
        effective_output_interpretation = output_interpretation

    raw_pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
    raw_channel_ranges = {
        channel: [int(raw_pixels[..., index].min()), int(raw_pixels[..., index].max())]
        for index, channel in enumerate(("Red", "Green", "Blue"))
    }
    input_tensor, resized_to, dtype_name, tensor_diagnostics = _preprocess(
        image, input_detail, preprocessing_used, effective_resize, effective_color_order,
        return_diagnostics=True, resize_interpolation=effective_resize_interpolation,
    )

    try:
        scores, interpretation_used, confidence_label, raw_output = _model_scores(
            interpreter, input_detail, output_detail, input_tensor, effective_output_interpretation
        )
    except Exception as exc:
        raise ValueError(f"Model inference failed: {str(exc).splitlines()[0][:250]}") from exc
    if model_profile:
        confidence_label = "TFLite graph SOFTMAX output (quantized; calibration not assessed)"
    confidence = float(scores.max()) if confidence_label else None

    if model_profile and not labels_text.strip():
        class_names = list(model_profile["class_names"])
        label_source = "Adjacent deployment script class order; training mapping is not embedded or verified"
    else:
        class_names = _parse_labels(labels_text, len(scores))
        label_source = "User-provided labels (training order not independently verified)" if labels_text.strip() else "Generic output indices; class mapping unknown"
    if model_profile and labels_text.strip():
        if class_names == model_profile["class_names"]:
            label_source = "User-provided labels match adjacent deployment scripts; training mapping is not verified"
        else:
            label_source = "User-provided labels conflict with adjacent deployment scripts; training mapping is not verified"
            profile_warnings.append("Supplied class order conflicts with the adjacent deployment scripts' class sequence.")
    order = np.argsort(scores)[::-1]
    preview = image.convert("RGB")
    preview.thumbnail((900, 700), Image.Resampling.LANCZOS)
    preview_buffer = BytesIO()
    preview.save(preview_buffer, format="JPEG", quality=88, optimize=True)
    preview_uri = "data:image/jpeg;base64," + b64encode(preview_buffer.getvalue()).decode("ascii")
    descriptors = _image_descriptors(image)
    descriptors["format"] = source_format
    target_index = int(order[0])
    controlled_experiments = []
    experiments_note = ""
    try:
        controlled_experiments = _controlled_experiments(
            image, input_detail, output_detail, interpreter, preprocessing_used,
            effective_resize, effective_color_order, interpretation_used, target_index,
            float(scores[target_index]), effective_resize_interpolation
        )
        experiments_note = (
            "Each row changes a single uploaded image and reruns this same model adapter. Score changes are "
            "conditional on these specific transformations; colour, texture, edges, and shape interact and "
            "the altered images may be outside the training distribution. These are not feature-importance percentages."
        )
    except Exception as exc:
        experiments_note = f"Controlled comparisons unavailable: {str(exc).splitlines()[0][:180]}"
    explanation_uri = None
    explanation_note = "Occlusion sensitivity could not be generated for this model input layout."
    if len(input_detail["shape"]) == 4:
        try:
            explanation = _occlusion_explanation(
                interpreter, input_detail, output_detail, input_tensor, scores, int(order[0]),
                interpretation_used, preview.size, preprocessing_used
            )
            if explanation is not None:
                explanation = Image.blend(preview, explanation, 0.45)
                explanation_buffer = BytesIO()
                explanation.save(explanation_buffer, format="JPEG", quality=88)
                explanation_uri = "data:image/jpeg;base64," + b64encode(explanation_buffer.getvalue()).decode("ascii")
                explanation_note = (
                    "Model-focused occlusion sensitivity. Red regions increased the target score when visible "
                    "(occluding them lowered it); blue regions increased the target score when occluded. "
                    "Magnitude is relative within this image and is not causal evidence."
                )
            else:
                explanation_note = "Neutral occlusion did not produce a measurable score change."
        except Exception as exc:
            explanation_note = f"Occlusion sensitivity unavailable: {str(exc).splitlines()[0][:180]}"
    selected_true_label = true_label.strip()
    if selected_true_label and selected_true_label not in class_names:
        raise ValueError("The supplied known label must exactly match one of the configured class labels.")
    truth_match = (selected_true_label == class_names[int(order[0])]) if selected_true_label else None
    q_scales, q_zero_points, q_axis = _input_quantization(input_detail)
    output_scales, output_zero_points, output_axis = _input_quantization(output_detail)
    if np.issubdtype(np.asarray(raw_output).dtype, np.integer):
        dequantized_output = _dequantize(raw_output, output_detail).reshape(-1)
    else:
        dequantized_output = np.asarray(raw_output, dtype=np.float64).reshape(-1)
    if model_profile:
        assumptions = [model_profile["unresolved"]]
        output_source = model_profile["output_source"]
    else:
        assumptions = [
            "Training-time image preprocessing is not encoded generally by TFLite tensor quantization; verify it from model documentation.",
            "Output semantics are inferred from the requested interpretation and probability-vector shape unless separately verified.",
        ]
        output_source = confidence_label or "Raw model values; output semantics are not verified."
    diagnostics = {
        "model_profile": model_profile["name"] if model_profile else "No matching model-specific profile",
        "raw_pixel_range": [int(raw_pixels.min()), int(raw_pixels.max())],
        "raw_channel_ranges": raw_channel_ranges,
        "normalized_tensor_range": [tensor_diagnostics["normalized_min"], tensor_diagnostics["normalized_max"]],
        "input_tensor_range": [tensor_diagnostics["tensor_min"], tensor_diagnostics["tensor_max"]],
        "input_tensor_dtype": tensor_diagnostics["tensor_dtype"],
        "input_tensor_shape": tensor_diagnostics["tensor_shape"],
        "input_quantization": {"scales": q_scales.tolist(), "zero_points": q_zero_points.tolist(), "axis": q_axis},
        "raw_model_output": np.asarray(raw_output).tolist(),
        "raw_model_output_dtype": np.asarray(raw_output).dtype.name,
        "raw_model_output_shape": list(np.asarray(raw_output).shape),
        "dequantized_output_values": [float(value) for value in dequantized_output],
        "interpreted_output_values": [float(value) for value in scores],
        "output_quantization": {"scales": output_scales.tolist(), "zero_points": output_zero_points.tolist(), "axis": output_axis},
        "output_interpretation": interpretation_used,
        "output_interpretation_source": output_source,
        "class_mapping": [{"index": index, "label": label} for index, label in enumerate(class_names)],
        "class_mapping_source": label_source,
        "preprocessing_assumptions": assumptions,
        "profile_warnings": profile_warnings,
        "requested": {"pixel_range": preprocessing, "resize": resize_strategy, "color_order": color_order, "output_interpretation": output_interpretation},
        "applied": {"pixel_range": preprocessing_used, "resize": effective_resize, "resize_interpolation": effective_resize_interpolation, "color_order": effective_color_order, "output_interpretation": interpretation_used},
    }
    return {
        "image_preview": preview_uri,
        "image_analysis": descriptors,
        "explanation_preview": explanation_uri,
        "explanation_note": explanation_note,
        "explanation_method": "Signed neutral-gray occlusion sensitivity (6 × 6 grid)",
        "explanation_limitations": "This shows spatial score sensitivity to gray replacement for this one image. Patch boundaries and replacement colour can affect results; it does not identify whether colour, texture, shape, or another learned pattern caused the response.",
        "controlled_experiments": controlled_experiments,
        "experiments_note": experiments_note,
        "target_score_label": confidence_label or "Raw model score; meaning is model-specific",
        "image_name": image_name,
        "model_name": model_name,
        "model_format": "TensorFlow Lite",
        "runtime": type(interpreter).__module__,
        "input_name": input_detail.get("name", "Unnamed input tensor"),
        "output_name": output_detail.get("name", "Unnamed output tensor"),
        "predicted_class": class_names[int(order[0])],
        "confidence": confidence,
        "confidence_label": confidence_label,
        "true_label": selected_true_label or None,
        "truth_match": truth_match,
        "preprocessing_used": preprocessing_used,
        "preprocessing_source": preprocessing_source,
        "preprocessing_mismatch": preprocessing_mismatch,
        "preprocessing_requested": preprocessing,
        "resize_strategy_used": effective_resize,
        "color_order": effective_color_order.upper(),
        "diagnostics": diagnostics,
        "input_quantization": {"scales": q_scales.tolist(), "zero_points": q_zero_points.tolist(), "axis": q_axis},
        "output_quantization": {"scales": output_scales.tolist(), "zero_points": output_zero_points.tolist(), "axis": output_axis},
        "output_interpretation": output_interpretation,
        "output_interpretation_used": interpretation_used,
        "label_source": label_source,
        "scores": [{"label": class_names[int(index)], "value": float(scores[index])} for index in order],
        "scores_are_probabilities": interpretation_used in {"probabilities", "logits"},
        "input_shape": list(input_detail["shape"]),
        "input_type": dtype_name,
        "resized_to": list(resized_to),
        "output_shape": list(raw_output.shape),
        "technical_checks": [
            "TFLite file parsed and tensors allocated",
            "Image decoded, resized, and converted to the model input type",
            "Input tensor converted using its dtype and affine quantization metadata where present",
            "Inference completed with finite output values",
            "Output interpretation checked against the selected mode",
        ],
    }
