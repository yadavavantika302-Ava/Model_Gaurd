# ModelGuard

ModelGuard is a local Flask app with separate workspaces for tabular model checking and training, plus **Image Model Testing** for TensorFlow Lite image classifiers.

## Run

1. Install the core packages with `python -m pip install -r requirements.txt`.
2. For `.tflite` inference, install a compatible runtime in the same Python environment. For example, `python -m pip install ai-edge-litert` or install TensorFlow if your Python version/platform supports it.
3. On Windows, run `run_offline_app.bat`.
4. Open <http://127.0.0.1:5000> and choose a workflow.

For Windows PowerShell, from the `Modelguard` folder:

```powershell
py -3.13 -m pip install -r requirements.txt
py -3.13 -m pip install ai-edge-litert
.\run_offline_app.bat
```

If `ai-edge-litert` is not available for the Python version in use, install a supported TensorFlow build and launch the app with that same Python environment.

## Model checking

The checker accepts `.keras` models loaded in safe mode, `.tflite` models with one input and output (requires a Lite runtime), and `.onnx` tabular models with one input and output (requires `onnxruntime`). CSV inputs must match the model's feature names, order, shape, and expected types. Image inputs are handled on the separate Image Model Testing page.

## Image model testing

Open **Test an Image Model**, upload a `.tflite` model and a JPG, JPEG, or PNG image, then select the resize and channel order used during training. Integer tensors can use automatic normalization selection when their affine quantization parameters describe a standard pixel range. Float models require a known range from their training pipeline. The app quantizes/dequantizes using tensor scales and zero points, including per-axis metadata. It checks output vectors for probability-like values and does not apply Softmax a second time when they already look probabilistic.

Enter class labels in their exact output order, one per line (or comma-separated). If no labels are provided, outputs are shown as generic class indices. The page separates image-level measurements (RGB histograms and dominant colours, LBP/GLCM texture summaries, grayscale entropy, and gradient/edge summaries), model explanation, and prediction validation. A signed 6 × 6 neutral-gray occlusion map shows spatial score sensitivity for compatible rank-4 inputs. Red means the visible region supported the selected class score; blue means hiding that region raised the score. Controlled experiments compare the target score for grayscale, Gaussian blur, an edge representation, and a central gray mask using the same TFLite preprocessing and output interpretation. These are individual modified-input comparisons, not percentages of feature importance. Transformations can alter multiple visual properties and may create out-of-distribution inputs; their score changes are conditional on the chosen transformations. Image descriptors are not proof of what the model learned. An optional known label is compared with the single prediction; it does not produce an accuracy metric.

For the exact CanSat `terrain_model_int8.tflite` fingerprint, ModelGuard applies the neighboring deployment scripts' preprocessing: RGB, stretch resize to 224 × 224 using OpenCV `INTER_LINEAR`, and raw `uint8` pixels. The model input metadata is `[1,224,224,3]`, `uint8`, scale `1`, zero point `0`; the graph performs its own internal input conversion. Its output is `[1,6]`, `uint8`, scale `1/256`, zero point `0`, and the graph ends with `SOFTMAX` followed by quantization. A conflicting submitted pixel range is explicitly reported and the recognized model profile is applied. The diagnostics panel displays raw pixels, actual tensor values, quantized output, interpretation, and label index mapping. The neighboring scripts list `barren, building, grass, road, vegetation, water`; this order is deployment-code-derived, not verified from an embedded training label map. The training configuration was not found, so that mapping remains a stated limitation.

This page currently supports only single-image TensorFlow Lite classifiers with one rank-3 or rank-4 image input and one multi-class score vector. Gradient-based Grad-CAM and Integrated Gradients are not available through the TFLite inference interface used here. The page does not provide image adapters for ONNX, Keras, PyTorch, or scikit-learn; those formats need their own architecture- and preprocessing-aware adapters. Object detection, segmentation, dynamic image dimensions, and multi-input/multi-output models are unsupported. Do not assume output label order or training preprocessing unless verified from training/export materials.

### CanSat terrain sample diagnosis

The model and `dataset/grass/meadow_016.jpg` are in the adjacent CanSat project directory, not bundled inside this workspace. The real-model regression test expects `grass` from that dataset folder. Run it by setting `MODELGUARD_CAN_SAT_MODEL` and `MODELGUARD_MEADOW_IMAGE` to those files before invoking unittest. This is a single-image regression check, not a model accuracy evaluation.

Run the image inference tests from the project folder with `py -3.13 -m unittest discover -s tests -v`. Most tests use a mock TFLite interpreter; the optional environment-configured regression uses the real model and labelled meadow image.

Pickle/joblib (`.pkl`, `.pickle`, `.joblib`, `.sav`), legacy Keras H5 (`.h5`), and TorchScript uploads are blocked because deserialization can execute code. ModelGuard does not deserialize arbitrary uploaded model files. A pipeline generated by ModelGuard training is available for checking through the saved training run selector; the app only loads its own generated artifact from its internal directory. Convert other trusted sklearn models to ONNX or a supported safe format first.

When available, SHAP permutation values are calculated for supported tabular models. Otherwise, the checker reports exposed native feature importance or a labeled one-feature-at-a-time sensitivity estimate. Missing attributions are not fabricated. These explanations describe model behavior and do not establish causation.

## Model training

Upload a CSV and inspect sample rows, columns, data types, missing values, and duplicates before training. Choose classification, regression, or clustering. Supervised workflows exclude the selected target from model inputs. Classification and regression use random holdouts, stratifying classification when possible, or an optional chronological holdout by a parseable date column. Preprocessing is fitted on training rows only: numeric values are median-imputed and scaled; categorical values are mode-imputed and one-hot encoded.

Available estimators include linear/ridge/lasso regression, logistic regression, decision trees, random forests, gradient boosting, SVM, KNN, KMeans, DBSCAN, and configurable scikit-learn neural networks. Neural network settings cover hidden layer sizes, activation, learning rate, batch size, epochs, and L2 regularization. Keras training and optional XGBoost/LightGBM are not included in this build.

Each completed run saves a fitted pipeline, JSON report, and holdout predictions under `models/training_runs/` and `reports/training_runs/`. Recent runs are listed in the training workspace. The report includes real holdout metrics for supervised tasks; clustering reports the number of clusters, noise rows, and silhouette score when defined.

## Limits and local security

The image testing route limits a request to 24 MB, the model to 20 MB, the image to 8 MB, and images to 20 megapixels. Existing CSV upload limits are unchanged. CSV readers process at most the first 100,000 rows. Training is capped at 50,000 rows and 100 columns per run; DBSCAN is capped at 20,000 rows. Tree count, network size, and training iterations are bounded. Supported checker runtimes run in the local app process; this version has no OS-level sandbox or multi-user isolation. Use only model files from sources you trust.
