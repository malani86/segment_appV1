SCRIPT_BASENAME = "quantify_droplets_batch"
VALID_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
DEFAULT_CHECKPOINT = "best_UNetDC_focal_model.pth"
DEFAULT_OUT_DIR = "quant_results"

DEFAULT_BATCH_SIZE = 8
DEFAULT_THRESHOLD = 0.3
DEFAULT_MIN_AREA = 29
DEFAULT_BACKGROUND_RADIUS = 50
DEFAULT_RESIZE_SIZE = 512
DEFAULT_PX_PER_MICRON = 5.0
DEFAULT_OVERLAY_ALPHA = 0.45
DEFAULT_SAVE_OVERLAYS = True
DEFAULT_SAVE_MASKS = True
DEFAULT_AUTO_QUANTIFICATION = True
DEFAULT_EXCEL_ENABLED = True
DEFAULT_HISTOGRAM_ENABLED = False
DEFAULT_USE_WATERSHED_COUNT = True
DEFAULT_TIFF_AS_PNG_STYLE = True
# Training loads one 2D RGB image per sample, so current-slice inference is the
# safest default for TIFF stacks. Max projection remains optional because a model
# trained on ordinary 2D images may behave differently on projected inputs.
DEFAULT_TIFF_STACK_MODE = "current_slice"
