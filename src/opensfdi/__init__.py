import logging
import sys
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

logger = logging.getLogger(__name__)

#formatter = logging.Formatter(fmt='%(threadName)s:%(message)s')
formatter = logging.Formatter(fmt='[%(levelname)s] %(message)s')

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)

logger.addHandler(handler)
logger.setLevel(logging.INFO)
