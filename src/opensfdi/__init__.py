import logging
import sys

logger = logging.getLogger(__name__)

#formatter = logging.Formatter(fmt='%(threadName)s:%(message)s')
formatter = logging.Formatter(fmt='[%(levelname)s] %(message)s')

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)

logger.addHandler(handler)
logger.setLevel(logging.INFO)