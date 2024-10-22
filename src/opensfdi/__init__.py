import logging
import sys
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--debug', action='store_true')

args, unknown = parser.parse_known_args()

args = vars(args)

DEBUG = args["debug"]

logger = logging.getLogger(__name__)

#formatter = logging.Formatter(fmt='%(threadName)s:%(message)s')
formatter = logging.Formatter(fmt='[%(levelname)s] %(message)s')

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)

logger.addHandler(handler)
logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)