# -*- coding: utf-8 -*-
import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.DEBUG)

log = logging.getLogger()

def create_log(filename):
    log = logging.getLogger()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                    datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(filename)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    # ch = logging.StreamHandler()
    # ch.setLevel(logging.INFO)
    # ch.setFormatter(formatter)

    # log.addHandler(ch)
    log.addHandler(fh)

    return log
