import logging


def get_logger(name):
    """pado logging setup"""
    # fixme: allow proper intialization in the future
    return logging.getLogger(name)
