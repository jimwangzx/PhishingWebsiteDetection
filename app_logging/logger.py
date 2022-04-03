import logging
import os


# Creating a logger function to log ML model building tasks
def create_log(log_file, filemode='w'):
	"""Creates a logger

	   Parameters:
	   log_file: str, name of a log file to create
	   filemode: str, mode in which log file will be created

	   Returns:
	   logger: a logger which outputs logging to a file and console
	"""
	# if os.path.isfile(f'{log_file}.log'):
	# 	os.remove(f'{log_file}.log')

	logging.basicConfig(level=logging.INFO, filemode=filemode)
	logger = logging.getLogger(__name__)

	c_handler = logging.StreamHandler()
	f_handler = logging.FileHandler(f'{log_file}.log')

	c_handler.setLevel(logging.INFO)
	f_handler.setLevel(logging.INFO)

	c_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
								 datefmt='%d-%b-%y %H:%M:%S')
	f_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
								 datefmt='%d-%b-%y %H:%M:%S')
	c_handler.setFormatter(c_format)
	f_handler.setFormatter(f_format)

	logger.addHandler(c_handler)
	logger.addHandler(f_handler)

	return logger