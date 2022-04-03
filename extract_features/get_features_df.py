from extract_features.feature_extractor import FeatureExtractor
import pandas as pd
from app_logging.logger import create_log


def get_features(x):
	"""Creates features from a URL
	
	   Params:
	   x: str, this can be a URL or a csv/xlsx file

	   Returns:
	   features_df: dataframe, contains extracted features
	"""
	logger = create_log("Feature_Extractor_Logs/extraction", filemode='a+')

	if isinstance(x, str):
		features_df = pd.DataFrame({'URL': [x]})
	else:
		features_df = x
	logger.info("DataFrame has been created.")

	extractor_instance = FeatureExtractor()

	features_df['dict'] = features_df['URL'].apply(lambda x: extractor_instance.split_url(x))

	logger.info("Starting URL based feature extraction")
	try:
		features_df = get_url_features(features_df, extractor_instance)
		logger.info("End of URL based feature extraction")
	except:
		logger.critical("URL based feature extraction failed", exc_info=True)

	logger.info("Starting URL domain based feature extraction")
	try:
		features_df = get_url_domain_features(features_df, extractor_instance)
		logger.info("End of URL domain based feature extraction")
	except:
		logger.critical("URL domain based feature extraction failed", exc_info=True)

	logger.info("Starting URL directory based feature extraction")
	try:
		features_df = get_url_directory_features(features_df, extractor_instance)
		logger.info("End of URL directory based feature extraction")
	except:
		logger.critical("URL directory based feature extraction failed", exc_info=True)

	logger.info("Starting URL file name based feature extraction")
	try:
		features_df = get_url_file_features(features_df, extractor_instance)
		logger.info("End of URL file name based feature extraction")
	except:
		logger.critical("URL file name based feature extraction failed", exc_info=True)

	logger.info("Starting URL parameters based feature extraction")
	try:
		features_df = get_url_parameter_features(features_df, extractor_instance)
		logger.info("End of URL parameters based feature extraction")
	except:
		logger.critical("URL parameters based feature extraction failed", exc_info=True)

	logger.info("Starting resolving URL and external services based feature extraction")
	try:
		features_df = get_url_extra_features(features_df, extractor_instance)
		logger.info("End of resolving URL and external services based feature extraction")
	except:
		logger.critical("Resolving URL and external services based feature extraction failed", exc_info=True)

	return features_df


def get_url_features(features_df, extractor_instance):
	"""Creates URL based features

	   Params:
       features_df: dataframe, features' dataframe to fill in
       extractor_instance: instance of FeatureExtractor class

       Returns:
       features_df: dataframe
	"""
	feat_sign_dict = {
		'qty_dot_url': '.',
		'qty_hyphen_url': '-',
		'qty_underline_url': '_',
		'qty_slash_url': '/',
		'qty_questionmark_url': '?',
		'qty_equal_url': '=',
		'qty_at_url': '@',
		'qty_and_url': '&',
		'qty_exclamation_url': '!',
		'qty_space_url': ' ',
		'qty_tilde_url': '~',
		'qty_comma_url': ',',
		'qty_plus_url': '+',
		'qty_asterisk_url': '*',
		'qty_hashtag_url': '#',
		'qty_dollar_url': '$',
		'qty_percent_url': '%'
	}

	for feat in feat_sign_dict.keys():
		features_df[feat] = features_df['dict'].apply(lambda x: extractor_instance.count(x['url'],
																						 feat_sign_dict[feat]))

	features_df['qty_tld_url'] = features_df['dict'].apply(lambda x: extractor_instance.count_tld(x['url']))
	features_df['length_url'] = features_df['dict'].apply(lambda x: extractor_instance.length(x['url']))
	features_df['email_in_url'] = features_df['dict'].apply(lambda x: int(extractor_instance.valid_email(x['url'])))
	
	return features_df


def get_url_domain_features(features_df, extractor_instance):
	"""Creates URL domain based features

	   Params:
	   features_df: dataframe, features' dataframe to fill in 
	   extractor_instance: instance of FeatureExtractor class

	   Returns:
	   features_df: dataframe
	"""
	feat_sign_dict = {
		'qty_dot_domain': '.',
		'qty_hyphen_domain': '-',
		'qty_underline_domain': '_',
		'qty_slash_domain': '/',
		'qty_questionmark_domain': '?',
		'qty_equal_domain': '=',
		'qty_at_domain': '@',
		'qty_and_domain': '&',
		'qty_exclamation_domain': '!',
		'qty_space_domain': ' ',
		'qty_tilde_domain': '~',
		'qty_comma_domain': ',',
		'qty_plus_domain': '+',
		'qty_asterisk_domain': '*',
		'qty_hashtag_domain': '#',
		'qty_dollar_domain': '$',
		'qty_percent_domain': '%'
	}

	for feat in feat_sign_dict.keys():
		features_df[feat] = features_df['dict'].apply(lambda x: extractor_instance.count(x['host'],
																						 feat_sign_dict[feat]))

	features_df['qty_vowels_domain'] = features_df['dict'].apply(lambda x: extractor_instance.count_vowels(x['host']))
	features_df['domain_length'] = features_df['dict'].apply(lambda x: extractor_instance.length(x['host']))
	features_df['domain_in_ip'] = features_df['dict'].apply(lambda x: int(extractor_instance.valid_ip(x['host'])))
	features_df['server_client_domain'] = features_df['dict'].apply(lambda x: int(extractor_instance.\
																				  check_word_server_client(x['host'])))
	
	return features_df


def get_url_directory_features(features_df, extractor_instance):
	"""Creates URL directory based features

	   Params:
	   features_df: dataframe, features' dataframe to fill in 
	   extractor_instance: instance of FeatureExtractor class

	   Returns:
	   features_df: dataframe
	"""
	feat_sign_dict = {
		'qty_dot_directory': '.',
		'qty_hyphen_directory': '-',
		'qty_underline_directory': '_',
		'qty_slash_directory': '/',
		'qty_questionmark_directory': '?',
		'qty_equal_directory': '=',
		'qty_at_directory': '@',
		'qty_and_directory': '&',
		'qty_exclamation_directory': '!',
		'qty_space_directory': ' ',
		'qty_tilde_directory': '~',
		'qty_comma_directory': ',',
		'qty_plus_directory': '+',
		'qty_asterisk_directory': '*',
		'qty_hashtag_directory': '#',
		'qty_dollar_directory': '$',
		'qty_percent_directory': '%'
	}

	for feat in feat_sign_dict.keys():
		features_df[feat] = features_df['dict'].apply(lambda x: extractor_instance.count(x['path'], feat_sign_dict[feat]) 
			                                                    if x['path'] else -1)

	features_df['directory_length'] = features_df['dict'].apply(lambda x: extractor_instance.length(x['path']) 
		                                                                  if x['path'] else -1)
	
	return features_df


def get_url_file_features(features_df, extractor_instance):
	"""Creates URL file name based features

	   Params:
	   features_df: dataframe, features' dataframe to fill in 
	   extractor_instance: instance of FeatureExtractor class

	   Returns:
	   features_df: dataframe
	"""
	feat_sign_dict = {
		'qty_dot_file': '.',
		'qty_hyphen_file': '-',
		'qty_underline_file': '_',
		'qty_slash_file': '/',
		'qty_questionmark_file': '?',
		'qty_equal_file': '=',
		'qty_at_file': '@',
		'qty_and_file': '&',
		'qty_exclamation_file': '!',
		'qty_space_file': ' ',
		'qty_tilde_file': '~',
		'qty_comma_file': ',',
		'qty_plus_file': '+',
		'qty_asterisk_file': '*',
		'qty_hashtag_file': '#',
		'qty_dollar_file': '$',
		'qty_percent_file': '%'
	}

	for feat in feat_sign_dict.keys():
		features_df[feat] = features_df['dict'].apply(lambda x: extractor_instance.count(x['file'], feat_sign_dict[feat]) 
			                                                    if x['path'] else -1)

	features_df['file_length'] = features_df['dict'].apply(lambda x: extractor_instance.length(x['file']) 
		                                                                  if x['path'] else -1)
	
	return features_df


def get_url_parameter_features(features_df, extractor_instance):
	"""Creates URL parameters based features

	   Params:
	   features_df: dataframe, features' dataframe to fill in 
	   extractor_instance: instance of FeatureExtractor class

	   Returns:
	   features_df: dataframe
	"""
	feat_sign_dict = {
		'qty_dot_params': '.',
		'qty_hyphen_params': '-',
		'qty_underline_params': '_',
		'qty_slash_params': '/',
		'qty_questionmark_params': '?',
		'qty_equal_params': '=',
		'qty_at_params': '@',
		'qty_and_params': '&',
		'qty_exclamation_params': '!',
		'qty_space_params': ' ',
		'qty_tilde_params': '~',
		'qty_comma_params': ',',
		'qty_plus_params': '+',
		'qty_asterisk_params': '*',
		'qty_hashtag_params': '#',
		'qty_dollar_params': '$',
		'qty_percent_params': '%'
	}

	for feat in feat_sign_dict.keys():
		features_df[feat] = features_df['dict'].apply(lambda x: extractor_instance.count(x['query'], feat_sign_dict[feat])
			                                                    if x['query'] else -1)

	features_df['params_length'] = features_df['dict'].apply(lambda x: extractor_instance.length(x['query']) 
		                                                               if x['query'] else -1)
	features_df['tld_present_params'] = features_df['dict'].apply(lambda x: int(extractor_instance.check_tld(x['query'])) 
		                                                               if x['query'] else -1)
	features_df['qty_params'] = features_df['dict'].apply(lambda x: int(extractor_instance.count_params(x['query'])) 
		                                                            if x['query'] else -1)
	
	return features_df


def get_url_extra_features(features_df, extractor_instance):
	"""Creates resolving URL and extra services based features

	   Params:
	   features_df: dataframe, features' dataframe to fill in 
	   extractor_instance: instance of FeatureExtractor class

	   Returns:
	   features_df: dataframe
	"""
	features_df['time_response'] = features_df['dict'].apply(lambda x: extractor_instance.\
															 check_time_response(x['protocol'] + '://' + x['host']))
	features_df['domain_spf'] = features_df['dict'].apply(lambda x: int(extractor_instance.valid_spf(x['host'])))
	features_df['asn_ip'] = features_df['dict'].apply(lambda x: extractor_instance.get_asn_number(x))
	features_df['time_domain_activation'] = features_df['dict'].apply(lambda x: extractor_instance.\
																	  time_activation_domain(x))
	features_df['time_domain_expiration'] = features_df['dict'].apply(lambda x: extractor_instance.\
																	  time_expiration_domain(x))
	features_df['qty_ip_resolved'] = features_df['dict'].apply(lambda x: extractor_instance.count_ips(x))
	features_df['qty_nameservers'] = features_df['dict'].apply(lambda x: extractor_instance.count_name_servers(x))
	features_df['qty_mx_servers'] = features_df['dict'].apply(lambda x: extractor_instance.count_mx_servers(x))
	features_df['ttl_hostname'] = features_df['dict'].apply(lambda x: extractor_instance.extract_ttl(x))
	features_df['tls_ssl_certificate'] = features_df['URL'].apply(lambda x: int(extractor_instance.\
																				check_ssl(x)))
	features_df['qty_redirects'] = features_df['dict'].apply(lambda x: extractor_instance.\
															 count_redirects(x['protocol'] + '://' + x['url']))
	features_df['url_google_index'] = features_df['dict'].apply(lambda x: int(extractor_instance.\
																			  google_search(x['url'])))
	features_df['domain_google_index'] = features_df['dict'].apply(lambda x: int(extractor_instance.\
																				 google_search(x['host'])))
	features_df['url_shortened'] = features_df['dict'].apply(lambda x: int(extractor_instance.check_shortener(x)))

	features_df.drop(columns=['URL', 'dict'], inplace=True)
	
	return features_df
