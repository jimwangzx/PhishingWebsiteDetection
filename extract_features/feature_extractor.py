from urllib import parse
import ipaddress
import re
import posixpath
import requests
from extract_features.spf import get_spf_record, check_spf
import geoip2.database
from dns import resolver, reversename
from datetime import datetime
import whois
from bs4 import BeautifulSoup


class FeatureExtractor:
	"""Extracts features from a URL
	"""

	def __init__(self):
		pass

	def split_url(self, url):
		"""Returns a dictionary containing attributes extracted from a URL
		"""
		protocol, host, path, params, query, fragment = parse.urlparse(url.strip())
		result = {
			'url': host + path + params + query + fragment,
			'protocol': protocol,
			'host': host,
			'path': path,
			'file': posixpath.basename(path),
			'params': params,
			'query': query,
			'fragment': fragment
		}
		return result

	def count(self, text, character):
		"""Returns the amount of a certain character in a string
		"""
		return text.count(character)

	def count_vowels(self, text):
		"""Returns number of vowels in a string
		"""
		vowels = ['a', 'e', 'i', 'o', 'u']
		text_vowel_list = [i for i in list(text) if i.lower() in vowels]
		count = len(text_vowel_list)
		return count

	def length(self, text):
		"""Returns the length of a string
		"""
		return len(text)

	def valid_ip(self, host):
		"""Checks whether the domain has a valid IP format (IPv4 or IPv6)
		"""
		try:
			ipaddress.ip_address(host)
			return True
		except Exception:
			return False

	def valid_email(self, text):
		"""Checks whether there is an email in the text
		"""
		if re.findall(r'[\w\.-]+@[\w\.-]+', text):
			return True
		else:
			return False
		
	def count_tld(self, text):
		"""Returns amount of Top-Level Domains (TLD) present in the URL
		"""
		file = open('extract_features/tlds.txt', 'r')
		count = 0
		pattern = re.compile("[a-zA-Z0-9.]")
		for line in file:
			i = (text.lower().strip()).find(line.strip())
			while i > -1:
				if ((i + len(line) - 1) >= len(text)) or not pattern.match(text[i + len(line) - 1]):
					count += 1
				i = text.find(line.strip(), i + 1)
		file.close()
		return count

	def check_tld(self, text):
	    """Checks for presence of top-Level domains (TLD)
	    """
	    file = open('extract_features/tlds.txt', 'r')
	    pattern = re.compile("[a-zA-Z0-9.]")
	    for line in file:
	        i = (text.lower().strip()).find(line.strip())
	        while i > -1:
	            if ((i + len(line) - 1) >= len(text)) or not pattern.match(text[i + len(line) - 1]):
	                file.close()
	                return True
	            i = text.find(line.strip(), i + 1)
	    file.close()
	    return False

	def check_word_server_client(self, text):
		"""Checks whether the "server" or "client" keywords exist in the domain
		"""
		if "server" in text.lower() or "client" in text.lower():
			return True
		else:
			return False

	def count_params(self, text):
		"""Returns number of parameters
		"""
		return len(parse.parse_qs(text))

	def check_time_response(self, domain):
		"""Returns domain response time in seconds
		"""
		try:
			latency = requests.get(domain, headers={'Cache-Control': 'no-cache'}).elapsed.total_seconds()
			return latency
		except Exception:
			return -1
	
	def valid_spf(self, domain):
		"""Checks whether a registered domain has SPF and it is valid
		"""
		spf = get_spf_record(domain)
		if spf is not None:
			return check_spf(spf)
		else:
			return False

	def get_asn_number(self, url_dict):
		"""Returns ANS number associated with IP
		"""
		try:
			with geoip2.database.Reader('extract_features\GeoLite2-ASN.mmdb') as reader:
				if self.valid_ip(url_dict['host']):
					ip = url_dict['host']
				else:
					ip = resolver.query(url_dict['host'], 'A')
					ip = ip[0].to_text()
	
				if ip:
					response = reader.asn(ip)
					return response.autonomous_system_number
				else:
					return -1
		except Exception:
			return -1
	
	def time_activation_domain(self, url_dict):
		"""Returns time (in days) of domain activation
		"""
		if url_dict['host'].startswith("www."):
			url_dict['host'] = url_dict['host'][4:]
	
		try:
			result_whois = whois.whois(url_dict['host'])
			creation = result_whois.creation_date
			if isinstance(creation, list):
				d1 = creation[0]
			else:
				d1 = creation
			d2 = datetime.now()
			return abs((d2 - d1).days)
		except Exception:
			return -1
	
	def time_expiration_domain(self, url_dict):
		"""Returns time (in days) of domain expiration
		"""
		if url_dict['host'].startswith("www."):
			url_dict['host'] = url_dict['host'][4:]
	
		try:
			result_whois = whois.whois(url_dict['host'])
			expiration = result_whois.expiration_date
			if isinstance(expiration, list):
				d1 = expiration[0]
			else:
				d1 = expiration
			d2 = datetime.now()
			return abs((d1 - d2).days)
		except Exception:
			return -1
	
	def count_ips(self, url_dict):
		"""Returns number of resolved IPs (IPv4)
		"""
		if self.valid_ip(url_dict['host']):
			return 1
	
		try:
			answers = resolver.resolve(url_dict['host'], 'A')
			return len(answers)
		except Exception:
			return -1

	def count_name_servers(self, url_dict):
		"""Returns number of name servers (NS) resolved
		"""
		count = 0
		if self.count_ips(url_dict):
			try:
				answers = resolver.resolve(url_dict['host'], 'NS')
				return len(answers)
			except (resolver.NoAnswer, resolver.NXDOMAIN):
				split_host = url_dict['host'].split('.')
				while len(split_host) > 0:
					split_host.pop(0)
					supposed_domain = '.'.join(split_host)
					try:
						answers = resolver.resolve(supposed_domain, 'NS')
						count = len(answers)
						break
					except Exception:
						count = 0
			except Exception:
				count = 0
		return count
	
	
	def count_mx_servers(self, url_dict):
		"""Returns number of resolved MX servers
		"""
		count = 0
		if self.count_ips(url_dict):
			try:
				answers = resolver.resolve(url_dict['host'], 'MX')
				return len(answers)
			except (resolver.NoAnswer, resolver.NXDOMAIN):
				split_host = url_dict['host'].split('.')
				while len(split_host) > 0:
					split_host.pop(0)
					supposed_domain = '.'.join(split_host)
					try:
						answers = resolver.resolve(supposed_domain, 'MX')
						count = len(answers)
						break
					except Exception:
						count = 0
			except Exception:
				count = 0
		return count
	
	def extract_ttl(self, url_dict):
		"""Returns time-to-live (TTL) value associated with hostname
		"""
		try:
			ttl = resolver.resolve(url_dict['host']).rrset.ttl
			return ttl
		except Exception:
			return -1
	
	def check_ssl(self, url):
		"""Checks whether the SSL certificate is valid
		"""
		try:
			requests.get(url, verify=True, timeout=3)
			return True
		except Exception:
			return False
	
	def count_redirects(self, url):
		"""Returns number of redirects in a URL
		"""
		try:
			response = requests.get(url, timeout=3)
			if response.history:
				return len(response.history)
			else:
				return 0
		except Exception:
			return -1
	
	def google_search(self, url):
		"""Checks whether a URL is indexed in google
		"""
		user_agent = 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/48.0.2564.116 Safari/537.36'
		headers = {'User-Agent': user_agent}
	
		query = {'q': 'info:' + url}
		google = "https://www.google.com/search?" + parse.urlencode(query)
		try:
			data = requests.get(google, headers=headers)
		except Exception:
			return -1
		data.encoding = 'ISO-8859-1'
		soup = BeautifulSoup(str(data.content), "html.parser")
		try:
			(soup.find(id="rso").find("div").find("div").find("h3").find("a"))['href']
			return True
		except AttributeError:
			return False
	
	def check_shortener(self, dict_url):
		"""Checks whether a domain is a shortener
		"""
		file = open('extract_features/shorteners.txt', 'r')
		for line in file:
			with_www = "www." + line.strip()
			if line.strip() == dict_url['host'].lower() or with_www == dict_url['host'].lower():
				file.close()
				return True
		file.close()
		return False
	