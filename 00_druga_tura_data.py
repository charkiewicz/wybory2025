import requests
from bs4 import BeautifulSoup
from bs4.element import Tag, NavigableString # Import Tag and NavigableString
import pandas as pd
import time
import re
import unicodedata
import os
from multiprocessing import Pool, Manager 

from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException

BASE_URL = "https://wybory.gov.pl"
ELECTION_PATH_PREFIX = "/prezydent2025/pl" 
ROUND_PATH_SEGMENT_RESULTS = "/2" 
ROUND_PATH_SEGMENT_OBKW = "/2"    

HIERARCHY_TABLE_ID = "DataTables_Table_0"
OBKW_LIST_TABLE_ID = "DataTables_Table_1"

POWIAT_MNPP_PATTERN = rf"{ELECTION_PATH_PREFIX}{ROUND_PATH_SEGMENT_RESULTS}/wynik/pow/\d+"
GMINA_PATTERN = rf"{ELECTION_PATH_PREFIX}{ROUND_PATH_SEGMENT_RESULTS}/wynik/gm/\d+"
PANSTWO_PATTERN = rf"{ELECTION_PATH_PREFIX}{ROUND_PATH_SEGMENT_RESULTS}/wynik/pan/[A-Z]{{2,3}}"
ZAGRANICA_OKREG_PATTERN = rf"{ELECTION_PATH_PREFIX}{ROUND_PATH_SEGMENT_RESULTS}/wynik/okr/90\d{{4}}"
OBKW_LINK_PATTERN = rf"{ELECTION_PATH_PREFIX}/obkw{ROUND_PATH_SEGMENT_OBKW}/\d+"

WOJEWODZTWA_CONFIG = [
    ("Województwo dolnośląskie", f"{BASE_URL}{ELECTION_PATH_PREFIX}{ROUND_PATH_SEGMENT_RESULTS}/wynik/woj/020000"),
    ("Województwo kujawsko-pomorskie", f"{BASE_URL}{ELECTION_PATH_PREFIX}{ROUND_PATH_SEGMENT_RESULTS}/wynik/woj/040000"),
    ("Województwo lubelskie", f"{BASE_URL}{ELECTION_PATH_PREFIX}{ROUND_PATH_SEGMENT_RESULTS}/wynik/woj/060000"),
    ("Województwo lubuskie", f"{BASE_URL}{ELECTION_PATH_PREFIX}{ROUND_PATH_SEGMENT_RESULTS}/wynik/woj/080000"),
    ("Województwo łódzkie", f"{BASE_URL}{ELECTION_PATH_PREFIX}{ROUND_PATH_SEGMENT_RESULTS}/wynik/woj/100000"),
    ("Województwo małopolskie", f"{BASE_URL}{ELECTION_PATH_PREFIX}{ROUND_PATH_SEGMENT_RESULTS}/wynik/woj/120000"),
    ("Województwo mazowieckie", f"{BASE_URL}{ELECTION_PATH_PREFIX}{ROUND_PATH_SEGMENT_RESULTS}/wynik/woj/140000"),
    ("Województwo opolskie", f"{BASE_URL}{ELECTION_PATH_PREFIX}{ROUND_PATH_SEGMENT_RESULTS}/wynik/woj/160000"),
    ("Województwo podkarpackie", f"{BASE_URL}{ELECTION_PATH_PREFIX}{ROUND_PATH_SEGMENT_RESULTS}/wynik/woj/180000"),
    ("Województwo podlaskie", f"{BASE_URL}{ELECTION_PATH_PREFIX}{ROUND_PATH_SEGMENT_RESULTS}/wynik/woj/200000"),
    ("Województwo pomorskie", f"{BASE_URL}{ELECTION_PATH_PREFIX}{ROUND_PATH_SEGMENT_RESULTS}/wynik/woj/220000"),
    ("Województwo śląskie", f"{BASE_URL}{ELECTION_PATH_PREFIX}{ROUND_PATH_SEGMENT_RESULTS}/wynik/woj/240000"),
    ("Województwo świętokrzyskie", f"{BASE_URL}{ELECTION_PATH_PREFIX}{ROUND_PATH_SEGMENT_RESULTS}/wynik/woj/260000"),
    ("Województwo warmińsko-mazurskie", f"{BASE_URL}{ELECTION_PATH_PREFIX}{ROUND_PATH_SEGMENT_RESULTS}/wynik/woj/280000"),
    ("Województwo wielkopolskie", f"{BASE_URL}{ELECTION_PATH_PREFIX}{ROUND_PATH_SEGMENT_RESULTS}/wynik/woj/300000"),
    ("Województwo zachodniopomorskie", f"{BASE_URL}{ELECTION_PATH_PREFIX}{ROUND_PATH_SEGMENT_RESULTS}/wynik/woj/320000"),
]
ZAGRANICA_CONFIG = {"name": "Zagranica", "url": f"{BASE_URL}{ELECTION_PATH_PREFIX}{ROUND_PATH_SEGMENT_RESULTS}/wynik/okr/900000", "level_key": "Zagranica_Main_Name", "path_name_override": "Zagranica"}
STATKI_CONFIG = {"name": "Statki", "url": f"{BASE_URL}{ELECTION_PATH_PREFIX}{ROUND_PATH_SEGMENT_RESULTS}/wynik/okr/910000", "level_key": "Statki_Name", "path_name_override": "Statki"}
CHROMEDRIVER_PATH = "./chromedriver-win64/chromedriver.exe"
GECKODRIVER_PATH = "geckodriver.exe" 
BROWSER_TO_USE = "chrome"

SUMMARY_FIELD_CONFIG = [
    ("liczba_kart_otrzymanych", "1", ["kart do gosowania otrzymanych przez obwodowa komisje wyborcza"]), 
    ("liczba_uprawnionych", "2", ["wyborcow uprawnionych do gosowania (umieszczonych w spisie"]), 
    ("liczba_kart_niewykorzystanych", "3", ["niewykorzystanych kart do gosowania"]),
    ("liczba_kart_wydanych_lokal", "4", ["wydano karty do gosowania w lokalu wyborczym (liczba podpisow"]),
    ("liczba_pakietow_wyslanych", "5", ["ktorym wysano pakiety wyborcze"]), 
    ("liczba_kart_wydanych_lacznie", "6", ["wydano karty do gosowania w lokalu wyborczym oraz w gosowaniu korespondencyjnym (acznie)"]), 
    ("liczba_glos_pelnnomocnik", "7", ["liczba wyborcow gosujacych przez penomocnika (liczba kart do gosowania wydanych"]),
    ("liczba_glos_zaswiadczenie", "8", ["liczba wyborcow gosujacych na podstawie zaswiadczenia o prawie do gosowania"]),
    ("liczba_kopert_zwrotnych", "9", ["liczba otrzymanych kopert zwrotnych w gosowaniu korespondencyjnym"]),
    ("koperty_bez_oswiadczenia", "9a", ["kopert zwrotnych w gosowaniu korespondencyjnym, w ktorych nie byo oswiadczenia o osobistym i tajnym oddaniu gosu"]),
    ("koperty_oswiadczenie_niepodpisane", "9b", ["kopert zwrotnych w gosowaniu korespondencyjnym, w ktorych oswiadczenie nie byo podpisane przez wyborce"]),
    ("koperty_bez_koperty_na_karte", "9c", ["kopert zwrotnych w gosowaniu korespondencyjnym, w ktorych nie byo koperty na karte do gosowania"]),
    ("koperty_niezaklejona_koperta_na_karte", "9d", ["kopert zwrotnych w gosowaniu korespondencyjnym, w ktorych znajdowaa sie niezaklejona koperta na karte do gosowania"]),
    ("koperty_na_karte_wrzucone_do_urny", "9e", ["liczba kopert na karte do gosowania w gosowaniu korespondencyjnym wrzuconych do urny"]),
    ("liczba_kart_wyjetych_z_urny", "10", ["liczba kart wyjetych z urny"]),
    ("karty_z_kopert_korespondencyjnych", "10a", ["w tym liczba kart wyjetych z kopert na karte do gosowania w gosowaniu korespondencyjnym"]),
    ("liczba_kart_niewaznych", "11", ["kart niewaznych (bez pieczeci obwodowej komisji wyborczej"]), 
    ("liczba_kart_waznych", "12", ["kart waznych"]), 
    ("liczba_glosow_niewaznych", "13", ["liczba gosow niewaznych (z kart waznych)"]), 
    ("glosy_niewazne_przyczyna_A", "13a", ["w tym z powodu postawienia znaku x obok nazwisk obu kandydatow"]),
    ("glosy_niewazne_przyczyna_B", "13b", ["w tym z powodu niepostawienia znaku x obok nazwiska zadnego kandydata"]),
    ("glosy_niewazne_przyczyna_C", "13c", ["w tym z powodu postawienia znaku x wyacznie obok nazwiska skreslonego kandydata"]), 
    ("liczba_glosow_waznych_na_kandydatow", "14", ["liczba gosow waznych oddanych acznie na obu kandydatow (z kart waznych)"]), 
]

def normalize_text(text):
    if text is None: return ""
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def _extract_number_from_text(text_content):
    if not text_content: return None
    match = re.search(r'(\d[\d\s]*\d|\d)$', text_content.strip())
    if match:
        num_str = match.group(1).replace(' ', '').replace('\xa0', '')
        try: return int(num_str)
        except ValueError:
            if text_content.strip().replace(' ', '').replace('\xa0', '').isdigit():
                return int(text_content.strip().replace(' ', '').replace('\xa0', ''))
            return None
    if text_content.strip().replace(' ', '').replace('\xa0', '').isdigit():
         return int(text_content.strip().replace(' ', '').replace('\xa0', ''))
    return None

def _get_value_from_summary_table_multi(list_of_table_soups, item_number_target, field_label_parts, field_id_for_debug="field"):
    for table_idx, summary_table_soup in enumerate(list_of_table_soups):
        for r_idx, row in enumerate(summary_table_soup.find_all('tr')):
            cells = row.find_all('td')
            if len(cells) == 3:
                item_no_cell_text_normalized = normalize_text(cells[0].get_text(strip=True))
                if item_no_cell_text_normalized.lower() == item_number_target.lower():
                    description_cell_text_normalized = normalize_text(cells[1].get_text(separator=' ', strip=True))
                    all_parts_found_in_description = True 
                    if field_label_parts: 
                        all_parts_found_in_description = all(part.lower() in description_cell_text_normalized.lower() for part in field_label_parts)
                    if all_parts_found_in_description:
                        num = _extract_number_from_text(normalize_text(cells[2].get_text(separator=' ', strip=True)))
                        return num
                    return None 
    return None

def get_webdriver():
    options = None
    if BROWSER_TO_USE == "chrome":
        options = webdriver.ChromeOptions(); options.add_argument("--headless"); options.add_argument("--disable-gpu"); options.add_argument("--no-sandbox"); options.add_argument("--log-level=3"); options.add_argument("--disable-dev-shm-usage"); options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"); prefs = {"profile.managed_default_content_settings.images": 2}; options.add_experimental_option("prefs", prefs)
        options.add_experimental_option('excludeSwitches', ['enable-logging']) 
        try: 
            service = ChromeService(executable_path=CHROMEDRIVER_PATH) 
            driver = webdriver.Chrome(service=service, options=options)
        except WebDriverException as e_chrome: print(f"Error starting Chrome WebDriver: {e_chrome}"); raise
    elif BROWSER_TO_USE == "firefox":
        options = webdriver.FirefoxOptions(); options.add_argument("--headless")
        try: 
            service = FirefoxService(executable_path=GECKODRIVER_PATH)
            driver = webdriver.Firefox(service=service, options=options)
        except Exception as e_ff: print(f"Error starting Firefox WebDriver: {e_ff}"); raise
    else: raise ValueError(f"Unsupported browser: {BROWSER_TO_USE}")
    return driver

def safe_driver_get(driver, url):
    try: driver.get(url); _ = driver.title; return True
    except WebDriverException as e: print(f"  WebDriverException during driver.get({url}): {e}"); return False
    except Exception as e_gen: print(f"  Unexpected general exception during driver.get({url}): {e_gen}"); return False

def get_links_from_ul_columns2(driver, page_url, link_regex_pattern):
    list_of_tuples = [] 
    if not safe_driver_get(driver, page_url): return []
    try:
        try: WebDriverWait(driver, 7).until(EC.presence_of_element_located((By.CSS_SELECTOR, "ul.columns2 li a")))
        except TimeoutException:
            try: WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.TAG_NAME, "ul")))
            except TimeoutException: pass
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        ul_specific = soup.find('ul', class_='columns2')
        if isinstance(ul_specific, Tag): # Check if ul_specific is a Tag
            for li_item in ul_specific.find_all('li', recursive=False):
                link_tag = li_item.find('a', href=True, recursive=False)
                if isinstance(link_tag, Tag): # Check if link_tag is a Tag
                    href_val = link_tag.get('href')
                    if isinstance(href_val, str) and re.search(link_regex_pattern, href_val):
                        full_url = BASE_URL + href_val if href_val.startswith('/') else href_val
                        list_of_tuples.append((normalize_text(link_tag.get_text()), full_url))
            if list_of_tuples: return list_of_tuples 
        
        all_uls = soup.find_all('ul')
        for potential_ul in all_uls:
            if potential_ul == ul_specific: continue # Already processed or not a Tag
            if isinstance(potential_ul, Tag): # Check if potential_ul is a Tag
                temp_links_fallback = []
                for li_item_fb in potential_ul.find_all('li', recursive=False):
                    link_tag_fb = li_item_fb.find('a', href=True, recursive=False)
                    if isinstance(link_tag_fb, Tag): # Check if link_tag_fb is a Tag
                        href_fb_val = link_tag_fb.get('href')
                        if isinstance(href_fb_val, str) and re.search(link_regex_pattern, href_fb_val):
                            full_url_fb = BASE_URL + href_fb_val if href_fb_val.startswith('/') else href_fb_val
                            temp_links_fallback.append((normalize_text(link_tag_fb.get_text()), full_url_fb))
                if temp_links_fallback: return temp_links_fallback
    except Exception as e: print(f"    Generic error processing ULs on {page_url}: {e}")
    return list_of_tuples 

def get_obkw_links_from_page(driver, page_url):
    obkw_links = []
    if not safe_driver_get(driver, page_url): return []
    try:
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, OBKW_LIST_TABLE_ID)))
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        table = soup.find('table', id=OBKW_LIST_TABLE_ID)
        if isinstance(table, Tag): # Check if table is a Tag
            obkw_url_prefix_for_selector = ELECTION_PATH_PREFIX + "/obkw"
            for link_tag in table.select(f'tbody tr td:first-of-type a[href*="{obkw_url_prefix_for_selector}"]'):
                if isinstance(link_tag, Tag): # link_tag from select is always a Tag if found
                    href_val = link_tag.get('href')
                    if isinstance(href_val, str) and re.search(OBKW_LINK_PATTERN, href_val):
                        full_url = BASE_URL + href_val if href_val.startswith("/") else href_val
                        if full_url not in obkw_links: obkw_links.append(full_url)
    except TimeoutException: print(f"    Timeout waiting for OBKW list table {OBKW_LIST_TABLE_ID} on {page_url}.")
    except Exception as e: print(f"    Error in get_obkw_links_from_page: {e}")
    return obkw_links

def get_hierarchical_links_from_page(driver, page_url, link_regex_pattern, table_id):
    links = []
    if not safe_driver_get(driver, page_url): return []
    try:
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, table_id)))
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        table = soup.find('table', id=table_id)
        if isinstance(table, Tag): # Check if table is a Tag
            for link_tag in table.select('tbody tr td:first-of-type a[href]'):
                if isinstance(link_tag, Tag): # link_tag from select is always a Tag if found
                    href_val = link_tag.get('href')
                    if isinstance(href_val, str) and re.search(link_regex_pattern, href_val):
                        full_url = BASE_URL + href_val if href_val.startswith('/') else href_val
                        links.append((normalize_text(link_tag.get_text()), full_url))
        elif not table: # table is None
            try:
                table_el = driver.find_element(By.ID, table_id)
                for row_el in table_el.find_elements(By.CSS_SELECTOR, "tbody tr"):
                    link_el = row_el.find_element(By.CSS_SELECTOR, "td:first-of-type a")
                    href_val = link_el.get_attribute('href') # get_attribute returns string or None
                    if isinstance(href_val, str) and re.search(link_regex_pattern, href_val):
                        full_url = BASE_URL + href_val if href_val.startswith('/') else href_val
                        links.append((normalize_text(link_el.text), full_url))
            except NoSuchElementException: print(f"    Table {table_id} not found by Selenium either.")
    except TimeoutException: print(f"    Timeout waiting for hierarchy table {table_id} on {page_url}.")
    except Exception as e: print(f"    Error in get_hierarchical_links_from_page: {e}")
    return links

def extract_commission_name_from_title(title_text):
    normalized_title = normalize_text(title_text)
    match = re.search(r"Obwodowa Komisja Wyborcza numer \d+ w .+? głosowaniu\s*(.*)", normalized_title, re.IGNORECASE)
    if match and match.group(1): return match.group(1).strip()
    simple_prefix_match = re.match(r"Obwodowa Komisja Wyborcza numer \d+\s*(.*)", normalized_title, re.IGNORECASE)
    if simple_prefix_match and simple_prefix_match.group(1):
        name_part = simple_prefix_match.group(1)
        name_part = re.sub(r"^w\s+(pierwszym|drugim|ponownym|)\s*głosowaniu\s*", "", name_part, flags=re.IGNORECASE).strip()
        if name_part: return name_part
    return normalized_title

def is_content_candidate_table(table_soup_element):
    if not isinstance(table_soup_element, Tag): return False # Ensure it's a Tag
    first_data_row = table_soup_element.select_one('tbody tr')
    if isinstance(first_data_row, Tag): # Ensure it's a Tag
        tds = first_data_row.find_all('td')
        if len(tds) >= 3: 
            candidate_link_check = tds[1].find('a')
            if isinstance(candidate_link_check, Tag): # Ensure it's a Tag
                href_attr = candidate_link_check.get('href','') 
                # Ensure href_attr is a string before calling startswith
                if isinstance(href_attr, str) and href_attr.startswith(f'{ELECTION_PATH_PREFIX}/kandydat/'):
                    lp_check_raw = normalize_text(tds[0].get_text())
                    votes_check_raw = normalize_text(tds[2].get_text())
                    if lp_check_raw.replace('.','').isdigit() and \
                       "razem" not in lp_check_raw.lower() and \
                       votes_check_raw.replace(' ', '').isdigit():
                        return True
    return False
def scrape_obkw_data_with_selenium(driver, obkw_url):
    candidate_rows = []
    summary_data = {"URL_ID": obkw_url.split('/')[-1]}
    commission_address_raw = "N/A"

    if not safe_driver_get(driver, obkw_url):
        return [], {}

    try:
        WebDriverWait(driver, 15).until(
            EC.any_of(
                EC.presence_of_element_located((By.XPATH, "//*[normalize-space(text())='Poszczególni kandydaci' or contains(normalize-space(text()),'Poszczególni kandydaci')]")),
                EC.presence_of_element_located((By.XPATH, "//h4[contains(normalize-space(.), 'USTALENIE WYNIKÓW GŁOSOWANIA')]")),
                EC.presence_of_element_located((By.CSS_SELECTOR, "table.table-striped"))
            )
        )
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')
        
        # For lambda in find: ensure text is treated as string-like, handle None
        address_dt = soup.find('dt', string=lambda text_content: text_content is not None and 'adres' in str(text_content).strip().lower())

        if isinstance(address_dt, Tag):
            dd_tag = address_dt.find_next_sibling('dd')
            if isinstance(dd_tag, Tag):
                commission_address_raw = normalize_text(dd_tag.get_text(separator=' ', strip=True))
        elif (h1_tag := soup.find('h1', class_='title')) and isinstance(h1_tag, Tag):
            commission_address_raw = extract_commission_name_from_title(normalize_text(h1_tag.get_text()))
        summary_data["Commission_Address_Raw"] = commission_address_raw
        
        summary_h_keywords = [
            normalize_text("ROZLICZENIE SPISU WYBORCÓW I KART DO GŁOSOWANIA").upper(),
            normalize_text("GŁOSOWANIE PRZEZ PEŁNOMOCNIKA").upper(),
            normalize_text("USTALENIE WYNIKÓW GŁOSOWANIA").upper()
        ]
        candidate_heading_keyword_norm = normalize_text("POSZCZEGÓLNI KANDYDACI").upper()
        all_striped_tables_on_page = soup.find_all('table', class_='table-striped')
        actual_candidate_table_soup = None
        candidate_heading_el = None
        possible_candidate_headings = soup.find_all(['p', 'h2', 'h3', 'h4', 'strong', 'b'])
        for el in possible_candidate_headings:
            if not isinstance(el, Tag): continue
            el_text_content = el.get_text(strip=True)
            el_text_normalized = normalize_text(el_text_content).upper()
            if candidate_heading_keyword_norm in el_text_normalized:
                is_parent_of_another_match = False
                for child_el in el.find_all(True, recursive=True): # el is Tag, so find_all is fine
                    if child_el is el: continue
                    if not isinstance(child_el, Tag): continue
                    if candidate_heading_keyword_norm in normalize_text(child_el.get_text(strip=True)).upper():
                        is_parent_of_another_match = True; break
                if not is_parent_of_another_match:
                    candidate_heading_el = el; break 
        if isinstance(candidate_heading_el, Tag):
            current_element_for_table_search = candidate_heading_el
            if candidate_heading_el.name in ['strong', 'b'] and isinstance(candidate_heading_el.parent, Tag) and candidate_heading_el.parent.name in ['p','h2','h3','h4']:
                current_element_for_table_search = candidate_heading_el.parent
            
            search_node = current_element_for_table_search
            while isinstance(search_node, Tag) and (search_node := search_node.find_next_sibling()): # Ensure search_node is a Tag
                if not isinstance(search_node, Tag): break # Reached end or found non-Tag
                table_to_check = None
                if search_node.name == 'table': table_to_check = search_node
                elif search_node.name == 'div' and isinstance(found_in_div := search_node.find('table'), Tag): table_to_check = found_in_div
                
                if isinstance(table_to_check, Tag) and 'table-striped' in table_to_check.get('class', []) and is_content_candidate_table(table_to_check):
                    actual_candidate_table_soup = table_to_check; break
                if search_node.name in ['h2', 'h3', 'h4'] or (search_node.name == 'p' and any(kw in normalize_text(search_node.get_text()).upper() for kw in summary_h_keywords)):
                    break
        if not isinstance(actual_candidate_table_soup, Tag): # check if it's still None or not a Tag
            for tbl in all_striped_tables_on_page:
                if isinstance(tbl, Tag) and is_content_candidate_table(tbl): 
                    actual_candidate_table_soup = tbl; break
        
        all_summary_tables_soup = []
        processed_table_objects = set() 
        if isinstance(actual_candidate_table_soup, Tag): processed_table_objects.add(actual_candidate_table_soup)
        
        summary_section_h4_tags = soup.find_all('h4')
        for h4_tag in summary_section_h4_tags:
            if not isinstance(h4_tag, Tag): continue
            h4_text_content = h4_tag.get_text(strip=True)
            h4_text_norm_upper = normalize_text(h4_text_content).upper()
            is_summary_h4 = any(keyword_norm in h4_text_norm_upper for keyword_norm in summary_h_keywords)
            if is_summary_h4:
                table_for_this_h4 = None
                next_element_after_h4 = h4_tag.find_next_sibling()
                if isinstance(next_element_after_h4, Tag):
                    potential_table = None
                    if next_element_after_h4.name == 'table' and 'table-striped' in next_element_after_h4.get('class',[]): potential_table = next_element_after_h4
                    elif next_element_after_h4.name == 'div' and isinstance(div_table := next_element_after_h4.find('table', class_='table-striped'), Tag): potential_table = div_table
                    
                    if isinstance(potential_table, Tag):
                        is_it_candidate_table_obj = (isinstance(actual_candidate_table_soup, Tag) and potential_table is actual_candidate_table_soup)
                        is_it_candidate_table_content = is_content_candidate_table(potential_table)
                        if not is_it_candidate_table_obj and not is_it_candidate_table_content:
                            table_for_this_h4 = potential_table
                if isinstance(table_for_this_h4, Tag) and table_for_this_h4 not in processed_table_objects:
                    all_summary_tables_soup.append(table_for_this_h4)
                    processed_table_objects.add(table_for_this_h4)
        
        if not all_summary_tables_soup and len(all_striped_tables_on_page) > 0 :
            for tbl_idx, tbl in enumerate(all_striped_tables_on_page):
                if not isinstance(tbl, Tag): continue
                is_it_candidate_table_obj = (isinstance(actual_candidate_table_soup, Tag) and tbl is actual_candidate_table_soup)
                is_it_candidate_table_content = is_content_candidate_table(tbl)
                if not is_it_candidate_table_obj and not is_it_candidate_table_content and tbl not in processed_table_objects: 
                     all_summary_tables_soup.append(tbl); processed_table_objects.add(tbl) 
        
        if all_summary_tables_soup:
            for field_key, item_no, search_terms in SUMMARY_FIELD_CONFIG:
                summary_data[field_key] = _get_value_from_summary_table_multi(all_summary_tables_soup, item_no, search_terms, field_key)
        else:
            print(f"    WARNING: NO SUMMARY TABLES were found to process data from for {obkw_url}.")
            for field_key, _, _ in SUMMARY_FIELD_CONFIG: summary_data[field_key] = None
        
        if isinstance(actual_candidate_table_soup, Tag):
            for row in actual_candidate_table_soup.select('tbody tr'):
                if not isinstance(row, Tag): continue
                cells = row.find_all('td')
                if not cells or len(cells) < 3 or "razem" in normalize_text(cells[0].get_text()).lower(): continue
                
                candidate_name_tag = cells[1].find('a')
                candidate_name = ""
                if isinstance(candidate_name_tag, Tag):
                    candidate_name = normalize_text(candidate_name_tag.get_text())
                elif isinstance(cells[1], Tag): # Fallback if <a> not found but <td> exists
                     candidate_name = normalize_text(cells[1].get_text())

                votes_str = normalize_text(cells[2].get_text())
                try:
                    votes = int(votes_str.replace(' ', '')) 
                    if candidate_name: candidate_rows.append({"URL_ID": summary_data["URL_ID"], "Candidate": candidate_name, "Votes": votes})
                except ValueError: print(f"      WARN: Could not parse votes '{votes_str}' for '{candidate_name}' on {obkw_url}")

    except TimeoutException: print(f"  Timeout on OBKW page {obkw_url}")
    except WebDriverException as e: 
        print(f"  WebDriverException (OBKW data) {obkw_url}: {e}")
        if "invalid session id" in str(e).lower() or "target frame detached" in str(e).lower(): raise 
    except Exception as e: 
        print(f"  Error processing OBKW page {obkw_url}: {e}")
        import traceback; traceback.print_exc()
    time.sleep(0.05) 
    return candidate_rows, summary_data

def determine_page_action(current_url, current_level_key):
    if current_level_key in ["Statki_Name", "Gmina_Name", "Zagranica_Okreg_Name"]: return "SCRAPE_OBKWS", None, None, None, OBKW_LIST_TABLE_ID
    if current_level_key == "Wojewodztwo_Name": return "GET_SUB_UNITS", "UL_COLUMNS2", POWIAT_MNPP_PATTERN, "Powiat_MnpP_Name", None
    if current_level_key == "Zagranica_Main_Name": return "GET_SUB_UNITS", "UL_COLUMNS2", PANSTWO_PATTERN, "Panstwo_Name", None
    if current_level_key == "Panstwo_Name": return "GET_SUB_UNITS", "TABLE_HIERARCHY", ZAGRANICA_OKREG_PATTERN, "Zagranica_Okreg_Name", HIERARCHY_TABLE_ID
    if current_level_key == "Powiat_MnpP_Name" and isinstance(current_url, str) and re.search(POWIAT_MNPP_PATTERN, current_url):
        return "GET_SUB_UNITS_UL_THEN_SCRAPE_OBKWS_FALLBACK", "UL_COLUMNS2", GMINA_PATTERN, "Gmina_Name", None
    return "UNKNOWN_TERMINAL", None, None, None, None

def process_unit(driver, current_url, unit_name_for_path, current_level_key, path_data_so_far, all_candidate_results_list, all_summary_results_list):
    current_path = {**path_data_so_far}
    if unit_name_for_path and current_level_key: current_path[current_level_key] = unit_name_for_path

    try:
        action_type, link_source_type, next_pattern, next_key, table_id = determine_page_action(current_url, current_level_key)
        if action_type == "SCRAPE_OBKWS":
            obkw_page_urls = get_obkw_links_from_page(driver, current_url)
            for obkw_u in obkw_page_urls:
                candidate_data_items, summary_data_item = scrape_obkw_data_with_selenium(driver, obkw_u)
                if summary_data_item and summary_data_item.get("URL_ID"): all_summary_results_list.append({**current_path, **summary_data_item})
                for cand_item in candidate_data_items: all_candidate_results_list.append({**current_path, **cand_item})
        elif action_type in ["GET_SUB_UNITS", "GET_SUB_UNITS_UL_THEN_SCRAPE_OBKWS_FALLBACK"]:
            sub_units = []
            if link_source_type == "UL_COLUMNS2": sub_units = get_links_from_ul_columns2(driver, current_url, next_pattern)
            elif link_source_type == "TABLE_HIERARCHY": sub_units = get_hierarchical_links_from_page(driver, current_url, next_pattern, table_id)
            
            if sub_units:
                for sub_name, sub_url in sub_units: 
                    process_unit(driver, sub_url, sub_name, next_key, current_path, all_candidate_results_list, all_summary_results_list)
            elif action_type == "GET_SUB_UNITS_UL_THEN_SCRAPE_OBKWS_FALLBACK": 
                obkw_page_urls = get_obkw_links_from_page(driver, current_url)
                for obkw_u in obkw_page_urls:
                    candidate_data_items, summary_data_item = scrape_obkw_data_with_selenium(driver, obkw_u)
                    if summary_data_item and summary_data_item.get("URL_ID"): all_summary_results_list.append({**current_path, **summary_data_item})
                    for cand_item in candidate_data_items: all_candidate_results_list.append({**current_path, **cand_item})
        elif action_type == "UNKNOWN_TERMINAL":
            obkw_page_urls = get_obkw_links_from_page(driver, current_url) 
            for obkw_u in obkw_page_urls:
                candidate_data_items, summary_data_item = scrape_obkw_data_with_selenium(driver, obkw_u)
                if summary_data_item and summary_data_item.get("URL_ID"): all_summary_results_list.append({**current_path, **summary_data_item})
                for cand_item in candidate_data_items: all_candidate_results_list.append({**current_path, **cand_item})
    except WebDriverException as e_session: print(f"  WebDriver session error (process_unit) {current_url}: {e_session}"); raise
    except Exception as e_proc: print(f"  Unexpected error in process_unit for {current_url}: {e_proc}"); import traceback; traceback.print_exc()


def worker_task(config_item):
    driver = None
    item_name = config_item['name']
    item_url = config_item['url']
    item_level_key = config_item['level_key']
    item_path_name = config_item.get('path_name_override', item_name)
    
    worker_pid = os.getpid()
    print(f"WORKER {worker_pid}: Starting scrape for {item_name} ({item_url})")
    
    candidate_data_list_worker = []
    summary_data_list_worker = []
    initial_path_data = {} 

    try:
        driver = get_webdriver() 
        process_unit(driver, item_url, item_path_name, item_level_key, 
                     initial_path_data, candidate_data_list_worker, summary_data_list_worker)
        print(f"WORKER {worker_pid}: Finished {item_name}. Candidates: {len(candidate_data_list_worker)}, Summaries: {len(summary_data_list_worker)}")
    except Exception as e:
        print(f"WORKER {worker_pid}: ERROR processing {item_name}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if driver:
            try:
                driver.quit()
            except Exception as e_quit:
                print(f"WORKER {worker_pid}: Error quitting WebDriver for {item_name}: {e_quit}")
    
    return candidate_data_list_worker, summary_data_list_worker


def main_parallel_batched():
    print("Starting PARALLEL BATCHED national election data scraper for TURA 2...")
    start_time = time.time()
    if not os.path.exists("data"): os.makedirs("data")

    all_configurations_to_process = []
    for woj_name, woj_url in WOJEWODZTWA_CONFIG:
        all_configurations_to_process.append({'name': woj_name, 'url': woj_url, 'level_key': 'Wojewodztwo_Name', 'path_name_override': woj_name})
    
    if isinstance(ZAGRANICA_CONFIG, dict) and all(k in ZAGRANICA_CONFIG for k in ['name', 'url', 'level_key']):
        all_configurations_to_process.append(ZAGRANICA_CONFIG)
    else:
        print(f"Warning: ZAGRANICA_CONFIG is not in the expected format: {ZAGRANICA_CONFIG}")

    if isinstance(STATKI_CONFIG, dict) and all(k in STATKI_CONFIG for k in ['name', 'url', 'level_key']):
         all_configurations_to_process.append(STATKI_CONFIG)
    else:
        print(f"Warning: STATKI_CONFIG is not in the expected format: {STATKI_CONFIG}")

    
    BATCH_SIZE = 4
    print(f"Total configurations to process: {len(all_configurations_to_process)}")
    print(f"Processing in batches of {BATCH_SIZE} parallel workers.")

    overall_candidate_data = []
    overall_summary_data = []

    for i in range(0, len(all_configurations_to_process), BATCH_SIZE):
        batch_configs = all_configurations_to_process[i:i + BATCH_SIZE]
        print(f"\n--- Starting Batch {i//BATCH_SIZE + 1} ---")
        for cfg in batch_configs: print(f"  - Processing: {cfg['name']}")

        current_batch_pool_size = min(len(batch_configs), BATCH_SIZE)
        with Pool(processes=current_batch_pool_size) as pool:
            batch_results = pool.map(worker_task, batch_configs)
        
        print(f"--- Finished Batch {i//BATCH_SIZE + 1} ---")

        for res_cand_list, res_summ_list in batch_results:
            if res_cand_list:
                overall_candidate_data.extend(res_cand_list)
            if res_summ_list:
                overall_summary_data.extend(res_summ_list)
        
        if i + BATCH_SIZE < len(all_configurations_to_process):
            print(f"Pausing for 10 seconds before next batch...")
            time.sleep(10) 

    total_time = time.time() - start_time
    print(f"\n--- All parallel batches completed in {total_time:.2f} seconds. ---")
    print(f"Total candidate records scraped: {len(overall_candidate_data)}")
    print(f"Total summary records (OBKWs) scraped: {len(overall_summary_data)}")

    if not overall_candidate_data: 
        print("\nNo CANDIDATE data was successfully scraped overall.")
    else:
        df_candidates = pd.DataFrame(overall_candidate_data)
        candidate_column_order = ['Wojewodztwo_Name', 'Powiat_MnpP_Name', 'Gmina_Name', 
                                  'Zagranica_Main_Name', 'Panstwo_Name', 'Zagranica_Okreg_Name', 'Statki_Name', 
                                  'URL_ID', 'Candidate', 'Votes']
        df_cand_cols = [col for col in candidate_column_order if col in df_candidates.columns]
        if df_cand_cols: df_candidates = df_candidates.reindex(columns=df_cand_cols)
        output_cand_filename = "data/polska_prezydent2025_tura2_obkw_kandydaci_NATIONAL_FINAL.csv"
        df_candidates.to_csv(output_cand_filename, index=False, encoding='utf-8-sig')
        print(f"\nCandidate data saved to {output_cand_filename}")

    if not overall_summary_data: 
        print("\nNo SUMMARY data was successfully scraped overall.")
    else:
        df_summary = pd.DataFrame(overall_summary_data)
        summary_column_order_base = ['Wojewodztwo_Name', 'Powiat_MnpP_Name', 'Gmina_Name', 
                                     'Zagranica_Main_Name', 'Panstwo_Name', 'Zagranica_Okreg_Name', 'Statki_Name', 
                                     'URL_ID', 'Commission_Address_Raw']
        summary_field_keys_from_config = [config_item[0] for config_item in SUMMARY_FIELD_CONFIG]
        final_summary_column_order = summary_column_order_base + summary_field_keys_from_config
        df_summ_cols = [col for col in final_summary_column_order if col in df_summary.columns]
        if df_summ_cols: df_summary = df_summary.reindex(columns=df_summ_cols)
        output_summary_filename = "data/polska_prezydent2025_tura2_obkw_podsumowanie_NATIONAL_FINAL.csv"
        df_summary.to_csv(output_summary_filename, index=False, encoding='utf-8-sig')
        print(f"\nSummary data saved to {output_summary_filename}")
        if len(df_summary) > 0: 
            print(f"Data from {df_summary['URL_ID'].nunique()} unique OBKWs (summary).")
            print("\nSample of final summary data (first 5 rows):")
            with pd.option_context('display.max_columns', None, 'display.width', 1000): 
                 print(df_summary.head())

if __name__ == "__main__":
    main_parallel_batched()